"""
combine_fim.py
--------------
Assembles per-batch FIM files, inverts them vectorially, and writes a single
HDF5 output file per network.

Usage
-----
    python combine_fim.py                        # all networks in network_psd_specs
    python combine_fim.py --networks 3G O4a      # specific networks
    python combine_fim.py --networks 3G --dry-run # check completeness only
"""

import argparse
import gc
import os
import sys
import warnings

import h5py as h5
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — mirror what compute_fisher_gw.py uses
# ─────────────────────────────────────────────────────────────────────────────

# DATA_PATH     = "scratch16-berti/tides_compas_veome/modern_tidal_interaction_models_dco_applications/data_files/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "..", "data_files") + os.sep
SNR_THRESHOLD = 8
BATCH_SIZE    = 25      # must match what was used during compute


# ─────────────────────────────────────────────────────────────────────────────
# Network definitions — keep in sync with compute_fisher_gw.py
# ─────────────────────────────────────────────────────────────────────────────

from gwfast import gwfastGlobals as glob

NETWORK_PSD_SPECS = {
    '3G': {
        'ETSL'    : os.path.join(glob.detPath, 'ET_designs_comparison_paper/HFLF_cryo', 'ETLength15km.txt'),
        'ETMRL45d': os.path.join(glob.detPath, 'ET_designs_comparison_paper/HFLF_cryo', 'ETLength15km.txt'),
        'CE1Id'   : os.path.join(glob.detPath, 'ce_strain', 'cosmic_explorer.txt'),
    },
    # add more networks here as needed
}


# ─────────────────────────────────────────────────────────────────────────────
# Covariance inversion
# ─────────────────────────────────────────────────────────────────────────────

def compute_cov_vectorized(FIM, rcond=1e-15, verbose=True):
    """
    Invert a stack of Fisher matrices → covariance matrices.

    Parameters
    ----------
    FIM     : (n_params, n_params, N_det)
    rcond   : singular value threshold for pseudoinverse
    verbose : print a single summary line

    Returns
    -------
    cov     : (n_params, n_params, N_det)
    inv_err : (N_det,) bool — True where matrix was ill-conditioned
    """
    # gwfast axis convention: (n_params, n_params, N_det) → (N_det, n, n)
    FIM_T = np.moveaxis(FIM, -1, 0)

    # Symmetrize: finite-difference derivatives can break exact symmetry
    FIM_T = 0.5 * (FIM_T + np.swapaxes(FIM_T, -1, -2))

    # Vectorized pseudoinverse — one LAPACK call for the full stack
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov_T = np.linalg.pinv(FIM_T, rcond=rcond, hermitian=True)

    # Condition number check (one SVD call, no eigenvectors needed)
    sv   = np.linalg.svd(FIM_T, compute_uv=False)          # (N_det, n_params)
    s_min = np.where(sv[..., -1] == 0, np.inf, sv[..., -1])
    cond  = sv[..., 0] / s_min
    inv_err = cond > (1.0 / rcond)

    if verbose:
        n_bad = int(inv_err.sum())
        n_tot = len(inv_err)
        if n_bad:
            print(f"    ill-conditioned: {n_bad}/{n_tot} matrices "
                  f"(cond > {1/rcond:.0e}) — SVD pseudoinverse used for all")
        else:
            print(f"    all {n_tot} matrices well-conditioned")

    cov = np.moveaxis(cov_T, 0, -1)                        # back to gwfast convention
    return cov, inv_err


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 writer
# ─────────────────────────────────────────────────────────────────────────────

def save_results(filepath, net_name, events, snr, cov, fim,
                 inv_err, param_names, snr_threshold, metadata=None):
    """
    Write assembled results to HDF5.

    Layout
    ------
    /events/<param>     (N,)               full injection set
    /snr                (N,)               full SNR array
    /det_indices        (N_det,)           indices into /events/ for det. subset
    /fim                (n_p, n_p, N_det)
    /cov                (n_p, n_p, N_det)
    /inv_err            (N_det,)           bool: ill-conditioned FIM flag
    /metadata/
        param_names     variable-length strings
        attrs: network_name, snr_threshold, + anything in `metadata`
    """
    det_indices = np.where(snr > snr_threshold)[0]

    with h5.File(filepath, "w") as f:

        # events
        grp_ev = f.create_group("events")
        for param, arr in events.items():
            grp_ev.create_dataset(param, data=np.asarray(arr, dtype=np.float64),
                                  compression="gzip", compression_opts=4)

        # SNR + index
        f.create_dataset("snr",         data=np.asarray(snr,         dtype=np.float64))
        f.create_dataset("det_indices", data=np.asarray(det_indices, dtype=np.int64))

        # matrices — compressed: FIM/cov can be large
        f.create_dataset("fim",     data=np.asarray(fim,     dtype=np.float64),
                         compression="gzip", compression_opts=4)
        f.create_dataset("cov",     data=np.asarray(cov,     dtype=np.float64),
                         compression="gzip", compression_opts=4)
        f.create_dataset("inv_err", data=np.asarray(inv_err, dtype=bool))

        # metadata
        grp_meta = f.create_group("metadata")
        dt = h5.special_dtype(vlen=str)
        grp_meta.create_dataset("param_names",
                                data=np.array(param_names, dtype=dt))
        grp_meta.attrs["network_name"]  = net_name
        grp_meta.attrs["snr_threshold"] = float(snr_threshold)
        if metadata:
            for k, v in metadata.items():
                grp_meta.attrs[k] = v

    size_mb = os.path.getsize(filepath) / 1e6
    print(f"    saved → {filepath}  ({size_mb:.1f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# Per-network assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_network(net_name, batch_size=BATCH_SIZE, rcond=1e-15,
                     dry_run=False, overwrite=False):
    """
    Full pipeline for one network:
        load SNR + events → check batches → assemble FIM → invert → save HDF5.

    Returns True on success, False if batches are incomplete.
    """
    batch_dir  = os.path.join(DATA_PATH, f"{net_name}_parallel_batches")
    snr_cache  = os.path.join(DATA_PATH, f"{net_name}_snr.npy")
    events_npz = os.path.join(DATA_PATH, f"{net_name}_events.npz")
    output_h5  = os.path.join(DATA_PATH, f"{net_name}_gwfast.h5")

    print(f"\n{'─'*60}")
    print(f"  Network : {net_name}")
    print(f"{'─'*60}")

    # ── sanity-check required inputs ─────────────────────────────────────────
    for path, label in [(snr_cache,  "SNR cache"),
                        (events_npz, "events npz"),
                        (batch_dir,  "batch directory")]:
        if not os.path.exists(path):
            print(f"  ✗ {label} not found: {path}")
            print(f"    Run compute_fisher_gw.py for this network first.")
            return False

    # ── load SNR + events ────────────────────────────────────────────────────
    snr         = np.load(snr_cache)
    events_raw  = np.load(events_npz)
    events      = dict(events_raw)
    param_names = list(events.keys())

    det_mask    = snr > SNR_THRESHOLD
    N_det       = int(det_mask.sum())
    N           = len(snr)
    n_batch     = int(np.ceil(N_det / batch_size))

    print(f"  Events      : {N_det} detectable / {N} total")
    print(f"  Batches     : {n_batch}  (batch_size={batch_size})")

    # ── check completeness ───────────────────────────────────────────────────
    missing = [
        i for i in range(n_batch)
        if not os.path.exists(os.path.join(batch_dir, f"batch_{i:04d}.npy"))
    ]

    if missing:
        print(f"  ✗ {len(missing)}/{n_batch} batches missing:")
        # print at most 20 indices, then summarise
        preview = missing[:20]
        print(f"    indices: {preview}{'...' if len(missing) > 20 else ''}")
        print(f"    Re-run compute_fisher_gw.py to fill gaps.")
        return False

    print(f"  ✓ all {n_batch} batch files present")

    if dry_run:
        print("  dry-run: skipping assembly.")
        return True

    if os.path.exists(output_h5) and not overwrite:
        print(f"  ✗ output already exists: {output_h5}")
        print(f"    pass --overwrite to replace it.")
        return False

    # ── load batch shape from first file to get n_params ────────────────────
    first_batch = np.load(os.path.join(batch_dir, "batch_0000.npy"))
    n_params    = first_batch.shape[0]
    print(f"  Parameters  : {n_params}  → {param_names}")
    del first_batch

    # ── assemble FIM ─────────────────────────────────────────────────────────
    print(f"  Assembling FIM ({n_params}×{n_params}×{N_det})...")
    FIM = np.empty((n_params, n_params, N_det), dtype=np.float64)

    for i in range(n_batch):
        lo, hi  = i * batch_size, min((i + 1) * batch_size, N_det)
        batch_path = os.path.join(batch_dir, f"batch_{i:04d}.npy")
        FIM[:, :, lo:hi] = np.load(batch_path)

        if (i + 1) % max(1, n_batch // 10) == 0 or (i + 1) == n_batch:
            print(f"    loaded {i+1}/{n_batch} batches", flush=True)

    print(f"  FIM assembled — shape {FIM.shape}")

    # ── invert ───────────────────────────────────────────────────────────────
    print("  Inverting FIM → covariance...")
    cov, inv_err = compute_cov_vectorized(FIM, rcond=1e-15, verbose=True)

    # ── save ─────────────────────────────────────────────────────────────────
    print("  Writing HDF5...")
    save_results(
        filepath      = output_h5,
        net_name      = net_name,
        events        = events,
        snr           = snr,
        cov           = cov,
        fim           = FIM,
        inv_err       = inv_err,
        param_names   = param_names,
        snr_threshold = SNR_THRESHOLD,
        metadata      = {"batch_size": batch_size, "n_batches": n_batch},
    )

    del FIM, cov, inv_err
    gc.collect()

    return True


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Assemble batch FIMs, invert, and write HDF5."
    )
    parser.add_argument(
        "--networks", nargs="+", default=None,
        help="Network names to process (default: all in NETWORK_PSD_SPECS)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check batch completeness only, do not assemble or write"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing HDF5 output files"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size used during compute (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--rcond", type=float, default=1e-15,
        help="Singular value threshold for pseudoinverse (default: 1e-15)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    networks = args.networks or list(NETWORK_PSD_SPECS.keys())

    print(f"Networks to process : {networks}")
    print(f"Dry run             : {args.dry_run}")
    print(f"Overwrite           : {args.overwrite}")
    print(f"Batch size          : {args.batch_size}")
    print(f"rcond               : {args.rcond}")

    results = {}
    for net_name in networks:
        if net_name not in NETWORK_PSD_SPECS:
            print(f"\n✗ Unknown network '{net_name}' — "
                  f"add it to NETWORK_PSD_SPECS in this script.")
            results[net_name] = False
            continue

        results[net_name] = assemble_network(
            net_name   = net_name,
            batch_size = args.batch_size,
            rcond      = args.rcond,
            dry_run    = args.dry_run,
            overwrite  = args.overwrite,
        )

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Summary")
    print(f"{'─'*60}")
    for net_name, ok in results.items():
        status = "✓ done" if ok else "✗ failed / incomplete"
        print(f"  {net_name:<20} {status}")
