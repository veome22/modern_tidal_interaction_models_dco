import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

from multiprocessing import Pool, cpu_count
from multiprocessing import get_context
import concurrent.futures
import psutil

import sys

import h5py as h5 
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck18

import os
import scipy
from collections import Counter
from collections import defaultdict
import gc
import copy

from sklearn.utils import resample
from scipy.stats import gaussian_kde, norm


pd.options.display.max_columns = None

# %%
from gwfast          import gwfastGlobals as glob
from gwfast.waveforms import IMRPhenomD
from gwfast.signal    import GWSignal
from gwfast.network   import DetNet
from gwfast.fisherTools import CovMatr


# %%
pop_names = ['realistic']

pop_paths = {
            'notides': ['../pop_sims/notides_pop_v329_combined.h5'],
            'realistic': ['../pop_sims/realistic_pop_v329_combined.h5'],
            'perfect': ['../pop_sims/perfect_pop_v329_combined.h5'] 
            }

pop_labels = {'notides': 'LEGACY',
              'realistic': 'K26',
              'perfect': 'PERFECT'}


data_path = '/home/vkapil1/scratch16-berti/tides_compas_veome/modern_tidal_interaction_models_dco_applications/data_files/'



# %% [markdown]
# # Detector Network Definitions
# ─────────────────────────────────────────────────────────────────────────────
# To add a new network:
#   1. Add an entry to `network_psd_specs` mapping det_name → psd_path
# ─────────────────────────────────────────────────────────────────────────────

network_psd_specs = {

    # # ── O4a: H1 + L1 + Virgo ─────────────────────────────────────────────
    # 'O4a': {
    #     'H1':    os.path.join(glob.detPath, 'observing_scenarios_paper', 'aligo_O4high.txt'),
    #     'L1':    os.path.join(glob.detPath, 'observing_scenarios_paper', 'aligo_O4high.txt'),
    #     'Virgo': os.path.join(glob.detPath, 'observing_scenarios_paper', 'avirgo_O4high_NEW.txt'),
    # },

    # ── 3G: ET (triangular, 15 km) + CE1 + A+ ────────────────────────────
    '3G': {
        'ETSL':    os.path.join(glob.detPath, 'ET_designs_comparison_paper/HFLF_cryo', 'ETLength15km.txt'),
        'ETMRL45d':os.path.join(glob.detPath, 'ET_designs_comparison_paper/HFLF_cryo', 'ETLength15km.txt'),
        'CE1Id':   os.path.join(glob.detPath, 'ce_strain', 'cosmic_explorer.txt'),
        #'LIGOI':   os.path.join(glob.detPath, 'observing_scenarios_paper', 'AplusDesign.txt'),
    },

    # ── Add more networks here ────────────────────────────────────────────
    # 'MyNet': {
    #     'H1': '/path/to/psd.txt',
    #     ...
    # },
}

SNR_THRESHOLD = 8   # detection threshold applied for fisher


# %% [markdown]
# # Network Builder
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr(net: DetNet, events: dict, res: int = 1000) -> np.ndarray:
    """Compute network SNR for all events; returns array of shape (N,)."""
    return net.SNR(events, res=res)


def save_network_results(filepath, net_name, events, snr, cov, fim=None,
                         snr_threshold=None, metadata=None):
    """
    Save one network's results to its own HDF5 file.

    File layout
    ───────────
    /events/
        <param_name>  (N,)
    /snr              (N,)
    /cov              (n_params, n_params, N_det)
    /fim              (n_params, n_params, N_det)   ← optional
    /det_indices      (N_det,)
    /metadata/        ← attrs: net_name, snr_threshold, + any extras
    """
    snr = np.asarray(snr, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    det_indices = np.where(snr > (snr_threshold or 0))[0]

    with h5.File(filepath, "w") as f:
        # ── events ──────────────────────────────────────────────────────
        grp_ev = f.create_group("events")
        for param, arr in events.items():
            grp_ev.create_dataset(param, data=np.asarray(arr, dtype=np.float64))

        # ── results ─────────────────────────────────────────────────────
        f.create_dataset("snr",         data=snr)
        f.create_dataset("cov",         data=cov)
        f.create_dataset("det_indices", data=det_indices)
        if fim is not None:
            f.create_dataset("fim", data=np.asarray(fim, dtype=np.float64))

        # ── metadata ────────────────────────────────────────────────────
        grp_meta = f.create_group("metadata")
        grp_meta.create_dataset(
            "param_names",
            data=np.array(list(events.keys()), dtype=h5.special_dtype(vlen=str))
        )
        grp_meta.attrs["network_name"] = net_name
        if snr_threshold is not None:
            grp_meta.attrs["snr_threshold"] = float(snr_threshold)
        if metadata:
            for k, v in metadata.items():
                grp_meta.attrs[k] = v

    print(f"  Saved → {filepath}")  



if __name__ == '__main__':
    import subprocess
    import time

    n_workers  = int(os.environ.get("GWFAST_N_WORKERS",  48))
    batch_size = int(os.environ.get("GWFAST_BATCH_SIZE", 25))
    res        = int(os.environ.get("GWFAST_RES",        500))

    wf      = IMRPhenomD()
    wf_name = 'IMRPhenomD'

    # ── Data loading ─────────────────────────────────────────────────────────
    pop_spin_dfs = {}
    for pop_name in pop_names:
        pop_spin_dfs[pop_name] = pd.read_csv(data_path + pop_name + '_spin_df.csv')

    first_pop = pop_spin_dfs[pop_names[0]]
    bbh    = first_pop[first_pop['BBH'] & first_pop['Merges_Hubble_Time']].copy()
    sample = bbh[bbh['z_merger'].notna()].copy()
    N      = len(sample)
    print(f"Processing {N} BBHs")

    rng   = np.random.default_rng(42)
    m1    = sample['Mass@DCO(1)'].values
    m2    = sample['Mass@DCO(2)'].values
    z     = sample['z_merger'].values
    chi1z = sample['a1'].values  * np.cos(sample['iota1'].values)
    chi2z = sample['a2'].values  * np.cos(sample['iota2'].values)
    Mt    = (m1 + m2) * (1 + z)
    eta   = (m1 * m2) / (m1 + m2)**2
    Mc    = Mt * eta**0.6
    dL    = Planck18.luminosity_distance(z).to(u.Gpc).value

    events = {
        'Mc'     : Mc,           'eta'    : eta,
        'chi1z'  : chi1z,        'chi2z'  : chi2z,
        'chi1x'  : np.zeros(N),  'chi1y'  : np.zeros(N),
        'chi2x'  : np.zeros(N),  'chi2y'  : np.zeros(N),
        'dL'     : dL,
        'iota'   : np.arccos(rng.uniform(-1, 1, N)),
        'psi'    : rng.uniform(0, 2*np.pi, N),
        'ra'     : rng.uniform(0, 2*np.pi, N),
        'dec'    : np.arcsin(rng.uniform(-1, 1, N)),
        'tcoal'  : rng.uniform(0, 1, N),
        'Phicoal': rng.uniform(0, 2*np.pi, N),
    }

    for net_name, psd_spec in network_psd_specs.items():

        batch_dir = f"{data_path}{net_name}_parallel_batches/"
        snr_cache = f"{data_path}{net_name}_snr.npy"
        log_path = f"{data_path}/log_files/"
        os.makedirs(batch_dir, exist_ok=True)
        os.makedirs(log_path, exist_ok=True)

        print(f"\n─── Network: {net_name} ───")

        # ── Save events for workers to load ──────────────────────────────────
        events_path = f"{data_path}{net_name}_events.npz"
        if not os.path.exists(events_path):
            np.savez(events_path, **events)
            print(f"  Events saved → {events_path}")

        # ── Save PSD spec for workers to load ────────────────────────────────
        psd_spec_path = f"{data_path}{net_name}_psd_spec.npz"
        if not os.path.exists(psd_spec_path):
            np.savez(psd_spec_path, **psd_spec)
            print(f"  PSD spec saved → {psd_spec_path}")

        # ── SNR ──────────────────────────────────────────────────────────────
        if not os.path.exists(snr_cache):
            print("  Computing SNR...")
            net = build_network(psd_spec, wf)
            snr = compute_snr(net, events)
            np.save(snr_cache, snr)
            print(f"  SNR cached → {snr_cache}")
        else:
            snr = np.load(snr_cache)
            print(f"  Loaded cached SNR")

        det_mask = snr > SNR_THRESHOLD
        N_det    = det_mask.sum()
        n_batch  = int(np.ceil(N_det / batch_size))
        print(f"  Detectable: {N_det} / {N} ({100*N_det/N:.1f}%)")
        print(f"  Total batches: {n_batch} | workers: {n_workers} | "
              f"batch_size: {batch_size}")

        # ── Check which batches are already done ─────────────────────────────
        done = [
            i for i in range(n_batch)
            if os.path.exists(os.path.join(batch_dir, f"batch_{i:04d}.npy"))
        ]
        print(f"  Already complete: {len(done)}/{n_batch} batches")

        if len(done) == n_batch:
            print("  All batches done — skipping to assembly.")
        else:
            # ── Launch one subprocess per worker ─────────────────────────────
            procs = []
            for task_id in range(n_workers):
                cmd = [
                    sys.executable, "fisher_worker.py",
                    net_name,
                    str(task_id),
                    str(n_workers),
                    str(batch_size),
                    str(res),
                ]
                log_out = open(f"{log_path}{net_name}_worker{task_id}.log", "w")
                p = subprocess.Popen(cmd, stdout=log_out, stderr=log_out)
                procs.append((task_id, p, log_out))
                print(f"  Launched worker {task_id}  (pid {p.pid})", flush=True)

            # ── Monitor until all workers finish ─────────────────────────────
            print("  Monitoring workers...")
            while True:
                time.sleep(10)
                done_now = [
                    i for i in range(n_batch)
                    if os.path.exists(os.path.join(batch_dir, f"batch_{i:04d}.npy"))
                ]
                still_running = [(tid, p, f) for tid, p, f in procs
                                 if p.poll() is None]
                failed        = [(tid, p, f) for tid, p, f in procs
                                 if p.poll() not in (None, 0)]

                print(f"  [{time.strftime('%H:%M:%S')}] "
                      f"batches: {len(done_now)}/{n_batch} | "
                      f"workers running: {len(still_running)} | "
                      f"failed: {len(failed)}",
                      flush=True)

                if failed:
                    for tid, p, f in failed:
                        print(f"  ✗ worker {tid} exited with code {p.returncode} "
                              f"— check data_files//log_files/{net_name}_worker{tid}.log")

                if not still_running:
                    for _, _, f in procs:
                        f.close()
                    break

            # ── Final gap check ───────────────────────────────────────────────
            missing = [
                i for i in range(n_batch)
                if not os.path.exists(os.path.join(batch_dir, f"batch_{i:04d}.npy"))
            ]
            if missing:
                print(f"\n  ✗ {len(missing)} batches still missing after all "
                      f"workers finished: {missing[:20]}"
                      f"{'...' if len(missing) > 20 else ''}")
                print(f"  Re-run the script — missing batches will be picked "
                      f"up automatically on next run.")
                continue     # skip assembly for this network
            else:
                print(f"  ✓ All {n_batch} batches complete.")

        # ── Assemble ─────────────────────────────────────────────────────────
        print("  Assembling FIM from batches...")
        fim_batches = [
            np.load(os.path.join(batch_dir, f"batch_{i:04d}.npy"))
            for i in range(n_batch)
        ]
        FIM = np.concatenate(fim_batches, axis=-1)
        del fim_batches
        gc.collect()
        print(f"  FIM shape: {FIM.shape}")

        cov, inv_err = CovMatr(FIM)
        print("  Covariance computed.")

        save_network_results(
            filepath      = f"{data_path}{net_name}_gwfast.h5",
            net_name      = net_name,
            events        = events,
            snr           = snr,
            cov           = cov,
            fim           = FIM,
            snr_threshold = SNR_THRESHOLD,
            metadata      = {"waveform": wf_name, "N": N},
        )

        del FIM, cov
        gc.collect()
        print(f"  ✓ Saved → data_files/{net_name}_gwfast.h5")
