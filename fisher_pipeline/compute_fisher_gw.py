# %%
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None 

from multiprocessing import Pool, cpu_count
from multiprocessing import get_context
import concurrent.futures
import psutil

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

from fim_parallel import compute_fim_parallel

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


data_path = '/home/vkapil1/scratch16-berti/tides_compas_veome/modern_tidal_interaction_models_dco_applications/data_files'



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

def build_network(psd_spec: dict, waveform, fmin: float = 10.) -> DetNet:
    """
    Build a gwfast DetNet from a dict of {det_name: psd_path}.

    Parameters
    ----------
    psd_spec  : dict  {detector_name: path_to_psd_file}
    waveform  : gwfast waveform object  (e.g. IMRPhenomD())
    fmin      : float  lower frequency cutoff in Hz

    Returns
    -------
    DetNet instance ready to call .SNR() on
    """
    signals = {}
    for det, psd_path in psd_spec.items():
        det_cfg = copy.deepcopy(glob.detectors[det])
        det_cfg['psd_path'] = psd_path
        signals[det] = GWSignal(
            waveform,
            psd_path       = det_cfg['psd_path'],
            detector_shape = det_cfg['shape'],
            det_lat        = det_cfg['lat'],
            det_long       = det_cfg['long'],
            det_xax        = det_cfg['xax'],
            verbose        = True,
            useEarthMotion = False,
            fmin           = fmin,
            IntTablePath   = None,
        )
    return DetNet(signals, verbose=True)


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

    batch_size = int(os.environ.get("GWFAST_BATCH_SIZE", 25))
    wf      = IMRPhenomD()
    wf_name = 'IMRPhenomD'

    # ── Data loading ─────────────────────────────────────────────────────────
    pop_spin_dfs = {}
    for pop_name in pop_names:
        pop_spin_dfs[pop_name] = pd.read_csv(data_path + '/' + pop_name + '_spin_df.csv')

    first_pop = pop_spin_dfs[pop_names[0]]
    bbh    = first_pop[first_pop['BBH'] & first_pop['Merges_Hubble_Time']].copy()
    sample = bbh[bbh['z_merger'].notna()].copy()
    N      = len(sample)
    print(f"Processing {len(sample)} BBHs")

    rng = np.random.default_rng(42)
    m1  = sample['Mass@DCO(1)'].values
    m2  = sample['Mass@DCO(2)'].values
    z   = sample['z_merger'].values

    chi1z = sample['a1'].values * np.cos(sample['iota1'].values)
    chi2z = sample['a2'].values * np.cos(sample['iota2'].values)

    Mt  = (m1 + m2) * (1 + z)
    eta = (m1 * m2) / (m1 + m2)**2
    Mc  = Mt * eta**0.6
    dL  = Planck18.luminosity_distance(z).to(u.Gpc).value

    events = {
        'Mc'     : Mc,
        'eta'    : eta,
        'chi1z'  : chi1z,
        'chi2z'  : chi2z,
        'chi1x'  : np.zeros(N),
        'chi1y'  : np.zeros(N),
        'chi2x'  : np.zeros(N),
        'chi2y'  : np.zeros(N),
        'dL'     : dL,
        'iota'   : np.arccos(rng.uniform(-1, 1, N)),
        'psi'    : rng.uniform(0, 2*np.pi,   N),
        'ra'     : rng.uniform(0, 2*np.pi,   N),
        'dec'    : np.arcsin(rng.uniform(-1, 1, N)),
        'tcoal'  : rng.uniform(0, 1, N),
        'Phicoal': rng.uniform(0, 2*np.pi, N),
    }

    for net_name, psd_spec in network_psd_specs.items():

        output_path = f"{data_path}/{net_name}_gwfast.h5"
        batch_index_path = f"{data_path}/{net_name}_batches/"
        os.makedirs(batch_index_path, exist_ok=True)

        print(f"\n─── Network: {net_name} ───")

        # ── Check if SNR already computed ────────────────────────────────────
        snr_cache = f"{data_path}/{net_name}_snr.npy"
        if os.path.exists(snr_cache):
            print(f"  Loading cached SNR from {snr_cache}")
            snr = np.load(snr_cache)
        else:
            print("  Computing SNR...")
            net = build_network(psd_spec, wf)
            snr = compute_snr(net, events)
            np.save(snr_cache, snr)
            print(f"  SNR computed and cached → {snr_cache}")

        n_det = (snr > SNR_THRESHOLD).sum()
        N_det = n_det
        print(f"  Detectable (SNR > {SNR_THRESHOLD}): {n_det} / {N} ({100*n_det/N:.1f}%)")
        print(f"  Median SNR (all):       {np.median(snr):.2f}")
        print(f"  Median SNR (det. only): {np.median(snr[snr > SNR_THRESHOLD]):.2f}")

        det_mask   = snr > SNR_THRESHOLD
        det_events = {k: v[det_mask] for k, v in events.items()}
        n_batch    = int(np.ceil(N_det / batch_size))

        # ── Build network only if we need to compute any batches ─────────────
        completed_batches = set(
            int(f.split('_')[1].split('.')[0])
            for f in os.listdir(batch_index_path)
            if f.startswith('batch_') and f.endswith('.npy')
        )
        remaining = [i for i in range(n_batch) if i not in completed_batches]

        if remaining:
            print(f"  {len(completed_batches)}/{n_batch} batches already done. "
                  f"Computing {len(remaining)} remaining...")
            # Only build network if there's work to do
            if 'net' not in dir() or net is None:
                net = build_network(psd_spec, wf)
        else:
            print(f"  All {n_batch} batches already computed. Skipping FIM.")

        # ── Batched FIM with resume ───────────────────────────────────────────
        for i in remaining:
            lo, hi  = i * batch_size, min((i + 1) * batch_size, N_det)
            batch   = {k: v[lo:hi] for k, v in det_events.items()}
            batch_path = os.path.join(batch_index_path, f"batch_{i:04d}.npy")

            print(f"  Batch {i+1}/{n_batch}  (events {lo}–{hi-1})", flush=True)
            fim_batch = net.FisherMatr(batch, res=500)
            np.save(batch_path, fim_batch)
            print(f"    Saved → {batch_path}", flush=True)

            del batch, fim_batch
            gc.collect()

        # ── Assemble FIM from saved batches ───────────────────────────────────
        print("  Assembling FIM from batches...")
        fim_batches = [
            np.load(os.path.join(batch_index_path, f"batch_{i:04d}.npy"))
            for i in range(n_batch)
        ]
        FIM = np.concatenate(fim_batches, axis=-1)
        del fim_batches
        gc.collect()
        print(f"  FIM shape: {FIM.shape}")

        # ── Covariance ────────────────────────────────────────────────────────
        cov, inv_err = CovMatr(FIM)
        print("  Computed Covariance Matrix")

        # ── Save final HDF5 ───────────────────────────────────────────────────
        save_network_results(
            filepath      = output_path,
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
        print(f"  Done: {net_name}")