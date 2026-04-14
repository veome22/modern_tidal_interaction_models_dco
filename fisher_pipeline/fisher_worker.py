"""
fisher_worker.py
----------------
Called as a subprocess by compute_fisher_parallel.py.
Owns a strided slice of batches for one network and saves them independently.

Usage (internal — do not call directly):
    python fisher_worker.py <net_name> <task_id> <n_tasks> <batch_size> <res>
"""
import sys
import os
import gc
import numpy as np

# ── Resolve project root so imports work regardless of cwd ───────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gwfast.waveforms import IMRPhenomD
from gwfast.signal    import GWSignal
from gwfast.network   import DetNet
from gwfast           import gwfastGlobals as glob
import copy

# ── Re-use your existing helpers ─────────────────────────────────────────────
# (copy build_network here, or factor it into a shared helpers.py)
def build_network(psd_spec, waveform, fmin=10.):
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
            verbose        = False,
            useEarthMotion = False,
            fmin           = fmin,
            IntTablePath   = None,
        )
    return DetNet(signals, verbose=False)


def main():
    # ── Args ─────────────────────────────────────────────────────────────────
    net_name   = sys.argv[1]
    task_id    = int(sys.argv[2])
    n_tasks    = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    res        = int(sys.argv[5])

    data_path = "/home/vkapil1/scratch16-berti/tides_compas_veome/modern_tidal_interaction_models_dco_applications/data_files/"
    batch_dir = f"{data_path}{net_name}_parallel_batches/"
    snr_cache = f"{data_path}{net_name}_snr.npy"

    os.makedirs(batch_dir, exist_ok=True)

    # ── Load shared inputs written by orchestrator ────────────────────────────
    # Orchestrator saves events + SNR before launching workers
    events_path = f"{data_path}{net_name}_events.npz"
    events_raw  = np.load(events_path)
    events      = dict(events_raw)          # {param: array}
    snr         = np.load(snr_cache)

    det_mask    = snr > 8                   # threshold baked in by orchestrator
    det_events  = {k: v[det_mask] for k, v in events.items()}
    N_det       = det_mask.sum()
    n_batch     = int(np.ceil(N_det / batch_size))

    # ── This worker's strided batch indices ──────────────────────────────────
    my_batches      = list(range(task_id, n_batch, n_tasks))
    my_batches_todo = [
        i for i in my_batches
        if not os.path.exists(os.path.join(batch_dir, f"batch_{i:04d}.npy"))
    ]

    print(f"[worker {task_id}] {len(my_batches)} assigned | "
          f"{len(my_batches_todo)} remaining", flush=True)

    if not my_batches_todo:
        print(f"[worker {task_id}] Nothing to do.", flush=True)
        return

    # ── Load network PSD spec saved by orchestrator ───────────────────────────
    psd_spec_path = f"{data_path}{net_name}_psd_spec.npz"
    psd_spec      = dict(np.load(psd_spec_path, allow_pickle=True))
    psd_spec      = {k: str(v) for k, v in psd_spec.items()}   # paths as strings

    wf  = IMRPhenomD()
    net = build_network(psd_spec, wf)

    # ── Compute and save ──────────────────────────────────────────────────────
    for i in my_batches_todo:
        lo, hi     = i * batch_size, min((i + 1) * batch_size, N_det)
        batch      = {k: v[lo:hi] for k, v in det_events.items()}
        # batch_path = os.path.join(batch_dir, f"batch_{i:04d}.npy")
        # tmp_path   = batch_path + ".tmp"
        tmp_path   = os.path.join(batch_dir, f"batch_{i:04d}.tmp.npy")
        batch_path = os.path.join(batch_dir, f"batch_{i:04d}.npy")

        print(f"[worker {task_id}] batch {i+1}/{n_batch} "
              f"(events {lo}–{hi-1})", flush=True)

        fim_batch = net.FisherMatr(batch, res=res)
        np.save(tmp_path, fim_batch)
        os.replace(tmp_path, batch_path)

        print(f"[worker {task_id}]   → {batch_path}", flush=True)

        del batch, fim_batch
        gc.collect()

    print(f"[worker {task_id}] Done.", flush=True)


if __name__ == '__main__':
    main()