# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None 

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

pd.options.display.max_columns = None

# %%
from gwfast          import gwfastGlobals as glob
from gwfast.waveforms import IMRPhenomD
from gwfast.signal    import GWSignal
from gwfast.network   import DetNet
from gwfast.fisherTools import CovMatr

# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
import matplotlib
from matplotlib.ticker import LogLocator, AutoMinorLocator, MultipleLocator
import matplotlib.ticker as ticker

fontparams = {
    "font.family": "serif",
    "mathtext.fontset" : "stix",
    "grid.color": "gray",
    "grid.linestyle": ":",
    "axes.titlesize": "18",
    "axes.labelsize": "16",
    "xtick.labelsize": "16",
    "ytick.labelsize": "16",
    "xtick.labelbottom": "True",
    "legend.framealpha": "1",
}
rcParams.update(fontparams) 

from cycler import cycler

colorPalette = {'red': "#E64D4E",
                'orange': "#EE9063",
                'yellow': "#FFDD7B",
                'green': "#77AC54",
                'blue': "#0B92B1",
                'violet': "#665191",
                'gray': "#B4B4B4"
}

custom_cycler = (cycler(color=[colorPalette['red'], colorPalette['blue'], colorPalette['green']]))


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

pop_colors = {'notides': colorPalette['red'],
              'realistic': colorPalette['blue'],
              'perfect': colorPalette['green']}

pop_cmaps = {'notides': 'Reds',
              'realistic': 'Blues',
              'perfect': 'Greens'}

dco_types = ['BBH', 'BHNS', 'BNS']
dco_st = [28, 27, 26]
mt_labels = [
    "No MT", "Stable MT 1→2", "Stable MT 2→1", 
    "CE_Primary", "CE_Secondary", "CE_Both", "MT to Merger"
]
branching_labels = {1: 'Primary', 2: 'Secondary', 3: 'Both', 4: 'MT_to_Merger'}

plot_path = 'pop_plots/'
data_path = 'data_files/'


# %%
st_labels = ['MS_LT_0.7', 'MS_GT_0.7', 'HG', 'FGB', 'CHeB', 'EAGB', 'TPAGB', 'HeMS', 'HeHG', 'HeGB', 'HeWD', 'COWD', 'ONeWD', 'NS', 'BH', 'MR','CHE',  '--', '--', 'None']
st_labels_plot = ['MS', 'MS', 'HG', 'FGB', 'CHeB', 'EAGB', 'TPAGB', 'HeMS', 'HeHG', 'HeGB', 'HeWD', 'COWD', 'ONeWD', 'NS', 'BH', 'MR','CHE',  '--', '--', 'None']
evo_labels = ['', 'Simulation completed', 'Evolution stopped because an error occurred', 'Allowed time exceeded', 'Allowed timesteps exceeded', 
            "No user-provided timesteps read", "User-provided timesteps exhausted", "User-provided timesteps not consumed",
            'SSE error', 'Error evolving binary', 'Time exceeded DCO merger', 
            'Stars touching', 'Merged', 'Stars merged at birth', 
            'DCO formed', 'Double White Dwarf', 'Massless Remnant', 'Unbound binary']

# %%
pop_spin_dfs = {}
for pop_name in pop_names:
    pop_spin_dfs[pop_name] = pd.read_csv(data_path + pop_name+'_spin_df.csv') 


# %% [markdown]
# # Detector Network Definitions
# ─────────────────────────────────────────────────────────────────────────────
# To add a new network:
#   1. Add an entry to `network_psd_specs` mapping det_name → psd_path
#   2. Add a matching entry to `network_plot_specs` with label, color, linestyle
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
        'LIGOI':   os.path.join(glob.detPath, 'observing_scenarios_paper', 'AplusDesign.txt'),
    },

    # ── Add more networks here ────────────────────────────────────────────
    # 'MyNet': {
    #     'H1': '/path/to/psd.txt',
    #     ...
    # },
}

# Plot styling for each network (must match keys in network_psd_specs)
network_plot_specs = {
    'O4a': dict(label='K26, SNR$_{\\rm O4a}$ > 8',  color=colorPalette['violet'], linestyle='--'),
    '3G':  dict(label='K26, SNR$_{\\rm 3G}$ > 8',   color=colorPalette['red'],  linestyle='--'),
    # 'MyNet': dict(label='My Network, SNR > 8', color=colorPalette['orange'], linestyle='-.'),
}

SNR_THRESHOLD = 8   # detection threshold applied in the plot


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


# %% [markdown]
# # Prepare BBH Sample and Event Dict
# ─────────────────────────────────────────────────────────────────────────────

# %%
first_pop = pop_spin_dfs[pop_names[0]]
bbh = first_pop[first_pop['BBH'] & first_pop['Merges_Hubble_Time']].copy()

sample = bbh[bbh['z_merger'].notna()].copy()
print(f"Processing {len(sample)} BBHs")

# Build event parameter dict (shared across all networks)
N   = len(sample)
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


# %% [markdown]
# # Compute SNRs for All Networks
# ─────────────────────────────────────────────────────────────────────────────

# %%
wf = IMRPhenomD()

snr_results = {}   # {network_name: snr_array}

for net_name, psd_spec in network_psd_specs.items():
    print(f"\n─── Building network: {net_name} ───")
    net = build_network(psd_spec, wf)
    snr = compute_snr(net, events)
    snr_results[net_name] = snr
    sample[f'SNR_{net_name}'] = snr

    n_det = (snr > SNR_THRESHOLD).sum()
    print(f"  Detectable (SNR > {SNR_THRESHOLD}):  {n_det} / {N}  ({100*n_det/N:.1f}%)")
    print(f"  Median SNR (all):       {np.median(snr):.2f}")
    print(f"  Median SNR (det. only): {np.median(snr[snr > SNR_THRESHOLD]):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# Stacked chi_eff histogram by SNR bin — 3G network
# ─────────────────────────────────────────────────────────────────────────────

snr_edges  = [8, 25, 50, 100, 250, 500, 1000, np.inf]
snr_labels = [
    r'$8 \leq \rho < 25$',
    r'$25 \leq \rho < 50$',
    r'$50 \leq \rho < 100$',
    r'$100 \leq \rho < 250$',
    r'$250 \leq \rho < 500$',
    r'$500 \leq \rho < 1000$',
    r'$\rho \geq 1000$',
]
n_snrbins = len(snr_edges) - 1

chi_edges   = np.linspace(-0.05, 1.05, 30)
bar_centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])
bar_width   = chi_edges[1] - chi_edges[0]

for net_name, snr in snr_results.items():

    # --- only plot detected events (SNR >= lower edge) ---
    det_mask    = snr >= snr_edges[0]
    chi_det     = sample['chi_eff'].values[det_mask]
    snr_det     = snr[det_mask]

    # assign each detected event to an SNR bin
    snr_bin_idx = np.digitize(snr_det, bins=snr_edges) - 1
    snr_bin_idx = np.clip(snr_bin_idx, 0, n_snrbins - 1)

    # build 2D count array (n_snrbins x n_chi_bins)
    counts = np.zeros((n_snrbins, len(chi_edges) - 1))
    for i in range(n_snrbins):
        mask = snr_bin_idx == i
        counts[i], _ = np.histogram(chi_det[mask], bins=chi_edges)

    # normalise by total detected N so y-axis is probability
    N_total = counts.sum()
    probs   = counts / N_total

    # colormap — low SNR = dark, high SNR = bright yellow
    cmap   = cm.get_cmap('plasma', n_snrbins)
    colors = [cmap(i) for i in range(n_snrbins)]

    fig, ax = plt.subplots(figsize=(10, 6))

    bottoms = np.zeros(len(chi_edges) - 1)
    for i in range(n_snrbins):
        ax.bar(bar_centers, probs[i], width=bar_width,
               bottom=bottoms, color=colors[i],
               label=snr_labels[i], edgecolor='none', linewidth=0)
        bottoms += probs[i]

    ax.set_xlabel(r'$\chi_{\rm eff}$', fontsize=22)
    ax.set_ylabel(r'$p(\chi_{\rm eff})$', fontsize=22)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=14)
    ax.grid(axis='y', alpha=0.4)

    # legend with every other bin labeled to avoid crowding
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::2], labels[::2],
              fontsize=12, loc='upper left',
              framealpha=0.8,
              title=rf'$\rho_{{\rm {net_name}}}$',
              title_fontsize=13)

    plt.tight_layout()
    out = plot_path + f"chi_eff_stacked_snr_{net_name}.pdf"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {out}")
    plt.close(fig)
