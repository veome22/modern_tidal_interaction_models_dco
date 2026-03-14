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

    # ── O4a: H1 + L1 + Virgo ─────────────────────────────────────────────
    'O4a': {
        'H1':    os.path.join(glob.detPath, 'observing_scenarios_paper', 'aligo_O4high.txt'),
        'L1':    os.path.join(glob.detPath, 'observing_scenarios_paper', 'aligo_O4high.txt'),
        'Virgo': os.path.join(glob.detPath, 'observing_scenarios_paper', 'avirgo_O4high_NEW.txt'),
    },

    # # ── 3G: ET (triangular, 15 km) + CE1 + A+ ────────────────────────────
    # '3G': {
    #     'ETSL':    os.path.join(glob.detPath, 'ET_designs_comparison_paper/HFLF_cryo', 'ETLength15km.txt'),
    #     'ETMRL45d':os.path.join(glob.detPath, 'ET_designs_comparison_paper/HFLF_cryo', 'ETLength15km.txt'),
    #     'CE1Id':   os.path.join(glob.detPath, 'ce_strain', 'cosmic_explorer.txt'),
    #     'LIGOI':   os.path.join(glob.detPath, 'observing_scenarios_paper', 'AplusDesign.txt'),
    # },

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


# # %% [markdown]
# # # Plot χ_eff Distributions
# # ─────────────────────────────────────────────────────────────────────────────

# # %%
# fig, ax = plt.subplots(ncols=1, figsize=(6, 3))

# chi_eff = sample['chi_eff']
# bins    = np.linspace(-0.05, 1.05, 28)

# # Full population (unfiltered)
# ax.hist(chi_eff, bins=bins, density=False, alpha=0.8,
#         histtype='step', lw=3, linestyle='-',
#         label='K26', color=colorPalette['blue'])

# # One curve per network
# for net_name, snr in snr_results.items():
#     spec = network_plot_specs[net_name]
#     ax.hist(chi_eff[snr > SNR_THRESHOLD], bins=bins, density=False, alpha=0.8,
#             histtype='step', lw=3,
#             linestyle=spec['linestyle'],
#             label=spec['label'],
#             color=spec['color'])

# ax.set_xlabel('$\chi_{\\rm{eff}}$', fontsize=20)
# ax.set_ylabel('N$_{\\rm Det}$', fontsize=20)
# ax.set_xticks(ticks=np.linspace(0, 1, 6))
# ax.tick_params(axis='both', which='major', labelsize=20)
# ax.semilogy()
# ax.grid()
# ax.legend(fontsize=13, loc=(0.2, 0.05))

# plt.tight_layout()
# plt.savefig(plot_path+'chi_eff_k26_snr_networks.pdf', dpi=300, bbox_inches='tight')
# plt.show()


# for net_name, snr in snr_results.items():
#     fig, axes = plt.subplots(ncols=3, figsize=(18, 5))

#     pop_df = sample[snr > SNR_THRESHOLD]


#     any_rlof = (pop_df['RLOF_Primary']==True) + (pop_df['RLOF_Secondary']==True) + (pop_df['RLOF_Both']==True)
#     any_ce = (pop_df['CE_Primary']==True) + (pop_df['CE_Secondary']==True) +(pop_df['CE_Both']==True)
#     any_che = (pop_df['CH_on_MS(1)']==True) + (pop_df['CH_on_MS(2)']==True)
#     smt_mask = any_rlof * (~any_ce) * (~any_che)
#     ce_mask = any_ce * (~any_che)
#     che_mask = any_che

#     m1, m2 = pop_df["Mass@DCO(1)"], pop_df["Mass@DCO(2)"]
#     z1 = pop_df["Metallicity@ZAMS(1)"]
#     chi_eff_inclination = pop_df['chi_eff']

#     alpha = 0.4
#     s = 10

#     ######################################
#     ax = axes[0]

#     x_vals = chi_eff_inclination
#     y_vals = np.minimum(m1,m2) / np.maximum(m1,m2)
#     y_vals = m1+m2


#     ax.scatter(x_vals[ce_mask], y_vals[ce_mask], alpha=alpha, s=s, color=colorPalette['blue'], label='CE', zorder=1)
#     ax.scatter(x_vals[smt_mask], y_vals[smt_mask], alpha=alpha, s=s, color=colorPalette['violet'], label='SMT', zorder=10)
#     ax.scatter(x_vals[che_mask], y_vals[che_mask], alpha=2*alpha, s=s, color=colorPalette['green'], label='CHE')

#     ax.set_xlabel(r'$\chi_{\rm eff}$', fontsize=25)
#     # ax.set_ylabel(r'$q$', fontsize=25)
#     ax.set_ylabel('$M_{\\rm tot}$ [M$_\odot$]', fontsize=25)

#     # ax.set_ylim(0, 1)
#     ax.set_ylim(0, 94)
#     ax.set_xlim(-0.05, 1.05)

#     ax.grid()

#     ############################################

#     # Get x and y values
#     y_vals = np.minimum(m1, m2) / np.maximum(m1, m2)
#     x_vals = chi_eff_inclination


#     ax = axes[1]
#     ax.scatter(x_vals[ce_mask], y_vals[ce_mask], alpha=alpha, s=s, color=colorPalette['blue'], label='CE', zorder=1)
#     ax.scatter(x_vals[smt_mask], y_vals[smt_mask], alpha=alpha, s=s, color=colorPalette['violet'], label='SMT', zorder=10)
#     ax.scatter(x_vals[che_mask], y_vals[che_mask], alpha=2*alpha, s=s, color=colorPalette['green'], label='CHE')

#     ax.set_xlabel(r'$\chi_{\rm eff}$', fontsize=25)
#     ax.set_ylabel(r'$q$', fontsize=25)

#     ax.set_xlim(-0.05, 1.05)

#     ax.grid()

#     ######

#     ax = axes[2]

#     y_vals = np.minimum(m1,m2) / np.maximum(m1,m2)
#     x_vals = m1+m2

#     ax.scatter(x_vals[ce_mask], y_vals[ce_mask], alpha=alpha, s=s, color=colorPalette['blue'], label='CE', zorder=0)
#     ax.scatter(x_vals[smt_mask], y_vals[smt_mask], alpha=alpha, s=s, color=colorPalette['violet'], label='SMT', zorder=1)
#     ax.scatter(x_vals[che_mask], y_vals[che_mask], alpha=2*alpha, s=s, color=colorPalette['green'], label='CHE')

#     ax.set_xlabel('$M_{\\rm tot}$ [M$_\odot$]', fontsize=25)
#     ax.set_ylabel(r'$q$', fontsize=25)
#     ax.set_xlim(0, 94)

#     ax.grid()

#     from matplotlib.lines import Line2D

#     legend_elements = [
#     Line2D([0], [0], marker='o', color='w', label='CE',
#     markerfacecolor=colorPalette['blue'], markersize=10, alpha=1),
#     Line2D([0], [0], marker='o', color='w', label='SMT',
#         markerfacecolor=colorPalette['violet'], markersize=10, alpha=1),
#     Line2D([0], [0], marker='o', color='w', label='CHE',
#         markerfacecolor=colorPalette['green'], markersize=10, alpha=1),
#     ]

#     axes[2].legend(handles=legend_elements, fontsize=20, loc=(0.6, 0.05))


#     plt.tight_layout()
#     plt.savefig(plot_path+"chi_eff_q_mtot_det_"+net_name+".png", dpi=300, bbox_inches='tight')
#     print("Saved "+ plot_path+"chi_eff_q_mtot_det_"+net_name+".png")


for net_name, snr in snr_results.items():
    fig, axes = plt.subplots(ncols=3, figsize=(18, 5))

    # --- masks over the FULL sample ---
    any_rlof = (sample['RLOF_Primary']==True) + (sample['RLOF_Secondary']==True) + (sample['RLOF_Both']==True)
    any_ce   = (sample['CE_Primary']==True)   + (sample['CE_Secondary']==True)   + (sample['CE_Both']==True)
    any_che  = (sample['CH_on_MS(1)']==True)  + (sample['CH_on_MS(2)']==True)

    smt_mask = any_rlof * (~any_ce) * (~any_che)
    ce_mask  = any_ce  * (~any_che)
    che_mask = any_che

    # --- detected / undetected split ---
    det      = snr > SNR_THRESHOLD          # bool array over full sample
    not_det  = ~det

    m1  = sample["Mass@DCO(1)"]
    m2  = sample["Mass@DCO(2)"]
    chi = sample['chi_eff']
    z   = sample['z_merger']
    tcoal = sample['Coalescence_Time']
    M   = m1 + m2
    q   = np.minimum(m1, m2) / np.maximum(m1, m2)

    ALPHA_HI  = 0.3
    ALPHA_LO  = 0.3
    s         = 10

    # panel definitions: (ax, x_data, y_data, xlabel, ylabel, xlim, ylim)
    panels = [
        (axes[0], chi, tcoal,
         r'$\chi_{\rm eff}$', r'$t_{\rm coal}$ $\rm [Myr]$',
         (-0.05, 1.05), (0, None)),
        (axes[1], chi, z,
         r'$\chi_{\rm eff}$', r'$z_{\rm merger}$',
         (-0.05, 1.05), (0, None)),
        (axes[2], tcoal,  z,
         r'$t_{\rm coal}$ $\rm [Myr]$', r'$z_{\rm merger}$',
         (0, None),       (0, None)),
    ]

    for ax, xv, yv, xlabel, ylabel, xlim, ylim in panels:

        for mask, color, label, zorder in [
            (ce_mask,  colorPalette['blue'],   'CE',  1),
            (smt_mask, colorPalette['violet'], 'SMT', 2),
            (che_mask, colorPalette['green'],  'CHE', 3),
        ]:
            # low-alpha background (SNR <= threshold)
            ax.scatter(xv[mask & not_det], yv[mask & not_det],
                       alpha=ALPHA_LO, s=s, color=color,
                       zorder=zorder, rasterized=True)

            # high-alpha foreground (SNR > threshold)
            ax.scatter(xv[mask & det], yv[mask & det],
                       alpha=ALPHA_HI, s=s, color=color,
                       label=label, zorder=zorder + 10,
                       rasterized=True)

        ax.set_xlabel(xlabel, fontsize=25)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.set_xlim(xlim)
        if ylim != (0, None):
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(bottom=0)
        ax.grid()

    # legend on last panel
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='CE',
               markerfacecolor=colorPalette['blue'],   markersize=10),
        Line2D([0], [0], marker='o', color='w', label='SMT',
               markerfacecolor=colorPalette['violet'], markersize=10),
        Line2D([0], [0], marker='o', color='w', label='CHE',
               markerfacecolor=colorPalette['green'],  markersize=10),
    ]
    axes[2].legend(handles=legend_elements, fontsize=20, loc=(0.6, 0.15))

    plt.tight_layout()
    out = plot_path + "chi_eff_tcoal_z_" + net_name + ".png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print("Saved " + out)
    plt.close(fig)



# for net_name, snr in snr_results.items():
#     fig, axes = plt.subplots(ncols=3, figsize=(18, 5))

#     # --- masks over the FULL sample ---
#     any_rlof = (sample['RLOF_Primary']==True) + (sample['RLOF_Secondary']==True) + (sample['RLOF_Both']==True)
#     any_ce   = (sample['CE_Primary']==True)   + (sample['CE_Secondary']==True)   + (sample['CE_Both']==True)
#     any_che  = (sample['CH_on_MS(1)']==True)  + (sample['CH_on_MS(2)']==True)

#     smt_mask = any_rlof * (~any_ce) * (~any_che)
#     ce_mask  = any_ce  * (~any_che)
#     che_mask = any_che

#     # --- detected / undetected split ---
#     det     = snr > SNR_THRESHOLD
#     not_det = ~det

#     m1   = sample["Mass@DCO(1)"]
#     m2   = sample["Mass@DCO(2)"]
#     chi  = sample['chi_eff']
#     z    = sample['z_merger']
#     M    = m1 + m2
#     q    = np.minimum(m1, m2) / np.maximum(m1, m2)
#     Z    = sample['Metallicity@ZAMS(1)']   # or (Z1+Z2)/2 if you prefer the mean

#     ALPHA_HI = 0.5
#     ALPHA_LO = 0.003
#     s        = 10

#     # panel definitions: (ax, x_data, y_data, xlabel, ylabel, xlim, ylim)
#     panels = [
#         (axes[0], chi, Z,
#          r'$\chi_{\rm eff}$', r'$Z$',
#          (-0.05, 1.05), None),
#         (axes[1], q, Z,
#          r'$q$', r'$Z$',
#          (0, 1.05), None),
#         (axes[2], chi, q,
#          r'$\chi_{\rm eff}$', r'$q$',
#          (-0.05, 1.05), (0, 1.05)),
#     ]

#     for ax, xv, yv, xlabel, ylabel, xlim, ylim in panels:

#         for mask, color, label, zorder in [
#             (ce_mask,  colorPalette['blue'],   'CE',  1),
#             (smt_mask, colorPalette['violet'], 'SMT', 2),
#             (che_mask, colorPalette['green'],  'CHE', 3),
#         ]:
#             # low-alpha background (SNR <= threshold)
#             ax.scatter(xv[mask & not_det], yv[mask & not_det],
#                        alpha=ALPHA_LO, s=s, color=color,
#                        zorder=zorder, rasterized=True)

#             # high-alpha foreground (SNR > threshold)
#             ax.scatter(xv[mask & det], yv[mask & det],
#                        alpha=ALPHA_HI, s=s, color=color,
#                        label=label, zorder=zorder + 10,
#                        rasterized=True)

#         ax.set_xlabel(xlabel, fontsize=25)
#         ax.set_ylabel(ylabel, fontsize=25)
#         ax.set_xlim(xlim)
#         if ylim is not None:
#             ax.set_ylim(ylim)
#         ax.set_yscale('log')   # metallicity spans orders of magnitude
#         ax.grid(which='both')

#     # legend on last panel
#     from matplotlib.lines import Line2D
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='CE',
#                markerfacecolor=colorPalette['blue'],   markersize=10),
#         Line2D([0], [0], marker='o', color='w', label='SMT',
#                markerfacecolor=colorPalette['violet'], markersize=10),
#         Line2D([0], [0], marker='o', color='w', label='CHE',
#                markerfacecolor=colorPalette['green'],  markersize=10),
#     ]
#     axes[2].legend(handles=legend_elements, fontsize=20, loc=(0.6, 0.5))

#     plt.tight_layout()
#     out = plot_path + "chi_eff_q_metallicity_" + net_name + ".png"
#     plt.savefig(out, dpi=300, bbox_inches='tight')
#     print("Saved " + out)
#     plt.close(fig)
