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
# pop_names = ['notides', 'realistic', 'perfect']
# # pop_names = ['realistic', 'perfect']
# pop_names = ['notides', 'perfect']
pop_names = ['realistic']

pop_paths = {
            'notides': 
                # [
                # '../pop_sims/notides_pop_v329_0_480k.h5',
                # '../pop_sims/notides_pop_v329_480k_960k.h5',
                # '../pop_sims/notides_pop_v329_960k_1440k.h5',
                # '../pop_sims/notides_pop_v329_1440k_1920k.h5',
                # '../pop_sims/notides_pop_v329_1920k_3840k.h5',
                # ],
                ['../pop_sims/notides_pop_v329_combined.h5'],
            
            'realistic':
                # [
                # '../pop_sims/realistic_pop_v329_0_480k.h5',
                # '../pop_sims/realistic_pop_v329_480k_960k.h5',
                # '../pop_sims/realistic_pop_v329_960k_1440k.h5',
                # '../pop_sims/realistic_pop_v329_1440k_1920k.h5',
                # '../pop_sims/realistic_pop_v329_1920k_3840k.h5',
                # ],
                ['../pop_sims/realistic_pop_v329_combined.h5'],
            
            'perfect':
                # [
                # '../pop_sims/perfect_pop_v329_0_480k.h5',
                # '../pop_sims/perfect_pop_v329_480k_960k.h5',
                # '../pop_sims/perfect_pop_v329_960k_1440k.h5',
                # '../pop_sims/perfect_pop_v329_1440k_1920k.h5',
                # '../pop_sims/perfect_pop_v329_1920k_3840k.h5',
                # ]
                ['../pop_sims/perfect_pop_v329_combined.h5'] 
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
# # Get SNRs with GWFast

# %%
first_pop = pop_spin_dfs[pop_names[0]]
bbh = first_pop[first_pop['BBH'] & first_pop['Merges_Hubble_Time']].copy()

# %%
# ── 0. Pull the BBH sample with merger redshifts already computed ─────────
# (bbh DataFrame with z_merger from the previous cell)
sample = bbh[bbh['z_merger'].notna()].copy()
print(f"Processing {len(sample)} BBHs")

# ── 1. Build O4 H1+L1+Virgo network ──────────────────────────────────────
# gwfast ships O4 ASDs under psds/observing_scenarios_paper/
# filenames from the Iacovelli+2022 observing scenarios paper
O4_psds = {
    'H1':    'aligo_O4high.txt',
    'L1':    'aligo_O4high.txt',
    'Virgo': 'avirgo_O4high_NEW.txt',
}

LVdetectors = {k: copy.deepcopy(glob.detectors[k]) for k in O4_psds}
for det, fname in O4_psds.items():
    LVdetectors[det]['psd_path'] = os.path.join(
        glob.detPath, 'observing_scenarios_paper', fname)

wf = IMRPhenomD()

signals = {
    det: GWSignal(
        wf,
        psd_path      = LVdetectors[det]['psd_path'],
        detector_shape= LVdetectors[det]['shape'],
        det_lat       = LVdetectors[det]['lat'],
        det_long      = LVdetectors[det]['long'],
        det_xax       = LVdetectors[det]['xax'],
        verbose       = True,
        useEarthMotion= False,   # Earth motion negligible for BBH
        fmin          = 10.,
        IntTablePath  = None,
    )
    for det in LVdetectors
}

net = DetNet(signals, verbose=True)

# %%
# ── 2. Build the gwfast event dictionary ─────────────────────────────────
# gwfast IMRPhenomD parameters: Mc, eta, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z,
#                                dL, iota, psi, ra, dec, tcoal, Phicoal
# (non-spin / non-angular params we randomise uniformly — sky pos doesn't matter
#  for population-level SNR statistics, but gwfast needs them)

N = len(sample)
rng = np.random.default_rng(42)

m1  = sample['Mass@DCO(1)'].values   # source-frame M_sun
m2  = sample['Mass@DCO(2)'].values
z   = sample['z_merger'].values

chi1z = sample['a1'].values * np.cos(sample['iota1'].values)          # dimensionless spin aligned with the orbital angular momentum
chi2z = sample['a2'].values * np.cos(sample['iota2'].values)

# detector-frame chirp mass and symmetric mass ratio
Mt  = (m1 + m2) * (1 + z)           # total detector-frame mass
eta = (m1 * m2) / (m1 + m2)**2
Mc  = Mt * eta**0.6                  # detector-frame chirp mass

dL  = Planck18.luminosity_distance(z).to(u.Gpc).value   # gwfast wants Gpc

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
    'ra'     : rng.uniform(0, 2*np.pi, N),
    'dec'    : np.arcsin(rng.uniform(-1, 1, N)),
    'tcoal'  : rng.uniform(0, 1, N),          # arbitrary reference GPS / GMST
    'Phicoal': rng.uniform(0, 2*np.pi, N),
}


# %%
# ── 3. Compute network SNRs ───────────────────────────────────────────────
snr_net = net.SNR(events, res=1000)   # shape (N,)

sample = sample.copy()
sample['SNR_O4'] = snr_net

print(f"Detectable (SNR > 8):  {(snr_net > 8).sum()} / {N}  "
      f"({100*(snr_net > 8).mean():.1f}%)")
print(f"Median SNR (all):      {np.median(snr_net):.2f}")
print(f"Median SNR (det. only):{np.median(snr_net[snr_net > 8]):.2f}")

# %%
fig, ax = plt.subplots(ncols=1, figsize=(6, 3))

dco_type = 'BBH'

pop_spin_df = sample

a1, a2, m1, m2 = pop_spin_df["a1"], pop_spin_df["a2"], pop_spin_df["Mass@DCO(1)"], pop_spin_df["Mass@DCO(2)"]
chi_eff_inclination = pop_spin_df['chi_eff']
snr_o4 = pop_spin_df['SNR_O4']

bins = np.linspace(-0.05, 1.05, 28)
ax.hist(chi_eff_inclination, bins=bins, density=True, alpha=0.8, histtype='step', lw=3, linestyle='-', label='K26', color=colorPalette['blue'])
ax.hist(chi_eff_inclination[snr_o4>8], bins=bins, density=True, alpha=0.8, histtype='step', lw=3, linestyle='-', label='K26, SNR$_{\\rm O4a}$ > 8', color=colorPalette['violet'])

# ax.set_title(dco_type, fontsize=20)
ax.set_xlabel('$\chi_{\\rm{eff}}$', fontsize=20)
ax.set_ylabel('P$(\chi_{\\rm{eff}})$', fontsize=20)
# ax.legend(fontsize=20)
ax.set_xticks(ticks=np.linspace(0, 1, 6))
# ax.set_yscale('log')
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid()

ax.legend(fontsize=14, loc=(0.5, 0.45))

plt.tight_layout()
plt.savefig(plot_path+'chi_eff_k26_snr_o4.pdf', dpi=300, bbox_inches='tight')



# # %%
# # ── 4. Compute Fisher matrices and covariances (detectable subset only) ───
# det_mask   = snr_net > 8
# det_events = {k: v[det_mask] for k, v in events.items()}

# # FisherMatr returns shape (n_params, n_params, N_det)
# FIM = net.FisherMatr(det_events, res=1000)

# # CovMatr inverts the Fisher matrix with condition-number checks
# # returns covariance (n_params, n_params, N_det) + inversion error array
# cov, inv_err = CovMatr(FIM)


# # %%
# # ── 5. Extract 1-sigma parameter errors ───────────────────────────────────
# ParNums = wf.ParNums   # dict: param_name → row/col index in Fisher matrix

# sigma_Mc  = np.sqrt(cov[ParNums['Mc'],  ParNums['Mc'],  :])   # Gpc (det-frame)
# sigma_eta = np.sqrt(cov[ParNums['eta'], ParNums['eta'], :])
# sigma_dL  = np.sqrt(cov[ParNums['dL'],  ParNums['dL'],  :])   # Gpc
# sigma_chi1z = np.sqrt(cov[ParNums['chi1z'], ParNums['chi1z'], :])
# sigma_chi2z = np.sqrt(cov[ParNums['chi2z'], ParNums['chi2z'], :])
# results = pd.DataFrame({
#     'SNR'         : snr_net[det_mask],
#     'Mc_det'      : Mc[det_mask],
#     'dL_Gpc'      : dL[det_mask],
#     'z_merger'    : z[det_mask],
#     'sigma_Mc'    : sigma_Mc,
#     'sigma_eta'   : sigma_eta,
#     'sigma_dL'    : sigma_dL,
#     'rel_err_Mc'  : sigma_Mc  / Mc[det_mask],
#     'rel_err_dL'  : sigma_dL  / dL[det_mask],
#     'inv_err'     : inv_err,
#     'sigma_chi1z' : sigma_chi1z,
#     'sigma_chi2z' : sigma_chi2z,
# }, index=sample.index[det_mask])

# print(results[['SNR','z_merger','rel_err_Mc','rel_err_dL']].describe().round(4))

# # %%



