This is the companion repository to [Modern tidal interaction models for rapid binary population synthesis: II. Binary black hole formation, mergers, and spins]([https://arxiv.org/abs/0707.2982](https://arxiv.org/abs/2606.23773)).

The code to reproduce all the plots depends on having access to a set of COMPAS simulations, which are too large to host on GitHub. The data files can be found on [Zenodo](https://zenodo.org/records/20941457?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJkMTdhMzc1LTc5ZGYtNDM5Ni04M2Q1LTg2NjhkMjVkYjBlNiIsImRhdGEiOnt9LCJyYW5kb20iOiJiMjVmNzQ5MGYxYzk5NmM2ODkyODAzN2Q3ZTFhODlmOSJ9.Y4YdblFR0RSDAEwbiC6FuzfTOLMhy6rDcwVTot_zfdShCeJaYnmXB2Ybbs6uQm31xWVV4iZYIpCSR8yYg2VnHQ).

## Producing COMPAS Simulations (skip if downloading processed files from Zenodo)
We ran COMPAS in parallel on an HPC cluster with sbatch to produce the 3.8 M simulations per tidal prescription used in this work. Examples of the sbatch scripts are shown in [sbatch_scripts](../main/sbatch_scripts/). The COMPAS workflow is best set up by each user to best suit their system, but the general parameters of the simulations may look something like this for the `KAPIL2026` tidal prescription (with COMPAS v3.29.00):

```
COMPAS -n 1000 --output-path pop_sims --initial-mass-function-min 5.0 --initial-mass-function-max 150.0 --orbital-period-min 1.0 --orbital-period-max 1000.0 --orbital-period-distribution FLATINLOG --eccentricity-distribution SANA2012 --metallicity-distribution LOGUNIFORM --detailed-output FALSE --logfile-definitions 'detailed_output_pop_definitions_min.txt' --maximum-number-timestep-iterations 1999999 --mass-change-fraction 0.001 --radial-change-fraction 0.001 --tides-prescription KAPIL2026
```
For the other simulations shown in this work, the `--tides-prescription` option may be set to `NONE`or `PERFECT` for the main paper, and `ZAHN1977` for the comparison in Appendix B.

Note that this repository relies on all the outputs defined by `--logfile-definitions 'detailed_output_pop_definitions_min.txt'`. The output definitions are included in this repository.

The following workflow outlines the steps after having produced and combined the COMPAS simulations from above. See the [COMPAS documentation](https://compas.readthedocs.io/en/latest/pages/User%20guide/Post-processing/hdf5/post-processing-h5copy.html) for information on how to combine the hdf5 files from across multiple runs.

1. The first step is to pre-process the raw COMPAS output files, which is done in [1_process_compas_sims.ipynb](../main/1_process_compas_sims.ipynb). This step only needs to be performed once, after which more convenient data files should be written.

## Paper Plots
This section can be followed as long as the user has processed the COMPAS output already, or downloaded the outputs from [Zenodo](https://zenodo.org/records/20941457?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJkMTdhMzc1LTc5ZGYtNDM5Ni04M2Q1LTg2NjhkMjVkYjBlNiIsImRhdGEiOnt9LCJyYW5kb20iOiJiMjVmNzQ5MGYxYzk5NmM2ODkyODAzN2Q3ZTFhODlmOSJ9.Y4YdblFR0RSDAEwbiC6FuzfTOLMhy6rDcwVTot_zfdShCeJaYnmXB2Ybbs6uQm31xWVV4iZYIpCSR8yYg2VnHQ). Note that Figure 13 requires the data files hosted on Zenodo.

2. The plots involving formation and merger yields from the complete simulations can be found in [2_merger_rates.ipynb](../main/2_merger_rates.ipynb). This notebook produces Figures 1 and 2.
3. Spins are plotted for the entire simulation in [3_spin_plots_full_simulations.ipynb](../main/3_spin_plots_full_simulations.ipynb), including Figures 5, 6, and 7.
4. We re-weight the simulations by cosmic star-formation history in [4_spin_plots_cosmic_integration.ipynb](../main/4_spin_plots_cosmic_integration.ipynb), including Figures 9, 10, 11, 12, 13 and 16.
5. Finally, [5_appendix_A_Bavera21.ipynb](../main/5_appendix_A_Bavera21.ipynb) contains the code to process an alternative set of spins, which are used to produce Figures 14 and 15.
