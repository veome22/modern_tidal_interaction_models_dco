This is the companion repository to [Modern tidal interaction models for rapid binary population synthesis: II. Binary black hole formation, mergers, and spins]([https://arxiv.org/abs/0707.2982](https://arxiv.org/abs/2606.23773)).

The code to reproduce all the plots depends on having access to a set of COMPAS simulations, which are too large to host on GitHub. The data files can be found on [Zenodo](https://zenodo.org/records/20941011?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjMTEzYTFhLTUyN2EtNDcwMC1hM2E5LTc5ZTViZjc1ZDMzNiIsImRhdGEiOnt9LCJyYW5kb20iOiI0NTJkY2YwMTJmMTBmNTliNWY4NmI2YmM1ZThlZDc5MiJ9.NZaUSntETuwIwXcamNX-iEBnFMLz9rk_N7j1-a0mWWk7jbZyEvrLHWu-wETDOX9Cfc3_L70ZTStAuFbJZ9y-mg).


1. The first step is to pre-process the raw COMPAS output files, which is done in [1_process_compas_sims.ipynb](../main/1_process_compas_sims.ipynb). This step only needs to be performed once, after which more convenient data files should be written.
2. The plots involving formation and merger yields from the complete simulations can be found in [2_merger_rates.ipynb](../main/2_merger_rates.ipynb). This notebook produces Figures 1 and 2.
3. Spins are plotted for the entire simulation in [3_spin_plots_full_simulations.ipynb](../main/3_spin_plots_full_simulations.ipynb), including Figures 5, 6, and 7.
4. We re-weight the simulations by cosmic star-formation history in [4_spin_plots_cosmic_integration.ipynb](../main/4_spin_plots_cosmic_integration.ipynb), including Figures 9, 10, 11, 12, 13 and 16.
5. Finally, [5_appendix_A_Bavera21.ipynb](../main/5_appendix_A_Bavera21.ipynb) contains the code to process an alternative set of spins, which are used to produce Figures 14 and 15.
