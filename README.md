# FRB_GW_cosmo: A Pipeline for FRB and Joint GWs-FRB Cosmological Analyses

This repository contains the full core Python pipeline and reproducibility materials for the cosmological analyses presented in the following papers:

* **Paper A:** *Hubble constant constraint using 117 FRBs with a more accurate probability density function for $DM_{\rm diff}$* [[10.3847/1538-4357/ae1a6d](https://iopscience.iop.org/article/10.3847/1538-4357/ae1a6d)] [[2508.05161](https://arxiv.org/abs/2508.05161)]
  In this project, we derived the uncertainty for $DM_{\rm diff}$ and computed a more accurate parameter $\sigma_{\rm diff}(S,z)$ to replace the traditional $F$-parameter. 

* **Paper B:** *Cosmological Constraints from GW-FRB Associations without Redshift Measurements for LIGO-Virgo and Cosmic Explorer* [[2604.03163](https://arxiv.org/abs/2604.03163)]
  In this project, we developed a general pipeline for joint GW-FRB association events. We compared the performance of different gravitational-wave detectors (LIGO-Virgo vs. Cosmic Explorer) and evaluated the impacts of different $DM_{\rm diff}$ probability density functions.

This pipeline is open-source and we welcome for further cosmological studies. If this software contributes to your research, please support our work by citing the relevant paper(s) above properly.

---

## 📂 Repository Structure

* `Notebooks/`: Contains the primary Jupyter Notebooks or Python scripts for reproducing the main results and figures.
  * `FRB_cosmo/`: Codes for constraining cosmological parameters (such as the Hubble constant $H_0$) using the 117 localized FRB samples (Paper A).
  * `GWs_FRB_cosmo/`: The general pipeline for analyzing joint GW-FRB association events under different detector configurations and $DM_{\rm diff}$ PDFs (Paper B).
  * `Realistic_sources/`: Notebooks dedicated to generating and analyzing GW mock events and realistic source simulations.
* `Data/`: Contains the observational data and catalogs used in this study, including the 117 localized FRB sample catalog (`./Data/FRB_data/FRB_new.csv`).
* `Python_scripts/`: Core modules, helper functions, and auxiliary scripts supporting the calculations in the Jupyter Notebooks.

---

## 🛠️ Prerequisites & Dependencies

To run the notebooks and scripts in this repository, you will need a standard Python 3 environment equipped with the following packages:
* `numpy`, `scipy`, `pandas`, `matplotlib` (Core scientific computing and data manipulation)
* `astropy` (Astronomical coordinates and cosmology utilities)
* `emcee` (Markov Chain Monte Carlo sampling)
* `bilby` (Bayesian inference for gravitational-wave data)
* `corner` (Visualization of MCMC posterior samples)
* `tqdm` (Progress bar tracking for long-running loops)

---

## ✉️ Contact & Support

Feel free to open an issue or email us if you have any questions regarding data reproduction or pipeline adaptation! 

* **Jiaming Zhuge**
  * Email: [jiaming.zhuge@unlv.edu](mailto:jiaming.zhuge@unlv.edu)

* **Marios Kalomenopoulos**
  * Email: [marios.kalomenopoulos@gmail.com](mailto:marios.kalomenopoulos@gmail.com)

---

## 🔗 Project Links

* **Zenodo Archive:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20282512.svg)](https://doi.org/10.5281/zenodo.20282512)
* **GitHub Repository:** [![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/MariosNT/FRB_GW_association_cosmo)