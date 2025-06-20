# Notebooks Folder Info

------------

## Extracting ${\rm DM}_{\rm MW,\ ISM}$.

In order to keep the cosmological and host dependence on ${\rm DM}$, we need to extract the impact of the interstellar medium of the Milky Way. For this, we use the galactic coordinates of the FRB data and compare 4 packages:

- `pygedm` (this is the package we use as default). Can be found [[here](https://anaconda.org/conda-forge/pygedm)].
- `ne2001`. Can be found [[here](https://ne2001.readthedocs.io/en/latest/readme.html)].
- `pyne2001`. Can be found [[here](https://pypi.org/project/pyne2001/)].
- `mwprop`. Can be found [[here](https://github.com/stella-ocker/mwprop)].

All the packages model the `NE2001` model for the galactic distribution of free electrons. Original papers [[Cordes & Lazio, 2002](https://arxiv.org/abs/astro-ph/0207156)] and [[Cordes & Lazio, 2003](https://arxiv.org/abs/astro-ph/0301598)].

We use the `pygedm` package for the final results, since it is consistent with the other packages, runs faster and has coverage of large redshifts, without throwing any errors.

**Note 1**: The old ${\rm DM}_{\rm ext}$ in `FRB.csv` is still using the `ne2001` model, but is collected from other papers, as specified in the dataset. The values there are recalculated only from the data coming from [[CHIME 2025](https://arxiv.org/abs/2502.11217)], where we recalculate the ${\rm DM}_{\rm ext}$ values using the `ne2001` model.

**Note 2**: In the paper, and in `FRB_new.csv` , we report the values of both the `ne2001` and `ymw16` models. In our work, we use the `ne2001` model, since it produces fewer FRBs with negative ${\rm DM}_{\rm ext}$ values. The `ymw16` is also calculated through the `pygedm` package.

**Relevant Notebooks**

- `FRB_DM_ext_recalculate.ipynb`: Uses the `pygedm` package and the galactic coordinates of the localised FRBs, to calculate ${\rm DM}_{\rm ext}$. Starts from the `FRB.csv` file and saves the new data in `FRB_new.csv`.
- `ne2001_compare.ipynb`: Compares the four packages mentioned above. This is moved to the `Archive` folder.

## 



**TO DO**: Combine `F_search` with `F_search_MK` and clear

**TO DO**: Clean `DM_z_constraints` and `MK` version
