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
