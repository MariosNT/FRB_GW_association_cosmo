# Data Folder Info

------------

## $C_0 - z - \sigma$ relation

In the folder `C_0_z_sigma_relations`, we collect useful data and the interpolations between the $C_0$ parameter (from the cosmic pdf) with redshift $z$ and the "variance" $\sigma$ in the same cosmic pdf.

- `sigma_C0_beta3.ascii` is from the `FRB` package [[here](https://github.com/FRBs/FRB/tree/107e37332c4857b7b363b637de0f5ea7d4d0761d/frb/data/DM)] (connected to the original paper by [[Macquart et al.](https://arxiv.org/abs/2005.13161)])
- `Zijian_TNG.xlsx` is from [[Zhang, Z. J. et al.](https://arxiv.org/abs/2011.14494)]. It is derived by fits with data from the IllustrisTNG simulation, summarised in their Table 1.
- `C_0_z.csv` 
- `C_0_z_v2.csv` 

-------

## FRB data (Old)

All ${\rm DM}$s are in units of pc/cm$^3$.

- `FRB_Kritti`: are from  [[Kritti et al., 2024](https://arxiv.org/abs/2409.16964)]. The `.csv` and `.xlsx` are basically the same. 
- `FRB_Surajit`: are from [[Surajit et al., 2024](https://arxiv.org/abs/2410.01974)]. The `.csv` and `.xlsx` are basically the same. 

`.xlsx` use color to emphasize duplicate samples.



- `FRB_Macquart_org` are from the 5 *golden* data points from Macquart's original paper [[Macquart et al.](https://arxiv.org/abs/2005.13161)].
- `FRB_Macquart` are the same data, with extra sky localisation info, taken from [[Surajit et al., 2024](https://arxiv.org/abs/2410.01974)].  

To calculate ${\rm DM}_{\rm ext}$ we have used: ${\rm DM}_{\rm ext}={\rm DM}-50-30$ as in the methods in Macquart's+ paper.



- `FRB_all.xlsx` include all the FRB data. For the duplicate ones, we have used [[Kritti et al., 2024](https://arxiv.org/abs/2409.16964)] (which is more updated). Only for one  FRB with negative ${\rm DM}_{\rm ext}$ (FRB  20220319D), we use Surajit's+ data. 
- `FRB_all.xlsx` is the original version with all FRB data. ${\rm DM}_{\rm ext}$ is calculated through ${\rm DM}_{\rm ext}={\rm DM}-{\rm DM}_{\rm MW}$ (NE2001). 



- `CHIME250211217` are the data from CHIME's catalog of local universe FRBs [[here](https://arxiv.org/abs/2502.11217)].



- `FRB_Blinkverse` are the data from the *Blinkverse* [[database](https://blinkverse.zero2x.org/availability/host)], with the FRBs with specified redshift $z$.



- `FRB_DM_compare.csv` compares 4 different models for calculating the ${\rm DM}_{\rm MW,\ ISM}$. (see `README` at the `Notebooks/FRB_cosmo` folder for a description of these packages)

--------------------------------

## FRB data

- `FRB_new.csv` is the new dataset, produce by preprocessing `FRB.csv` with `FRB_DM_ext_recalculate.ipynb`, which uses the `pygedm` package to calculate ${\rm DM}_{\rm MW,\ ISM}$ with the `ne2001` model. For the other parameters for the ${\rm DM}$:
  - We assume ${\rm DM}_{\rm MW,\ halo}=30$.
  - ${\rm DM}_{\rm ext}$ is calculated as: ${\rm DM}_{\rm ext}({\rm ne2001})={\rm DM}_{\rm obs}-{\rm DM}_{\rm MW, \ ISM}-{\rm DM}_{\rm MW,\ halo} = {\rm DM}_{\rm diff} + {\rm DM}_{\rm host,\ src}$. 
  - We drop the FRB  20220319D, which has negative ${\rm DM}_{\rm ext}$. 
  - In the `FRB.csv`, for the calculation of ${\rm DM_{\rm ext}}$ we take the values from the literature (as specified in the file)

