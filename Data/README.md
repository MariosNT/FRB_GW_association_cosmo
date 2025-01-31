# FRB data from

The datasets are from Kritti (new localized data) and Surajit. Saved as `FRB_Surajit` and `FRB_Kritti`. `.csv` and `.xlsx` are basically the same, `.xlsx` use color to emphasize duplicate one. 

`FRB_all.xlsx` include all the FRB data. For the duplicate one, I choose Kritti (new), only for one negative DM FRB use Surajit's data. `FRB_all.xlsx` is the original version with all FRB data. $DM_{ext}$ is the original one which is $DM_{ext}=DM-DM_{MW}$. 

FRB.csv is the dataset preprocessed from `FRB_all.xlsx`, consider $DM_{halo}=50$ and drop $DM_{ext}$ negative FRB. The $DM_{ext}$ in `DM_all.csv` is $DM_{ext}=DM-DM_{MW}-DM_{halo}$. Note here do not reduce $DM_{MW,halo}\sim 30$, so $ DM_{ext}=DM_{MW,halo}+DM_{IGM}+\frac{DM_{host}}{1+z} $. The preprocess in `DM_show.ipynb`.

FRB_Macquart.csv is Macquart's golden data select from FRB.csv. FRB_Macquart_org is the golden data in Macquart's paper. $DM_{ext}=DM-50-30$ in Macquart paper method part.