# Questions on the Jupyter Notebooks

### In `Cosmo_constraints.ipynb`:

1. Question

   ```[code:python] 
   Omega_fit = Omega_m_array[id_Om_min]
   sigma_Om = np.abs(Omega_m_array[id_Om_min]-Omega_m_array[id_Om_sigma_1])
   # This is how to consider 1 sigma for mock events?
   ```

   Yes, cross-check with Gregory (p.262). In the $\chi^2$ method, they find the minimum and depending how far you are from the minimum and the number of degrees of freedom, you find your "percentage level".

   In this specific case, we have only one free parameter $\Omega_m$, and in the code above that, the sigma is calculated by looking for: `sigma1_chi_fit = chi_1D_min+1`, which corresponds to the $68.3\%$ level (usually denoted as $1\sigma$). 

   This part of the code, finds at which $\Omega_m$ values, the $1\sigma_{\Omega}$ corresponds to. Note that first we find the error in the $\chi^2$ distribution, and then we find where it corresponds in physical space.

   This is true for any data - not just the **mock** events.

### In `FRB_DM_z.ipynb`:

Catalogue of **FRB (cleaned) data**:

1 - The data from *Kritti* seem to be inconsistent with the fiducial model. But for $N_{\rm FRB} \sim 80$, the $\chi^2 \sim 160$, which could potentially indicate that the errors have been underestimated. Or do you think there is another explanation for this discrepancy?

2 - Some FRB data are duplicate in Kritti and Surajit, and they report different ${\rm DM}_{\rm ext}$. For example the `FRB 20220418A` has ${\rm DM}_{\rm ext} = 395.8$ in Kritti, and ${\rm DM}_{\rm ext} = 585.65$ in Surajit. In the joint catalogue, you use the Surajit ${\rm DM}_{\rm ext}$. Why?

