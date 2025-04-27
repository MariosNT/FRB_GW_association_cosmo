###############################################
### Functions for the cosmological analysis ###
###############################################

from config import *
from cosmo_support import *
from support import normalise


###############################################

def create_mock_frb_data(N_mock_data, frb_input_data=[0], z_min=0.2, z_max=1.2, pdf_modelling='Gaussian',\
                         H0=HUBBLE, Omega_b=OMEGA_BARYONS, Omega_m=OMEGA_MATTER, f_IMG=f_IGM, w=W_LAMBDA, alpha=ALPHA_IGM,\
                         e_mu_mock=50, sigma_mu_mock=0.5, Dv_mock=5.5,\
                         DM_mu_host=50, A_host_mock=0.1, A_diff_mock=0.1):
    
    HOf_mock=H0*Omega_b*f_IGM
    
    if len(frb_input_data)>1:
        ## We fit a KDE in the real data and draw z-samples from that
        kde_z = gaussian_kde(frb_input_data['z'])
        z_high_res=np.linspace(np.floor(np.min(frb_input_data['z'])), np.ceil(np.max(frb_input_data['z'])), 100)
        z_kde_values = kde_z(z_high_res)

        ## Mock z-values
        z_kde_mock = kde_z.resample(size=N_mock_data)[0]

        ## Removing any very small-z samples,
        ## making sure that all mock redshifts satisfy the low-z cut performed in the original data
        while np.min(z_kde_mock)<np.min(frb_input_data['z']):
            bool_small_z = z_kde_mock < np.min(frb_input_data['z'])
            z_kde_mock[bool_small_z] = kde_z.resample(size=np.sum(bool_small_z))[0]
            
    else:
        z_high_res=np.linspace(z_min, z_max, 10_000)
        z_kde_mock=rng.choice(z_high_res, size=N_mock_data, p=normalise(z_high_res**2))

    
    
    if pdf_modelling == 'Gaussian':
        
        ## Host DMs
        DM_mock_host = rng.normal(loc=DM_mu_host, scale=A_host_mock*DM_mu_host, size=N_mock_data)
        
        ## IGM DMs
        DM_theory_IGM = np.zeros_like(z_kde_mock)
        DM_mock_IGM = np.zeros_like(z_kde_mock)
        for idx, z in enumerate(z_kde_mock):   
            DM_theory_IGM[idx] = dispersion_measure(z_kde_mock[idx], H0=H0, Om=Omega_m, w=w, alpha=alpha, f_IGM_0 = f_IGM)
            DM_mock_IGM[idx] = rng.normal(loc=DM_theory_IGM[idx], scale=A_diff_mock*DM_theory_IGM[idx], size=1)
    
        true_values = pd.DataFrame()

        true_values[r"$\sigma_{\rm host}$"] = np.array([sigma_mu_mock])
        true_values[r"$D_v$"] = np.array([Dv_mock])
        true_values['HOf'] = np.array([HOf_mock])
        true_values[r"$e_{\mu}$"] = np.array(e_mu_mock)
    
    
    elif pdf_modelling == 'Realistic':
        
        ###########################
        #### UNDER construction ###
        ###########################
        
        true_values = pd.DataFrame()

        true_values[r"$A_{\rm host}$"] = np.array([A_host_mock])
        true_values[r"$A_{\rm diff}$"] = np.array([A_diff_mock])
        true_values['HOf'] = np.array([HUBBLE*OMEGA_BARYONS*f_IGM])
        true_values[r"$DM_{\rm host}$"] = np.array([DM_mu_host])
        
    else:
        print(f"Pdf modelling choice: {pdf_modelling} not compatible! Please choose between: 'Gaussian' or 'Realistic'.")
        
        
    mock_data = pd.DataFrame()

    ## Add FRB column (to be consistent with analysis script)
    mock_data['FRB'] = np.arange(len(mock_data))      

    mock_data['z'] = z_kde_mock
    mock_data['DM_ext'] = DM_mock_IGM+DM_mock_host/(1+z_kde_mock)
  
    
    return true_values, mock_data