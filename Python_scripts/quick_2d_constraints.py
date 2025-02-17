################################################
### Functions to reproduce 2D cosmo analysis ###
################################################

from config import *
from cosmo_support import *
from support import normalise

###############################################

def wei_2d_analysis(z_min=0.2, z_max=2.0, N_events=20, z_method='rates'):
    
    ## Redshift range of analysis
    z_range = np.linspace(z_min, z_max, 500)

    z_obs = draw_redshift_distribution(z_range, H0=HUBBLE, Omega_m=OMEGA_MATTER, N_draws=N_events, method=z_method)

    dL_values_fid = np.zeros_like(z_range)
    DM_values_fid = np.zeros_like(z_range)
    dLDM_fid = np.zeros_like(z_range)

    dL_values_obs = np.zeros_like(z_obs)
    DM_values_obs = np.zeros_like(z_obs)
    s_dL_obs = np.zeros_like(z_obs)

    dLDM_obs = np.zeros_like(z_obs)
    s_dLDM_obs = np.zeros_like(z_obs)

    ## obs values
    for idx, z_val in enumerate(z_obs): 
        dL_fid = luminosity_distance(z_val, HUBBLE, OMEGA_MATTER, w=W_LAMBDA)
        DM_fid = dispersion_measure(z_val, HUBBLE, OMEGA_MATTER, w=W_LAMBDA)

        s_dL_obs[idx] = sigma_dL(z_val, HUBBLE, OMEGA_MATTER, w=W_LAMBDA, method='Wei')
        s_DM_fid = SIGMA_DM

        dL_values_obs[idx] = np.random.normal(dL_fid, s_dL_obs[idx])
        DM_values_obs[idx] = np.random.normal(DM_fid, s_DM_fid)

        dLDM_obs[idx] = dL_values_obs[idx]*DM_values_obs[idx]
        s_dLDM_obs[idx] = sigma_dLDM(dL_values_obs[idx], DM_values_obs[idx], s_dL_obs[idx])

    ## fid values
    for idx, z_val in enumerate(z_range): 
        dL_values_fid[idx] = luminosity_distance(z_val, HUBBLE, OMEGA_MATTER, w=W_LAMBDA)
        DM_values_fid[idx] = dispersion_measure(z_val, HUBBLE, OMEGA_MATTER, w=W_LAMBDA)   
        dLDM_fid[idx] = dLDM_measure(z_val, HUBBLE, OMEGA_MATTER, w=W_LAMBDA)
        
        
    ## We create a grid of values for Om and Ho
    Omega_m_array = np.linspace(0.0, 0.6, 100)
    w_array = np.linspace(-3.0, 0.0, 140)

    Om_, w_ = np.meshgrid(Omega_m_array, w_array)
    
    ## Fitting Om        
    chi_fit = np.zeros_like(Omega_m_array)

    for idx_Om, Om in enumerate(Omega_m_array):
        chi_square = 0
        for idx_z, z_val in enumerate(z_obs):
            dLDM = dLDM_measure(z_val, HUBBLE, Om, w=W_LAMBDA)        

            chi_square += (dLDM_obs[idx_z]-dLDM)**2/s_dLDM_obs[idx_z]**2

        chi_fit[idx_Om] = chi_square    
        
        
    ## Finding the best-fit Om & error 

    ## Find Om value
    chi_1D_min = np.min(chi_fit)
    id_Om_min = np.where(chi_fit==chi_1D_min)[0][0]

    ## Find sigma
    sigma1_chi_fit = chi_1D_min+1
    delta_chi = np.abs(chi_fit-sigma1_chi_fit)
    delta_chi_min = np.min(delta_chi)
    id_Om_sigma_1 = np.where(delta_chi==delta_chi_min)[0][0]

    ## Transform to the physical parameter of interest
    Omega_fit = Omega_m_array[id_Om_min]
    sigma_Om = np.abs(Omega_m_array[id_Om_min]-Omega_m_array[id_Om_sigma_1])      
    
    ## Fitting Om & w
    chi_fit_2D = np.zeros_like(Om_)

    for idx_Om, Om in enumerate(Omega_m_array):
        for idx_w, w_val in enumerate(w_array):
            chi_square = 0
            for idx_z, z_val in enumerate(z_obs):
                dLDM = dLDM_measure(z_val, HUBBLE, Om, w=w_val)        

                chi_square += (dLDM_obs[idx_z]-dLDM)**2/s_dLDM_obs[idx_z]**2

            chi_fit_2D[idx_w, idx_Om] = chi_square       
            
            
    ## Finding the best-fit Om & error [by marginalizing over w of the 2D model]

    P2d = np.exp(-chi_fit_2D/2)  # Finding the pdf
    P2d /= np.sum(P2d)  # Normalizing the pdf
    P_Om = np.sum(P2d, axis=0)  # Marginalizing to get the parameter of interest
    
    return z_obs, dL_values_obs, s_dL_obs, DM_values_obs, SIGMA_DM, chi_fit_2D, Om_, w_, Omega_m_array, Omega_fit, sigma_Om, P_Om