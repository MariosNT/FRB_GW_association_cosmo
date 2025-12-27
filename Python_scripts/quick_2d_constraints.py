################################################
### Functions to reproduce 2D cosmo analysis ###
################################################

from config import *
from cosmo_support import *
from support import *

###############################################

def wei_2d_analysis(z_min=0.2, z_max=2.0, N_events=20, z_method='rates'):
    """
    Function that automates the Wei+2018 analysis: Recovers Om, and (Om, w).
    """
    
    ## Redshift range of analysis
    z_range = np.linspace(z_min, z_max, 500)

    ## Draw N_events sources for analysis
    z_obs = draw_redshift_distribution(z_range, H0=HUBBLE, Omega_m=OMEGA_MATTER, N_draws=N_events, method=z_method)

    dL_values_obs = np.zeros_like(z_obs)
    DM_values_obs = np.zeros_like(z_obs)
    s_dL_obs = np.zeros_like(z_obs)

    dLDM_obs = np.zeros_like(z_obs)
    s_dLDM_obs = np.zeros_like(z_obs)

    ## Observed values
    ## We perturb the distance measures, using Gaussians, around the theoretical values
    for idx, z_val in enumerate(z_obs): 
        dL_fid = luminosity_distance(z_val, HUBBLE, OMEGA_MATTER, w=W_LAMBDA)
        DM_fid = dispersion_measure(z_val, HUBBLE, OMEGA_MATTER, w=W_LAMBDA)

        s_dL_obs[idx] = sigma_dL(z_val, HUBBLE, OMEGA_MATTER, w=W_LAMBDA, method='Wei')
        s_DM_fid = SIGMA_DM

        dL_values_obs[idx] = rng.normal(dL_fid, s_dL_obs[idx])
        DM_values_obs[idx] = rng.normal(DM_fid, s_DM_fid)

        ## Joint dL-DM observation + error
        dLDM_obs[idx] = dL_values_obs[idx]*DM_values_obs[idx]
        s_dLDM_obs[idx] = sigma_dLDM(dL_values_obs[idx], DM_values_obs[idx], s_dL_obs[idx])       
        
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

    for idx_Om, Om in enumerate(tqdm(Omega_m_array)):
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



def ideal_distances_analysis(z_min=0.2, z_max=2.0, N_events=20, z_method='rates', Ho_res=50, Om_res=40, w_res=45, fit='2D'):
    """
    Function that automates our analysis: Recovers (H0, Om) & (H0, Om, w).

    Here we assume ideal luminosity distances, so we skip the step of transforming distances to redshifts
    and use directly the z_obs values, to compare between DM_obs and DM_theory.
    """
    
    ## Redshift range of analysis
    z_range = np.linspace(z_min, z_max, 500)

    z_obs = draw_redshift_distribution(z_range, H0=HUBBLE, Omega_m=OMEGA_MATTER, N_draws=N_events, method=z_method)

    DM_values_obs = np.zeros_like(z_obs)

    ## Create mock observations (following the fiducial cosmology + scatter)
    for idx, z_val in enumerate(z_obs): 
        DM_fid = dispersion_measure(z_val, HUBBLE, OMEGA_MATTER, w=W_LAMBDA)

        s_DM_fid = SIGMA_DM

        DM_values_obs[idx] = rng.normal(DM_fid, s_DM_fid)
        
        
    ## We create a grid of values for Om, Ho and w
    H0_array = np.linspace(10, 140, Ho_res)
    Omega_m_array = np.linspace(0.1, 0.5, Om_res)
    w_array = np.linspace(-3.0, 0.0, w_res)

    H_Om, Om_H0 = np.meshgrid(H0_array, Omega_m_array)
    H_w, w_H = np.meshgrid(H0_array, w_array)
    Om_w, w_Om = np.meshgrid(Omega_m_array, w_array)

    H3D_, Om3D_, w3D_ = np.meshgrid(H0_array, Omega_m_array, w_array)
    
    
    if fit=='2D':
        ## Fitting H0 & Om
        chi_fit_2D = np.zeros_like(H_Om)

        for idx_h, H0 in enumerate(tqdm(H0_array)):
            for idx_Om, Om in enumerate(Omega_m_array):
                chi_square = 0
                for idx_z, z_val in enumerate(z_obs):
                    DM = dispersion_measure(z_val, H0, Om, w=W_LAMBDA)

                    chi_square += (DM_values_obs[idx_z]-DM)**2/SIGMA_DM**2

                chi_fit_2D[idx_Om, idx_h] = chi_square 


        ## Finding the best-fit H0 & error [by marginalising over Om of the 2D model]
        P2d = normalise(np.exp(-chi_fit_2D/2))
        P_H0 = np.sum(P2d, axis=0)
        
        return z_obs, chi_fit_2D, H_Om, Om_H0, H0_array, P_H0
        
        
    elif fit=='3D':
        ## Fitting H0 & Om & w
        chi_fit_3D = np.zeros_like(H3D_)

        for idx_h, H0 in enumerate(tqdm(H0_array)):
            for idx_Om, Om in enumerate(Omega_m_array):
                for idx_w, W in enumerate(w_array):
                    chi_square = 0
                    for idx_z, z_val in enumerate(z_obs):
                        DM = dispersion_measure(z_val, H0, Om, w=W)

                        chi_square += (DM_values_obs[idx_z]-DM)**2/SIGMA_DM**2

                    chi_fit_3D[idx_Om, idx_h, idx_w] = chi_square


        ## In this part, we fix one of the three cosmological variables to their default value, 
        ## and plot the remaining 2D constraints (of the other two variables).

        index_W_LAMBDA = np.where(np.abs(w_array-W_LAMBDA)==np.min(np.abs(w_array-W_LAMBDA)))[0][0]
        chi_2D_H0Om = chi_fit_3D[:, :, index_W_LAMBDA]

        index_OMEGA_MATTER = np.where(np.abs(Omega_m_array-OMEGA_MATTER)==np.min(np.abs(Omega_m_array-OMEGA_MATTER)))[0][0]
        chi_2D_H0w = chi_fit_3D[index_OMEGA_MATTER, :, :]

        index_HUBBLE = np.where(np.abs(H0_array-HUBBLE)==np.min(np.abs(H0_array-HUBBLE)))[0][0]
        chi_2D_Omw = chi_fit_3D[:, index_HUBBLE, :]
        
        
        ## Finding the best-fit H0 & Om & w [by marginalising over parameters of the 3D model]

        P3d = np.exp(-chi_fit_3D/2)
        P3d_norm = np.sum(P3d)

        P3d /= P3d_norm

        P2d_H0Om = np.sum(P3d, axis=2)
        P2d_H0w = np.sum(P3d, axis=0)
        P2d_Omw = np.sum(P3d, axis=1)
        
        t_contours_H0Om = posterior_contour_2D(P2d_H0Om, n=1000)
        t_contours_H0w = posterior_contour_2D(P2d_H0w, n=1000)
        t_contours_Omw = posterior_contour_2D(P2d_Omw, n=1000)
        
        return z_obs, H_Om, Om_H0, H_w, w_H, Om_w, w_Om, chi_2D_H0Om, chi_2D_H0w, chi_2D_Omw, P2d_H0Om, P2d_H0w, P2d_Omw, t_contours_H0Om, t_contours_H0w, t_contours_Omw
    
    else:
        print("Wrong fit method! Choose between '2D' or '3D'!")

    
