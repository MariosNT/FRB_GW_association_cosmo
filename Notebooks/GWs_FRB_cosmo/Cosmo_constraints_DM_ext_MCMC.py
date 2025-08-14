# A .py version of Cosmo_constraints_DM_ext_MCMC.ipynb for running in the cluster

import sys
sys.path.append('../../Python_scripts')
sys.path.append('../FRB_cosmo/FRB_cosmo/interpolation')

## Import packages
from config import *
from support import *
from cosmo_support import *

### MCMCM packages
path='../FRB_cosmo/interpolation/095_C0mean.npz' # './interpolation/StandardD_C0mean.npz' 

import mcmc_support_GW_FRB_DM_ext
mcmc_support_GW_FRB_DM_ext.DATA_PATH = path
mcmc_support_GW_FRB_DM_ext.reload_with_path(path)
from mcmc_support_GW_FRB_DM_ext import *


# initial parameters
Hubble0 = 70
e_mu0 = 150
sigma_host0 = 0.5

# MCMC parameters
N_WALKERS = 100
HEATING = 10
N_STEPS = 100

# Find use quadratic function may get negative error in some large redshift

def GW_error_LVK(z, H0, Om, w=-1, order=1):
    if (order==1):
        a0=19.202
        a1=86.773
        dL=luminosity_distance(z, H0, Om, w)
        return (a1*z+a0)*dL
    elif (order==2):
        a0=17.015
        a1=131.750
        a2=-174.911
        dL=luminosity_distance(z, H0, Om, w)
        return (a2*z*z+a1*z+a0)*dL
    else:
        print('Choose order from 1 or 2. Use default instead')
        return sigma_dL(z_val, H0, Om, w=w, method='Wei')
    
def GW_error_CE(z, H0, Om, w=-1, order=2):
    if (order==1):
        a0=9.792
        a1=9.704
        dL=luminosity_distance(z, H0, Om, w)
        return (a1*z+a0)*dL
    elif (order==2):
        a0=7.649
        a1=18.581
        a2=-4.559
        dL=luminosity_distance(z, H0, Om, w)
        return (a2*z*z+a1*z+a0)*dL
    else:
        print('Choose order from 1 or 2. Use default instead')
        return sigma_dL(z_val, H0, Om, w=w, method='Wei')
    
## Choice of observed luminosity distance
sigma_dL = 0.1*dL_centre
dL_obs_centre = np.random.normal(dL_centre, sigma_dL)

DM_obs_centre=np.zeros_like(z_centre)
s_DM_obs = np.zeros_like(z_centre)

for idx, z_val in enumerate(z_centre):
    DM_obs_centre[idx], s_DM_obs[idx] = DM_ext_sampling(z=z_val, 
                                                       S=S, HOF=HOF, SIGMA_HOST=SIGMA_HOST, EXP_MU=EXP_MU,
                                                       sigma_error_inter=sigma_error_inter,
                                                       C0_sigma_inter=C0_sigma_inter,
                                                       A_sigma_inter=A_sigma_inter,
                                                       Om=OMEGA_MATTER, w=W_LAMBDA, N_draws=1, int_N=2000
                                                       )

events=pd.DataFrame({
    'z': z_centre,
    'dL': dL_centre,
    'dL_obs': dL_obs_centre,
    's_dL': sigma_dL,
    'DM': DM_centre,
    'DM_obs': DM_obs_centre,
    's_DM': s_DM_obs
})

# Define initial parameters: [F, HOf, sigma_host, e_mu]
initial_params = np.array([Hubble0, e_mu0, sigma_host0])

# Run MCMC
sampler = run_mcmc(events, initial_params, nwalkers=N_WALKERS, heating=HEATING, nsteps=N_STEPS)
    
# Analyze results
samples, params_median, params_errors = mcmc_analyze_results(sampler, burn_in=HEATING)

# Print results
param_names = [r'$ H_0$ ', r'$ exp(\mu)$ ', r'$ \sigma_{\rm host}$ ']
print("MCMC Results:")
for i, name in enumerate(param_names):
    print(f"{name} = {params_median[i]:.3f} ± {params_errors[i]:.3f}")

# Save samples to file for later analysis if needed
np.save('./posterior/GW_FRB_MCMC_DM_ext.npy', samples)

mcmc_plot_results(samples, param_names, savetitle='MCMC_DM_ext')