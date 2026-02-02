# A .py script for data generation

# # Example for loading the saved data
# # import pickle
# print(f"Loading data from {DATA_FILE}...")
# with open(DATA_FILE, 'rb') as f:
#     saved_data = pickle.load(f)
    
# z_centre = saved_data['z_centre']

# dL_obs_centre = saved_data['dL_obs_centre']
# sigma_dL = saved_data['sigma_dL']

# DM_diff_obs = saved_data['DM_diff_obs']
# sigma_DM_diff = saved_data['sigma_DM_diff']

# DM_ext_obs = saved_data['DM_ext_obs']
# sigma_DM_ext = saved_data['sigma_DM_ext']

# # Also load the theoretical values
# dL_centre = saved_data['dL_centre']
# DM_centre = saved_data['DM_centre']

# print(f"Successfully loaded {len(z_centre)} events from saved data.")

import sys
sys.path.append('../../Python_scripts')
sys.path.append('../FRB_cosmo/FRB_cosmo/interpolation')

## Import packages
from config import *
from support import *
from cosmo_support import *

import pickle
import os

from pathlib import Path

# savefile
DATA_FILE = './checkpoint/data_2.pkl'
DATA_FIG='./plot/data_2.pdf'

ITP_PATH = '../FRB_cosmo/interpolation/095_C0mean.npz'
interpolations = np.load(f'../Realistic_sources/quantile_linear_interpolations.npz')

REDSHIFT_METHOD = 'rates'  # choose from 'rates', 'uniform', 'gaussian', 'lognormal' and 'powerlaw'

N_EVENTS = 50
Error_factor = 1.0   # times the error in event generation, =1, set other value for test

# Redshift range for events, default (0.25, 2.0)
Z_min = 0.2
Z_max = 2.0

########################################
### Load standard parameters for pdf ###
########################################

S=0.133
S_LN=0.06
EXP_MU=182.937
SIGMA_HOST=0.605
HOF=2.78318 # 2.813

###################################
### Load interpolations for pdf ###
###################################

def _load_and_create_interpolators():
    load_arrays = np.load(ITP_PATH)
    Sigmas = load_arrays['a']
    Errors = load_arrays['d']
    C0s = load_arrays['c'] 
    As = load_arrays['b']
    
    sigma_error_inter = interpolate.interp1d(Errors, Sigmas, kind=1, bounds_error=False,fill_value='extrapolate')
    C0_sigma_inter = interpolate.interp1d(Sigmas, C0s, kind=1, bounds_error=False,fill_value='extrapolate')
    A_sigma_inter = interpolate.interp1d(Sigmas, As, kind=1, bounds_error=False,fill_value='extrapolate')
    
    return Sigmas, Errors, C0s, As, sigma_error_inter, C0_sigma_inter, A_sigma_inter

Sigmas, Errors, C0s, As, sigma_error_inter, C0_sigma_inter, A_sigma_inter = _load_and_create_interpolators()
    
#######################
### DL error model ###
#######################

LVK_linear = interpolations['LVK_interpolation']
CE_linear = interpolations['CE_interpolation']

# Find use quadratic function may get negative error in some large redshift

def func_lin(x, a0, a1):
    return a0+a1*x


def GW_error_LVK(z, H0, Om, w=-1, order=1):
    if (order==1):
        sigma_ratio = func_lin(z, *LVK_linear)/100
        dL=luminosity_distance(z, H0, Om, w)
        return sigma_ratio*dL
    elif (order==2):
        a0=17.015
        a1=131.750
        a2=-174.911
        dL=luminosity_distance(z, H0, Om, w)
        return (a2*z*z+a1*z+a0)*dL
    else:
        print('Choose order from 1 or 2. Use default instead')
        return sigma_dL(z_val, H0, Om, w=w, method='Wei')
    
def GW_error_CE(z, H0, Om, w=-1, order=1):
    if (order==1):
        sigma_ratio = func_lin(z, *CE_linear)/100
        dL=luminosity_distance(z, H0, Om, w)
        return sigma_ratio*dL
    elif (order==2):
        a0=7.649
        a1=18.581
        a2=-4.559
        dL=luminosity_distance(z, H0, Om, w)
        return (a2*z*z+a1*z+a0)*dL
    else:
        print('Choose order from 1 or 2. Use default instead')
        return sigma_dL(z_val, H0, Om, w=w, method='Wei')

#######################
### Generate events ###
#######################
    
if os.path.exists(DATA_FILE):
    print(f"RESUME=False: Removing old save data {DATA_FILE}...")
    os.remove(DATA_FILE)
    
# Generate new data
print("Generating new simulation data...")
    
z_range = np.linspace(Z_min, Z_max, 1000)
z_centre = draw_redshift_distribution(z_range, H0=HUBBLE, Omega_m=OMEGA_MATTER, 
                                          N_draws=N_EVENTS, method=REDSHIFT_METHOD)

# Theoretical dL, fiducial cosmo
dL_centre = luminosity_distance(z=z_centre, H0=HUBBLE, Om=OMEGA_MATTER, w=W_LAMBDA)
# Theoretical DM, fiducial cosmo
DM_centre = dispersion_measure(z_centre, H0=HUBBLE, Om=OMEGA_MATTER)

## Choice of observed luminosity distance
# Use this for fixed error/redshift
# sigma_dL = 0.1*dL_centre

# Use this for redshift dependent errors
sigma_dL = Error_factor * GW_error_CE(z_centre, H0=HUBBLE, Om=OMEGA_MATTER)

dL_obs_centre = np.random.normal(dL_centre, sigma_dL)

DM_diff_obs = np.zeros_like(z_centre)
sigma_DM_diff = np.zeros_like(z_centre)

DM_ext_obs = np.zeros_like(z_centre)
sigma_DM_ext = np.zeros_like(z_centre)

DM_diff_obs_ln = np.zeros_like(z_centre)
sigma_DM_diff_ln = np.zeros_like(z_centre)

DM_ext_obs_ln = np.zeros_like(z_centre)
sigma_DM_ext_ln = np.zeros_like(z_centre)

for idx, z_val in enumerate(z_centre):
    print(f"Processing event {idx+1}/{N_EVENTS}...")
    DM_diff_obs[idx], sigma_DM_diff[idx] = \
        DM_diff_sampling(z=z_val, 
                        S=S, HOF=HOF,
                        sigma_error_inter=sigma_error_inter,
                        C0_sigma_inter=C0_sigma_inter,
                        A_sigma_inter=A_sigma_inter,
                        H0=HUBBLE, f_diff=0.84, f_diff_alpha=0,
                        Om=OMEGA_MATTER, w=W_LAMBDA, N_draws=1,
                        mode=None, #'standard'
                        Error_factor = Error_factor
                        )
        
    DM_ext_obs[idx], sigma_DM_ext[idx] = \
        DM_ext_sampling_fast(z=z_val, 
                            S=S, HOF=HOF, SIGMA_HOST=SIGMA_HOST, EXP_MU=EXP_MU,
                            sigma_error_inter=sigma_error_inter,
                            C0_sigma_inter=C0_sigma_inter,
                            A_sigma_inter=A_sigma_inter,
                            Om=OMEGA_MATTER, w=W_LAMBDA, N_draws=1, int_N=1000, 
                            Error_factor = Error_factor
                            )
        
    DM_diff_obs_ln[idx], sigma_DM_diff_ln[idx] = DM_diff_ln_sampling(z=z_val, S=S_LN)
        
    DM_ext_obs_ln[idx], sigma_DM_ext_ln[idx] = DM_ext_ln_sampling(z=z_val, S=S_LN, SIGMA_HOST=SIGMA_HOST, EXP_MU=EXP_MU)

#################
### Save data ###
#################

save_data = {
    # events redshifts
    'z_centre': z_centre,
    # DL data
    'dL_obs_centre': dL_obs_centre,
    'sigma_dL': sigma_dL,
    # DM_diff data
    'DM_diff_obs': DM_diff_obs,
    'sigma_DM_diff': sigma_DM_diff,
    'DM_diff_obs_ln': DM_diff_obs_ln,
    'sigma_DM_diff_ln': sigma_DM_diff_ln,
    # DM_ext data
    'DM_ext_obs': DM_ext_obs,
    'sigma_DM_ext': sigma_DM_ext,
    'DM_ext_obs_ln': DM_ext_obs_ln,
    'sigma_DM_ext_ln': sigma_DM_ext_ln,
    # Theoratical values (DM is the DM_diff)
    'dL_centre': dL_centre,
    'DM_centre': DM_centre,
    # constant
    'REDSHIFT_METHOD': REDSHIFT_METHOD,
    'S': S,
    'S_LN': S_LN,
    'EXP_MU': EXP_MU,
    'SIGMA_HOST': SIGMA_HOST,
    'HOF': HOF,
    'Z_min': Z_min,
    'Z_max': Z_max
}
    
directory = os.path.dirname(DATA_FILE)
if directory:
    os.makedirs(directory, exist_ok=True)
with open(DATA_FILE, 'wb') as f:
    pickle.dump(save_data, f)

print(f"Data saved to {DATA_FILE}")

#################
### Plot data ###
#################

fig = plt.figure(figsize=(18, 5))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

ax1.plot(np.sort(z_centre), np.sort(dL_centre))
ax1.errorbar(z_centre, dL_obs_centre, yerr=sigma_dL, marker='o', ls='', ms=3, c='r', label=f'CE')
ax1.set_ylabel(r'$d_{L}$ [Mpc]')
ax1.set_xlabel(r'$z$')
ax1.legend(loc='upper left')

ax2.plot(np.sort(z_centre), np.sort(DM_centre))
ax2.errorbar(z_centre, DM_diff_obs, yerr=sigma_DM_diff, marker='o', ls='', ms=3, label=f'Macquart\' PDF')
ax2.errorbar(z_centre, DM_diff_obs_ln, yerr=sigma_DM_diff_ln, marker='o', ls='', ms=3, label=f'Log-normal PDF')
ax2.legend(loc='upper left')
ax2.set_ylabel(r'$DM_{\rm diff}$ [pc/cm$^3$]')
ax2.set_xlabel(r'$z$')

ax3.plot(np.sort(z_centre), np.sort(DM_centre)+100)
ax3.errorbar(z_centre, DM_ext_obs, yerr=sigma_DM_ext, marker='o', ls='', ms=3, label=f'Macquart\' PDF')
ax3.errorbar(z_centre, DM_ext_obs_ln, yerr=sigma_DM_ext_ln, marker='o', ls='', ms=3, label=f'Log-normal PDF')
ax3.legend(loc='upper left')
ax3.set_ylabel(r'DM$_{\rm ext}$ [pc/cm$^3$]')
ax3.set_xlabel(r'$z$')

Path(DATA_FIG).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(DATA_FIG)
# plt.tight_layout()