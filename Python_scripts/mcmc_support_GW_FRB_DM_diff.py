##########################################
### Functions to run the MCMC analysis ###
##########################################


#######################
### General modules ###
#######################

from config import *
from cosmo_support import *
from support import *


####################
### MCMC modules ###
####################

import emcee
from multiprocessing import Pool, cpu_count

###################################
### Load interpolations for pdf ###
###################################

DATA_PATH = '../FRB_cosmo/interpolation/095_C0mean.npz'

def _load_and_create_interpolators():
    load_arrays = np.load(DATA_PATH)
    Sigmas = load_arrays['a']
    Errors = load_arrays['d']
    C0s = load_arrays['c'] 
    As = load_arrays['b']
    
    sigma_error_inter = interpolate.interp1d(Errors, Sigmas, kind=1, bounds_error=False,fill_value='extrapolate')
    C0_sigma_inter = interpolate.interp1d(Sigmas, C0s, kind=1, bounds_error=False,fill_value='extrapolate')
    A_sigma_inter = interpolate.interp1d(Sigmas, As, kind=1, bounds_error=False,fill_value='extrapolate')
    
    return Sigmas, Errors, C0s, As, sigma_error_inter, C0_sigma_inter, A_sigma_inter

Sigmas, Errors, C0s, As, sigma_error_inter, C0_sigma_inter, A_sigma_inter = _load_and_create_interpolators()

def reload_with_path(path):
    """reload"""
    global DATA_PATH, Sigmas, Errors, C0s, As, sigma_error_inter, C0_sigma_inter, A_sigma_inter
    DATA_PATH = path
    Sigmas, Errors, C0s, As, sigma_error_inter, C0_sigma_inter, A_sigma_inter = _load_and_create_interpolators()

########################################
### Load standard parameters for pdf ###
########################################

S=0.133
EXP_MU=182.937
SIGMA_HOST=0.605
DM_MWHALO=30
HOF=2.813

#######################
### Generate events ###
#######################

## Random choice of redshift
REDSHIFT_METHOD = 'rates'  # choose from 'rates', 'uniform', 'gaussian', 'lognormal' and 'powerlaw'

N_EVENTS = 50

z_range = np.linspace(0.25, 2.0, 500)
z_centre = draw_redshift_distribution(z_range, H0=HUBBLE, Omega_m=OMEGA_MATTER, N_draws=N_EVENTS, method=REDSHIFT_METHOD)

# Theoretical dL, fiducial cosmo
dL_centre = luminosity_distance(z=z_centre, H0=HUBBLE, Om=OMEGA_MATTER, w=W_LAMBDA)
# Theoretical DM, fiducial cosmo
DM_centre = dispersion_measure(z_centre, H0=HUBBLE, Om=OMEGA_MATTER)

###############################
### MCMC Analysis functions ###
###############################

z_array=np.linspace(0.25, 4.0, 8000)

p_selection = redshift_distribution(z_array=z_array, H0=HUBBLE, Omega_m=OMEGA_MATTER, method=REDSHIFT_METHOD)
p_selection=normalise(p_selection, x_array=z_array)

def log_likelihood(theta, data):
    # 
    # Calculate the log likelihood for a set of parameters given the data.

    # Args:
    #     theta: Array of parameters [F, HOf, sigma_host, e_mu]
    #    data: Pandas DataFrame containing FRB data

    # Returns:
    #   Log likelihood
    # 
    hubble, omega, w = theta

    log_like = 0.0

    try:
        for _, row in data.iterrows():
            ####### dL kde ######
            dL_gaussian = np.random.normal(row['dL_obs'], row['s_dL'], 1000)
            dL_gaussian = np.maximum(dL_gaussian, 0)
            GW_dL_kde = gaussian_kde(dL_gaussian)
            
            ######## p_DM(z) and p_dL(z) ########
            
            lum_distance=luminosity_distance(z=z_array, H0=hubble, Om=omega, w=w)
            DM_th_array=dispersion_measure(z=z_array, H0=hubble, Om=omega, w=w, alpha=0, f_IGM_0 = 0.84)
            Delta_array = row['DM_obs']/ DM_th_array
            
            p_DM=np.zeros_like(z_array)
            
            for idx, (z_val, Delta, DM_th) in enumerate(zip(z_array, Delta_array, DM_th_array)):
                error=f_variance_delta(S=S, z=z_val, Om=omega, w=w)

                sigma_diff=sigma_error_inter(error)
                C0=C0_sigma_inter(sigma_diff)
                A=A_sigma_inter(sigma_diff)
                
                if (np.isnan([error,C0,A,sigma_diff]).any()):
                    p_DM[idx]=0.0
                    # print(f'NaN found for error at z={z_val}, H0={hubble}, Om={omega}, w={w} for error={error}, C0={C0}, A={A}, sigma_diff={sigma_diff}')
                else:
                    p_DM[idx]=pdf_DM_cosmo(Delta=Delta, C_0=C0, A=A, sigma=sigma_diff, alpha=3, beta=3)/DM_th
            
            p_DM=normalise(p_DM, z_array)
            p_dL=normalise(GW_dL_kde(lum_distance), z_array)
            prob = np.trapz(p_selection*p_dL*p_DM, z_array)

            if prob > 0:
                log_like += np.log(prob)
            else:
                return -np.inf

        return log_like
    except Exception as e:
        print(f"Error in log_likelihood: {e} with parameters {theta}")
        return -np.inf


def log_prior(theta):
    """
    Calculate the log of the prior probability for a set of parameters.

    Args:
        theta: Array of parameters [F, HOf, sigma_host, e_mu]

    Returns:
        Log prior probability
    """
    hubble, omega, w = theta

    # Define your prior ranges here
    hubble_min, hubble_max = 40, 100 #0.016 # 0.2 # 2.0 #0.2 # Example range, adjust based on your model
    omega_min, omega_max = 0.0, 1.0  # Example range, adjust based on your model
    w_min, w_max = -3.0, 0.0  # Example range

    # Check if parameters are within prior ranges
    if (hubble_min <= hubble <= hubble_max and 
        omega_min <= omega <= omega_max and 
        w_min <= w <= w_max ):
        return 0.0  # Log(1) = 0, flat prior
    else:
        return -np.inf  # Log(0) = -inf, outside prior range       

def log_probability(theta, data):
    """
    Calculate the log probability (posterior) for a set of parameters.

    Args:
        theta: Array of parameters [F, HOf, sigma_host, e_mu]
        data: Pandas DataFrame containing FRB data

    Returns:
        Log posterior probability
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta, data)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll

def run_mcmc(data, initial_params, nwalkers=32, heating=10, nsteps=10000):
    """
    Run the MCMC analysis.

    Args:
        data: Pandas DataFrame containing FRB data
        initial_params: Initial parameter values [F, HOf, sigma_host, e_mu]
        nwalkers: Number of walkers
        nsteps: Number of steps per walker
        ndim: Number of dimensions (parameters)

    Returns:
        sampler: emcee sampler object with results
    """

    ndim=len(initial_params)

    # Set initial positions with small random offsets
    pos = initial_params + 0.1 * np.random.randn(nwalkers, ndim)
    # pos = [initial_params + _*1e-2 * np.random.randn(ndim) for _ in range(nwalkers)]

    for i in range(nwalkers):
        while log_prior(pos[i]) == -np.inf:
            pos[i] = initial_params + 0.01 * np.random.randn(ndim)

    # Set up the sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, 
            args=(data,), pool=pool,
            moves=[(emcee.moves.DEMove(), 0.8),
                   (emcee.moves.DESnookerMove(), 0.2)]
        )

        # Run the MCMC
        print("Running MCMC...")

        print("heating...")
        state = None
        with tqdm(total=heating) as pbar:
            for i, result in enumerate(sampler.sample(pos, iterations=heating)):
                pbar.update(1)
                state = result
                if i % 100 == 0:
                    # Calculate acceptance fraction periodically
                    acc_frac = np.mean(sampler.acceptance_fraction)
                    pbar.set_description(f"Acceptance fraction: {acc_frac:.3f}")

        print("main running...")
        with tqdm(total=nsteps) as pbar:
            for i, result in enumerate(sampler.sample(state.coords, iterations=nsteps)):
                pbar.update(1)

                # check acceptance fraction
                if i % 100 == 0:
                    acc_frac = np.mean(sampler.acceptance_fraction)
                    pbar.set_description(f"Acceptance fraction: {acc_frac:.3f}")

                    # if acceptance fraction always = 0，reset initial parameters
                    if i > 500 and acc_frac < 0.001:
                        print("warning: acceptance fraction too low，reset parameters or resun MCMC")

    # check acceptance fraction
    final_acc_frac = np.mean(sampler.acceptance_fraction)
    print(f"final acceptance fraction: {final_acc_frac:.3f}")

    if final_acc_frac < 0.01:
        print("warning: acceptance fraction too low，reset parameters or resun MCMC")

    return sampler


##################################################################################
### Analyse results
###


'''
Because running MCMC depend on proior and likelihood function defination, so only include analyze function here.
'''

def mcmc_analyze_results(sampler, burn_in=10, thin=15, target_prob=0.6827):
    """
    Analyze the MCMC results.
    
    Args:
        sampler: emcee sampler with results
        burn_in: Number of initial steps to discard
        thin: Thinning factor for samples
    
    Returns:
        samples: Flattened, burned-in, and thinned samples
        params_median: Median parameter values
        params_errors: Parameter uncertainties (16th and 84th percentiles)
    """
    # Discard burn-in, flatten and thin the samples
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    
    # Calculate the median and 16th and 84th percentiles for the parameters
    params_median = np.median(flat_samples, axis=0)
    params_lower = np.percentile(flat_samples, 50-target_prob*50, axis=0) # np.percentile(flat_samples, 16, axis=0)
    params_upper = np.percentile(flat_samples, 50+target_prob*50, axis=0) #np.percentile(flat_samples, 84, axis=0)
    
    # Calculate errors
    params_errors = [(params_upper[i] - params_lower[i]) / 2 for i in range(len(params_median))]
    
    return flat_samples, params_median, params_errors

def mcmc_plot_results(samples, param_names, savetitle=None, bins=30, target_prob=0.6827):
    """
    Plot the MCMC results.
    
    Args:
        samples: MCMC samples
        param_names: Names of the parameters
    """
    
    # Create corner plot
    
    fig = corner.corner(
        samples, 
        labels=param_names,
        quantiles=[0.5-target_prob/2, 0.5, 0.5+target_prob/2], # [0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        title_fmt='.3f',
        bins=bins,
        smooth=True,
        color='tab:blue'
    )
    
    if savetitle is not None:
        plt.savefig(savetitle+"_corner_plot.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot chains for each parameter
    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 2*len(param_names)), sharex=True)
    
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.plot(samples[:, i], 'k-', alpha=0.3)
        ax.set_ylabel(name)
        if i == (len(param_names)-1):
            ax.set_xlabel("Sample Number")
    
    plt.tight_layout()
    if savetitle is not None:
        plt.savefig(savetitle+"_chains.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()