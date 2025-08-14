##########################################
### Flexible MCMC Analysis Framework ###
##########################################

from config import *
from cosmo_support import *
from support import *

import emcee
from multiprocessing import Pool, cpu_count
from abc import ABC, abstractmethod
import inspect
from functools import partial

###################################
### Base Classes for Flexibility ###
###################################

class MCMCConfig:
    """Configuration class for MCMC parameters and setup"""
    
    def __init__(self, param_names, param_bounds, param_initial=None):
        """
        Initialize MCMC configuration
        
        Args:
            param_names: List of parameter names
            param_bounds: Dict of {param_name: (min, max)} or list of tuples [(min, max)]
            param_initial: Initial parameter values (if None, uses midpoint of bounds)
        """
        self.param_names = param_names
        self.ndim = len(param_names)
        
        # Handle bounds format
        if isinstance(param_bounds, dict):
            self.param_bounds = [param_bounds[name] for name in param_names]
        else:
            self.param_bounds = param_bounds
            
        # Set initial parameters
        if param_initial is None:
            self.param_initial = [(bounds[0] + bounds[1]) / 2 for bounds in self.param_bounds]
        else:
            self.param_initial = param_initial
    
    def get_param_dict(self, theta):
        """Convert parameter array to dictionary"""
        return dict(zip(self.param_names, theta))

class LikelihoodFunction(ABC):
    """Abstract base class for likelihood functions"""
    
    def __init__(self, interpolator_manager=None):
        """
        Initialize likelihood function
        
        Args:
            interpolator_manager: InterpolatorManager instance (if None, uses global)
        """
        if interpolator_manager is None:
            self.interpolator_manager = get_interpolator_manager()
        else:
            self.interpolator_manager = interpolator_manager
    
    @abstractmethod
    def log_likelihood(self, theta, data):
        """Calculate log likelihood"""
        pass
    
    @abstractmethod
    def log_prior(self, theta, config):
        """Calculate log prior"""
        pass
    
    def get_interpolator(self, name):
        """Convenience method to get interpolator"""
        return self.interpolator_manager.get_interpolator(name)

class StandardCosmologyLikelihood(LikelihoodFunction):
    """Original cosmology likelihood function"""
    
    def __init__(self, interpolator_manager=None):
        """
        Initialize with interpolator manager
        
        Args:
            interpolator_manager: InterpolatorManager instance
        """
        super().__init__(interpolator_manager)
        
        if not self.interpolator_manager.is_loaded():
            raise ValueError("Interpolators not loaded successfully")
    
    def log_likelihood(self, theta, data):
        """Original log likelihood function with interpolators from manager"""
        hubble, omega, w = theta
        
        # Get interpolators
        sigma_error_inter = self.get_interpolator('sigma_error_inter')
        C0_sigma_inter = self.get_interpolator('C0_sigma_inter')
        A_sigma_inter = self.get_interpolator('A_sigma_inter')
        
        # Standard parameters
        S = 0.133
        z_array = np.linspace(0.2, 3.0, 500)
        p_selection = redshift_distribution(z_array=z_array, H0=HUBBLE, Omega_m=OMEGA_MATTER, method='rates')
        
        log_like = 0.0

        try:
            for _, row in data.iterrows():
                # dL kde
                dL_gaussian = np.random.normal(row['dL_obs'], row['s_dL'], 1000)
                dL_gaussian[dL_gaussian<0] = 0
                GW_dL_kde = gaussian_kde(dL_gaussian)
                
                # p_DM(z) and p_dL(z)
                lum_distance = luminosity_distance(z=z_array, H0=hubble, Om=omega, w=w)
                DM_th_array = dispersion_measure(z=z_array, H0=hubble, Om=omega, w=w, alpha=0, f_IGM_0=0.84)
                Delta_array = row['DM_obs'] / DM_th_array
                
                p_DM = np.zeros_like(z_array)
                
                for idx, (z_val, Delta, DM_th) in enumerate(zip(z_array, Delta_array, DM_th_array)):
                    error = f_variance_delta(S=S, z=z_val, Om=omega, w=w)
                    
                    sigma_diff = sigma_error_inter(error)
                    C0 = C0_sigma_inter(sigma_diff)
                    A = A_sigma_inter(sigma_diff)
                    
                    if (np.isnan([error, C0, A, sigma_diff]).any()):
                        p_DM[idx] = 0.0
                    else:
                        p_DM[idx] = pdf_DM_cosmo(Delta=Delta, C_0=C0, A=A, sigma=sigma_diff, alpha=3, beta=3) / DM_th
                
                p_dL = normalise(GW_dL_kde(lum_distance))
                prob = np.trapz(p_selection * p_dL * p_DM, z_array)
                
                if prob > 0:
                    log_like += np.log(prob)
                else:
                    return -np.inf

            return log_like
        except Exception as e:
            print(f"Error in log_likelihood: {e} with parameters {theta}")
            return -np.inf
    
    def log_prior(self, theta, config):
        """Standard uniform prior"""
        for i, (val, bounds) in enumerate(zip(theta, config.param_bounds)):
            if not (bounds[0] <= val <= bounds[1]):
                return -np.inf
        return 0.0

class CustomLikelihoodFunction(LikelihoodFunction):
    """Custom likelihood function that can be defined by user"""
    
    def __init__(self, likelihood_func, prior_func=None, interpolator_manager=None):
        """
        Initialize with custom functions
        
        Args:
            likelihood_func: Function that takes (theta, data, interpolators) and returns log likelihood
            prior_func: Function that takes (theta, config) and returns log prior
                       If None, uses uniform prior
            interpolator_manager: InterpolatorManager instance (if None, uses global)
        """
        super().__init__(interpolator_manager)
        self._likelihood_func = likelihood_func
        self._prior_func = prior_func
    
    def log_likelihood(self, theta, data):
        # Check if likelihood function accepts interpolators argument
        sig = inspect.signature(self._likelihood_func)
        if 'interpolators' in sig.parameters:
            return self._likelihood_func(theta, data, self.interpolator_manager.get_all_interpolators())
        else:
            return self._likelihood_func(theta, data)
    
    def log_prior(self, theta, config):
        if self._prior_func is None:
            # Default uniform prior
            for i, (val, bounds) in enumerate(zip(theta, config.param_bounds)):
                if not (bounds[0] <= val <= bounds[1]):
                    return -np.inf
            return 0.0
        else:
            return self._prior_func(theta, config)

###################################
### Flexible MCMC Runner Class ###
###################################

class FlexibleMCMC:
    """Flexible MCMC runner that can handle different parameter sets and likelihood functions"""
    
    def __init__(self, config, likelihood_function):
        """
        Initialize MCMC runner
        
        Args:
            config: MCMCConfig object
            likelihood_function: LikelihoodFunction object
        """
        self.config = config
        self.likelihood_function = likelihood_function
    
    def log_probability(self, theta, data):
        """Calculate log probability (posterior)"""
        lp = self.likelihood_function.log_prior(theta, self.config)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.likelihood_function.log_likelihood(theta, data)
        if not np.isfinite(ll):
            return -np.inf
        
        return lp + ll
    
    def run_mcmc(self, data, nwalkers=32, heating=10, nsteps=10000, moves=None):
        """
        Run MCMC analysis
        
        Args:
            data: Data for likelihood calculation
            nwalkers: Number of walkers
            heating: Number of heating steps
            nsteps: Number of main steps
            moves: Custom moves for sampler
        
        Returns:
            sampler: emcee sampler object
        """
        # Set initial positions
        pos = self.config.param_initial + 0.1 * np.random.randn(nwalkers, self.config.ndim)
        
        # Ensure initial positions are within prior
        for i in range(nwalkers):
            while self.likelihood_function.log_prior(pos[i], self.config) == -np.inf:
                pos[i] = self.config.param_initial + 0.01 * np.random.randn(self.config.ndim)
        
        # Default moves
        if moves is None:
            moves = [(emcee.moves.DEMove(), 0.8),
                     (emcee.moves.DESnookerMove(), 0.2)]
        
        # Set up sampler
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, self.config.ndim, self.log_probability,
                args=(data,), pool=pool, moves=moves
            )
            
            # Run heating
            print("Running heating...")
            state = None
            with tqdm(total=heating) as pbar:
                for i, result in enumerate(sampler.sample(pos, iterations=heating)):
                    pbar.update(1)
                    state = result
                    if i % 100 == 0:
                        acc_frac = np.mean(sampler.acceptance_fraction)
                        pbar.set_description(f"Acceptance fraction: {acc_frac:.3f}")
            
            # Run main chain
            print("Running main chain...")
            with tqdm(total=nsteps) as pbar:
                for i, result in enumerate(sampler.sample(state.coords, iterations=nsteps)):
                    pbar.update(1)
                    if i % 100 == 0:
                        acc_frac = np.mean(sampler.acceptance_fraction)
                        pbar.set_description(f"Acceptance fraction: {acc_frac:.3f}")
                        
                        if i > 500 and acc_frac < 0.001:
                            print("Warning: acceptance fraction too low")
        
        final_acc_frac = np.mean(sampler.acceptance_fraction)
        print(f"Final acceptance fraction: {final_acc_frac:.3f}")
        
        if final_acc_frac < 0.01:
            print("Warning: acceptance fraction too low")
        
        return sampler

###################################
### Analysis Functions ###
###################################

def mcmc_analyze_results(sampler, burn_in=10, thin=15, target_prob=0.6827):
    """Analyze MCMC results (unchanged from original)"""
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    
    params_median = np.median(flat_samples, axis=0)
    params_lower = np.percentile(flat_samples, 50-target_prob*50, axis=0)
    params_upper = np.percentile(flat_samples, 50+target_prob*50, axis=0)
    
    params_errors = [(params_upper[i] - params_lower[i]) / 2 for i in range(len(params_median))]
    
    return flat_samples, params_median, params_errors

def mcmc_plot_results(samples, param_names, savetitle=None, bins=30, target_prob=0.6827):
    """Plot MCMC results (unchanged from original)"""
    fig = corner.corner(
        samples, 
        labels=param_names,
        quantiles=[0.5-target_prob/2, 0.5, 0.5+target_prob/2],
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

###################################
### Example Usage Functions ###
###################################

def create_cosmology_config():
    """Create standard cosmology configuration"""
    param_names = ['H0', 'Omega_m', 'w']
    param_bounds = [(40, 100), (0.0, 1.0), (-3.0, 0.0)]
    param_initial = [70, 0.3, -1.0]
    
    return MCMCConfig(param_names, param_bounds, param_initial)

def create_custom_config(param_names, param_bounds, param_initial=None):
    """Create custom configuration"""
    return MCMCConfig(param_names, param_bounds, param_initial)

# Example custom likelihood function
def example_custom_likelihood(theta, data):
    """Example of a custom likelihood function"""
    # Unpack parameters - adapt this to your needs
    param1, param2, param3 = theta
    
    # Your custom likelihood calculation here
    log_like = 0.0
    
    try:
        for _, row in data.iterrows():
            # Your custom calculation using param1, param2, param3
            # and data from the row
            predicted_value = param1 * row['x'] + param2
            residual = row['y'] - predicted_value
            log_like += -0.5 * (residual / param3)**2 - 0.5 * np.log(2 * np.pi * param3**2)
        
        return log_like
    except Exception as e:
        print(f"Error in custom likelihood: {e}")
        return -np.inf

def example_custom_prior(theta, config):
    """Example of a custom prior function"""
    param1, param2, param3 = theta
    
    # Custom prior constraints
    if param3 <= 0:  # param3 must be positive
        return -np.inf
    
    # Check bounds
    for i, (val, bounds) in enumerate(zip(theta, config.param_bounds)):
        if not (bounds[0] <= val <= bounds[1]):
            return -np.inf
    
    # Add custom prior shape (e.g., Gaussian prior on param1)
    log_prior = -0.5 * ((param1 - 70) / 10)**2  # Gaussian prior centered at 70 with sigma=10
    
    return log_prior