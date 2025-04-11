###############################################
### Functions for the cosmological analysis ###
###############################################

from config import *
from support import normalise


###############################################

def Hubble_function(z, H0, Omega_m):
    """
    Hubble function
    """
    return np.sqrt(H0**2*(Omega_m*(1+z)**3+(1-Omega_m)))


def rate_function(z):
    rate = (1+2*z)*(z<=1)+3/4*(5-z)*(z>1)*(z<5)
    
    return rate

def D_comoving(z, H0, Omega_m):
    return FlatLambdaCDM(H0, Omega_m).comoving_distance(z).value


###############################################

def distribution_redshift_Gaussian(z, epi=-7.553):
    """
    Gaussian time-delay model for NSBH merger delay models in 
    [2011.02717]
    """
    
    f=(1+z)**(3.879*epi)+((1+z)/73.5)**(-0.4901*epi)+((1+z)/3.672)**(-5.691*epi)+((1+z)/3.411)**(-11.46*epi)+((1+z)/3.546)**(-16.38*epi)+((1+z)/3.716)**(-20.66*epi)
    f=f**(1/epi)
    
    return normalise(f)


def distribution_redshift_LogNormal(z, epi=-5.51):
    """
    LogNormal time-delay model for NSBH merger delay models in 
    [2011.02717]
    """
    
    f=(1+z)**(4.131*epi)+((1+z)/22.37)**(-0.5789*epi)+((1+z)/2.978)**(-4.735*epi)+((1+z)/2.749)**(-10.77*epi)+((1+z)/2.867)**(-17.51*epi)+((1+z)/3.04)**(-(0.08148+z**0.574/0.08682)*epi)
    f=f**(1/epi)
    
    return normalise(f)


def distribution_redshift_Power(z, epi=-8.161):
    """
    Power-law time-delay model for NSBH merger delay models in 
    [2011.02717]
    """
    
    f=(1+z)**(1.895*epi)+((1+z)/5.722)**(-3.759*epi)+((1+z)/11.55)**(-0.7426*epi)
    f=f**(1/epi)
    
    return normalise(f)


def draw_redshift_distribution(z_array, H0=HUBBLE, Omega_m=OMEGA_MATTER, N_draws=50, method='rates'):
    """
    Function that generates random redshifts for our GWs/FRB events.

    Input
    ----------
    z_array : redshift range from which to draw samples
    
    H0 : Hubble constant [km/s/Mpc]
    
    Omega_m : Omega matter
    
    N_draws : Number of mock redshifts to draw
    
    method : Choose between `rates` and `uniform`. Defines the method used to draw random samples.
    
    Output
    ---------
    redshift_draws : Mock redshift observations
    """
    
    if method == 'rates':
        Dc_squared = D_comoving(z_array, H0, Omega_m)**2
        rate = rate_function(z_array)
        Hz = Hubble_function(z_array, H0, Omega_m)

        pdf = normalise(4*np.pi*Dc_squared*rate/(Hz*(1+z_array)))
        
        redshift_draws = np.random.choice(z_array, p=pdf, replace=True, size=N_draws)
        
    elif method == 'uniform':
        redshift_draws = np.random.choice(z_array, replace=False, size=N_draws)
    
    elif method == 'gaussian':
        pdf = distribution_redshift_Gaussian(z_array)
        redshift_draws = np.random.choice(z_array, p=pdf, replace=True, size=N_draws)
        
    elif method == 'lognormal':
        pdf = distribution_redshift_LogNormal(z_array)
        redshift_draws = np.random.choice(z_array, p=pdf, replace=True, size=N_draws)
        
    elif method == 'powerlaw':
        pdf = distribution_redshift_Power(z_array)
        redshift_draws = np.random.choice(z_array, p=pdf, replace=True, size=N_draws)        
        
    else:
        raise Exception("Wrong method chosen! Choose between 'rates', 'uniform', 'gaussian', 'lognormal' and 'powerlaw'.")
        
    ## Check if we have many events with the same redshift
    if np.unique(redshift_draws).size/N_draws < 0.8:
        raise Exception("Many replications in redshifts drawn. Retry the sampling!")      
    
    return redshift_draws



###############################################


def f_IGM_redshift(z, alpha=0.11, f_IGM_0 = f_IGM):
    return f_IGM_0*(1+alpha*z/(1+z))
    

def dDL_integrand_w(z, Om, w):
    """
    Function of the integrand of the dL formula, 
    eq. (5) in [arXiv:1805.12265].
    
    Input
    ----------
    z : redshift
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    """
    return 1/np.sqrt(Om*(1+z)**3+(1-Om)*(1+z)**(3*(1+w)))


def dDM_integrand_w(z, Om, w, alpha=0.11, f_IGM_0 = f_IGM):
    """
    Function of the integrand of the DM formula, 
    eq. (12) in [arXiv:1805.12265].
    
    Input
    ----------
    z : redshift
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    """
    f_IGM_z = f_IGM_redshift(z, alpha, f_IGM_0)
    
    return f_IGM_z*(1+z)/np.sqrt(Om*(1+z)**3+(1-Om)*(1+z)**(3*(1+w)))


def dispersion_measure(z, H0, Om, w=-1, alpha=0.11, f_IGM_0 = f_IGM):
    """
    Function of the DM formula, 
    eq. (12) in [arXiv:1805.12265].
    
    Input
    ----------
    z : redshift
    
    H0 : Hubble constant [km/s/Mpc]
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    
    Output
    ---------
    DM : Dispersion measure [pc/cm^3]
    """    

    factor = 3*C_LIGHT*(H0*KM_2_MPC)*OMEGA_BARYONS/(8*PI*G_NEWTON*M_PROTON)*(7/8)
    integral = quad(dDM_integrand_w, 0, z, args=(Om, w, alpha, f_IGM_0))[0]
    
    unit_transform = DM_2_PCCM3
    
    DM = unit_transform*factor*integral
    
    return DM


def DM_IGM_O_bh_70(z, O_bh_70, Om=OMEGA_MATTER, w=-1, alpha=0.11, f_IGM_0 = f_IGM):
    """
    Function of the DM formula, 
    eq. (12) in [arXiv:1805.12265].
    
    Update to compare with Macquart+ paper,
    [arXiv:2005.13161], Eq. (2).
    
    Input
    ----------
    z : redshift
    
    H0 : Hubble constant [km/s/Mpc]
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    
    Output
    ---------
    DM : Dispersion measure [pc/cm^3]
    """    
    
    O_bH_0=O_bh_70*70

    factor = 3*C_LIGHT*KM_2_MPC*O_bH_0/(8*PI*G_NEWTON*M_PROTON)*(7/8)
    integral = quad(dDM_integrand_w, 0, z, args=(Om, w, alpha, f_IGM_0))[0]
    
    unit_transform = DM_2_PCCM3
    
    DM = unit_transform*factor*integral
    
    return DM

def luminosity_distance(z, H0, Om, w=-1):
    """
    Function of the dL formula, 
    eq. (5) in [arXiv:1805.12265].
    
    Input
    ----------
    z : redshift
    
    H0 : Hubble constant [km/s/Mpc]
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    
    Output
    ---------
    dL : Luminosity distance [Mpc]
    """      

    factor = (1+z)*(C_LIGHT*1e-3)/H0
    integral = quad(dDL_integrand_w, 0, z, args=(Om, w))[0]
        
    dL = factor*integral
    
    return dL


def dLDM_measure(z, H0, Om, w=-1):
    """
    Function that calculates the dLDM product.
    
    Input
    ----------
    z : redshift
    
    H0 : Hubble constant [km/s/Mpc]
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    
    Output
    ---------
    dLDM : Luminosity distance x Dispersion Measure [Mpc x pc/cm^3]
    """

    dL = luminosity_distance(z, H0, Om, w)
    DM = dispersion_measure(z, H0, Om, w)
    
    dLDM = dL*DM
    
    return dLDM


def sigma_dL(z, H0, Om, w=-1, method='Wei'):
    """
    Function that calculates the error of dL,
    eq. (10) in [arXiv:1805.12265].
    
    Input
    ----------
    z : redshift
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    
    Output
    ---------
    s_dL : distance error in Mpc
    """      
    
    dL = luminosity_distance(z, H0, Om, w)
    
    if method == 'Wei':
        
        first_term = 2*dL/SNR_GW
        second_term = 0.05*z*dL
        
        s_dL = np.sqrt(first_term**2+second_term**2)
        
    elif method == 'constant':
        s_dL = dL/DL_ERROR_PERC
        
    else:
        print("Wrong error method!")
    
    return s_dL
    


def sigma_dLDM(dL, DM, error_dL):
    """
    Function that calculates the error of dLDM product,
    eq. (14) in [arXiv:1805.12265].
    
    Input
    ----------
    dL : Luminosity distance [Mpc]
    
    DM : Dispersion measure [pc/cm^3]
    
    error_dL : Error in luminosity distance [Mpc]
    
    Output
    ---------
    s_dLDM : Error in the product dLDM [Mpc x pc/cm^3]
    """    
       
    s_dL = error_dL
    s_DM = SIGMA_DM
    
    first_term = (DM*s_dL)**2
    second_term = (dL*s_DM)**2
    
    s_dLDM = np.sqrt(first_term+second_term)
    
    return s_dLDM


def sigma_DM_IGM(z, sigma_obs=1.5, sigma_MW=30, sigma_IGM=50, sigma_host=50):
    """
    Function that calculated the total error in the dispersion measure of an FRB,
    eq. (13) in [arXiv:1805.12265].
    """
    
    sigma_total=sigma_obs**2+sigma_MW**2+sigma_IGM**2+(sigma_host/(1+z))**2
    
    return np.sqrt(sigma_total)


###############################################
### Macquart PDFs 


################
### PDF HOST ###
################

def pdf_DM_host(DM, e_mu, sigma_host):
    # e^\mu with 20-200 pc cm^{-3} and \sigma_{host} in 0.2-2.0
    mu=np.log(e_mu)
    
    pdf=1.0/(sigma_host*np.sqrt(2*np.pi)*DM)*np.exp(-(np.log(DM)-mu)**2/(2*(sigma_host**2)))
    
    return pdf


def Norm_pdf_host(e_mu,sigma_host):
    try:
        int, _=quad_vec(lambda x: pdf_DM_host(x, e_mu, sigma_host), 0, 1e20)
        
        return int
    except:
        print('Normalization pdf_DM_host error')
        
        
##################
### PDF COSMIC ###
##################

def f_sigma_DM(F, z, met="Mac"):
    if (met=="log"):
        return F/np.log(1+z)
    else:
        return F/np.sqrt(z)

def pdf_DM_cosmo(Delta, C_0, A, sigma, alpha=3, beta=3):
    pdf=A*(Delta**(-beta))*np.exp(-((Delta**(-alpha)-C_0)**2)/(2*(alpha**2)*(sigma**2)))
    return pdf

def DM_diff_HOf(z, HOf, Om=OMEGA_MATTER, w=-1):
    
    def integrand(z, Om, w):
        return (1+z)/np.sqrt(Om*(1+z)**3+(1-Om)*(1+z)**(3*(1+w)))

    factor = 3*C_LIGHT*HOf/(8*PI*G_NEWTON*M_PROTON)*(7/8)
    
    integral, _ = quad(integrand, 0, z, args=(Om, w))
    
    unit_transform = DM_2_PCCM3*KM_2_MPC
    
    DM = unit_transform*factor*integral
    
    return DM

def C0_sigma(sigma, x_min=0, x_max=np.inf, alpha=3, beta=3,condition='mean',initial_guess=1.0):
    """
    Use fsolve to find C_0 when to_C_0 = 1
    
    Parameters:
    -----------
    F: float - Structure factor parameter 
    z: float - Redshift
    alpha: float - Alpha parameter
    initial_guess: float - C_0 initial guess
    condition: condition choose from "mean", "median" or "mode" value equal to 1. For the median recommend minimum sigma from 0.3.
    
    Returns:
    --------
    float: C_0 or None if solution not found
    """
    
    if (condition=="mean"):
        def objective_function(C_0):
            result1,_= quad(lambda x: x*pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta), x_min, x_max)
            result2,_= quad(lambda x: pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta), x_min, x_max)
      
            return result1-result2

    elif (condition=="median"):
        def objective_function(C_0):
        
            result,_= quad(lambda x: pdf_DM_cosmo(x, C_0, 1.0, sigma, alpha, beta), x_min, 1.0)
            int,_= quad(lambda x: pdf_DM_cosmo(x, C_0, 1.0, sigma, alpha, beta), 0, np.inf)
    
            return result-0.5*int

    elif (condition=="mode"):
        def objective_function(C_0):
        
            criteria=C_0**2+4*alpha*beta*sigma**2
            d3=2/(C_0+np.sqrt(criteria))
    
            return d3-1

    else:
        print('Choose condition from["mean","median","mode"]')
        
    try:
        solution = fsolve(objective_function, [initial_guess], full_output=True)
    
        if solution[2] == 1:  # Check if solution is found
            return solution[0][0]
            
    except Exception as e:
        print(f"find_C0 error ({condition}), sigma={sigma}, error: {e}, initial_guess={initial_guess}, check input or change initial guess")
        return None
    
    
def find_C0(F, z, sigmas, C0s, alpha=3, beta=3, sigma_met="Mac",method="interpolation", x_min=0, x_max=np.inf):
    """
    Use fsolve to find C_0 when to_C_0 = 1
    
    Parameters:
    -----------
    F: float - Structure factor parameter 
    z: float - Redshift
    alpha: float - Alpha parameter
    initial_guess: float - C_0 initial guess
    
    Returns:
    --------
    float: C_0 or None if solution not found
    """
    
    if (method=="interpolation"):
        sigma=f_sigma_DM(F,z,met=sigma_met)
        DM_sigma = interpolate.interp1d(sigmas, C0s, kind='cubic',bounds_error=False, 
    fill_value='extrapolate')
        C0 = DM_sigma(sigma)
        return C0
        
    else:
        print("Do accurate method")
        
        sigma=f_sigma_DM(F,z,met=sigma_met)
        C0 = C0_sigma(sigma, x_min=x_min, x_max=x_max, alpha=alpha, beta=beta)
    
        return C0    
    

def find_A(C_0, F, z, alpha=3, beta=3, x_min=0, x_max=np.inf, sigma_met="num"):
    sigma=f_sigma_DM(F,z,met=sigma_met)
    
    pdf, error = quad(lambda x: pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta),  x_min, x_max)
    
    try:
        return 1/pdf
            
    except Exception as e:
        print(f"find_A error，pdf={pdf}, C_0={C_0}, F={F}, z={z}, error: {e}")
        return None    
    
    # for vaiance-sigma relation calculation
    
######### For vaiance-sigma version #########

'''
The following functions are used for calculate the variance for ∆ and get the sigma (but in function we do a reverse way which from each sigma to calculate variance). This is because we find the \sigma_{diff} in P(∆) is not exactly its variance and they also don't show linear relation.
'''

def var_z(z):
    Om=OMEGA_MATTER
    def dDc(x):
        return 1/np.sqrt(Om*(1+x)**3+(1-Om))
    
    def dDM(x):
        return (1+x)/np.sqrt(Om*(1+x)**3+(1-Om))
    
    int1,_=quad(dDc, 0, z)
    int2,_=quad(dDM, 0, z)
    
    return int1/int2**2

def var_z_fast(z):
    Om=OMEGA_MATTER
    def dDc(x):
        return 1/np.sqrt(Om*(1+x)**3+(1-Om))
    
    def dDM(x):
        return (1+x)/np.sqrt(Om*(1+x)**3+(1-Om))
    
    z_array = np.linspace(0, z, 1000)
    int1 = np.trapz(dDc(z_array), x=z_array)
    int2 = np.trapz(dDM(z_array), x=z_array)
    
    return int1/int2**2

def f_variance_delta(F,z,met='num'):
    '''
    please do sigma-variance interpolate in code to finish variance-sigma convert
    example:
    sigma_var = interpolate.interp1d(Vars, Sigmas, kind=1,bounds_error=False, 
    # fill_value='extrapolate'
    )
    '''
    if (met=='num'):
        return F*var_z(z)
    else:
        return F/z
    
def f_variance_delta_fast(F,z,met='num'):
    '''
    please do sigma-variance interpolate in code to finish variance-sigma convert
    example:
    sigma_var = interpolate.interp1d(Vars, Sigmas, kind=1,bounds_error=False, 
    # fill_value='extrapolate'
    )
    '''
    if (met=='num'):
        return F*var_z_fast(z)
    else:
        return F/z    
    
def find_C0_sigma(sigma, sigmas, C0s, alpha=3, beta=3, method="interpolation", x_min=0, x_max=np.inf):
    """
    Use fsolve to find C_0 when to_C_0 = 1
    
    Parameters:
    -----------
    F: float - Structure factor parameter 
    z: float - Redshift
    alpha: float - Alpha parameter
    initial_guess: float - C_0 initial guess
    
    Returns:
    --------
    float: C_0 or None if solution not found
    """
    
    if (method=="interpolation"):
        DM_sigma = interpolate.interp1d(sigmas, C0s, kind=2,bounds_error=False, 
        # fill_value='extrapolate'
        )
        C0 = DM_sigma(sigma)
        return C0
        
    else:
        print("Do accurate method")
        
        C0 = C0_sigma(sigma, x_min=x_min, x_max=x_max, alpha=alpha, beta=beta)
    
        return C0    
    
def find_A_sigma(C_0, sigma, alpha=3, beta=3, x_min=0, x_max=np.inf):
    
    pdf, error = quad(lambda x: pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta),  x_min, x_max)
    
    try:
        return 1.0/pdf
            
    except Exception as e:
        print(f"find_A error，pdf={pdf}, C_0={C_0}, sigma={sigma}, error: {e}")
        return None 
    
def find_A_sigma_fast(C_0, sigma, alpha=3, beta=3, x_min=0, x_max=np.inf):
    
    x_array = np.logspace(-1, 5, 10_000)
    pdf = np.trapz(pdf_DM_cosmo(x_array, C_0, 1, sigma, alpha, beta),  x=x_array)
    
    try:
        return 1.0/pdf
            
    except Exception as e:
        print(f"find_A error，pdf={pdf}, C_0={C_0}, sigma={sigma}, error: {e}")
        return None     
    
def calculate_var(C0, A, sigma_DM, alpha=3, beta=3, x_min=0, x_max=np.inf, error=1e-20, limit=200):
    
    def first_moment_integrand(delta):
        return delta * pdf_DM_cosmo(delta, C0, A, sigma_DM, alpha, beta)
    
    def second_moment_integrand(delta):
        return delta**2 * pdf_DM_cosmo(delta, C0, A, sigma_DM, alpha, beta)
    
    if (x_max!=np.inf):
        x_max1=int_limit(first_moment_integrand, init=x_max, error=error)
        x_max2=int_limit(second_moment_integrand, init=x_max, error=error)
        x_max=np.max([x_max1,x_max2])
        #print(x_max)
    
    mean, _ = quad(first_moment_integrand, x_min, x_max,limit=limit)

    second_moment, _ = quad(second_moment_integrand, x_min, x_max,limit=limit)
    
    variance = second_moment - mean**2
    
    return variance

######## For the third immediate environment component ########

def pdf_DM_src(DM, b, DM_min, DM_max):
    '''
    Assume a power-law distribution for DM_src:
    DM=C*t^{-b}
    If p\propto t, one have:
    P(DM)=C*DM^{-1/b}
    where C is the normalization parameter. a is the free parameter we do the grid search. Note C should be also related with the integrate limitation.
    b within -3 to 2/5 according to Yang & Zhang 2017
    '''
    def int(DM):
        # integration for the pdf with normalization parameter C=1
        index=1-1/b
        return DM**(index)/index
    
    C=1/(int(DM_max)-int(DM_min))
    
    return C*DM**(-1/b)*((DM>=DM_min)&(DM<=DM_max))
    
