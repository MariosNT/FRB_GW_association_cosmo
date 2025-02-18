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
    s_dL : distance erro in Mpc
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

def f_sigma_DM(F, z):
    return F/np.sqrt(z)


def pdf_DM_cosmo(Delta, C_0, A, sigma, alpha=3, beta=3):
    pdf=A*(Delta**(-beta))*np.exp(-((Delta**(-alpha)-C_0)**2)/(2*(alpha**2)*(sigma**2)))
    return pdf


def C0_sigma(sigma, x_min=0, x_max=np.inf, alpha=3, beta=3):
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
    
    def objective_function(C_0):
        result1,_= quad(lambda x: x*pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta), x_min, x_max)
        result2,_= quad(lambda x: pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta), x_min, x_max)
      
        return result1-result2

    try:
        initial_guess=1.0
        solution = fsolve(objective_function, [initial_guess], full_output=True)
        
        if solution[2] == 1:  # Check if solution is found
            return solution[0][0]
        else:
            print(f"find_C0 warning: sigma={sigma}")
            return None
            
    except Exception as e:
        print(f"find_C0 error, sigma={sigma}, error: {e}")
        return None
    
    
def find_C0(F, z, sigmas, C0s, alpha=3, beta=3, method="interpolation", x_min=0, x_max=np.inf):
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
        sigma=f_sigma_DM(F,z)
        DM_sigma = interpolate.interp1d(sigmas, C0s, kind='cubic')
        C0 = DM_sigma(sigma)
        return C0
        
    else:
        print("Do accurate method")
        
        sigma=f_sigma_DM(F,z)
        C0 = C0_sigma(sigma, x_min=x_min, x_max=x_max, alpha=3, beta=3)
    
        return C0    
    

def find_A(C_0, F, z, alpha=3, beta=3, x_min=0, x_max=np.inf):
    sigma=f_sigma_DM(F,z)
    pdf, error = quad(lambda x: pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta),  x_min, x_max)
    
    try:
        return 1/pdf
            
    except Exception as e:
        print(f"find_A error，pdf={pdf}, C_0={C_0}, F={F}, z={z}, error: {e}")
        return None    