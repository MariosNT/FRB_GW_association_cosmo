###############################################
### Functions for the cosmological analysis ###
###############################################

from config import *



###############################################

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


def dDM_integrand_w(z, Om, w):
    """
    Function of the integrand of the DM formula, 
    eq. (12) in [arXiv:1805.12265].
    
    Input
    ----------
    z : redshift
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    """
    return (1+z)/np.sqrt(Om*(1+z)**3+(1-Om)*(1+z)**(3*(1+w)))


def dispersion_measure(z, H0, Om, w=-1):
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

    factor = 3*C_LIGHT*(H0*KM_2_MPC)*OMEGA_BARYONS*f_IGM/(8*PI*G_NEWTON*M_PROTON)*(7/8)
    integral = quad(dDM_integrand_w, 0, z, args=(Om, w))[0]
    
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







######################################

# def dDM_integrand(z, Om):
#     """
#     Function that it inside the integral of the DM formula.
    
#     Input
#     ----------
#     z : redshift
#     Om: Omega matter
#     """
#     return (1+z)/np.sqrt(Om*(1+z)**3+1-Om)

# def dispersion_measure(z, Om, H0):

#     factor = 3*C_LIGHT*(H0*KM_2_MPC)*OMEGA_BARYONS*f_IGM/(8*PI*G_NEWTON*M_PROTON)*(7/8)
#     integral = quad(dDM_integrand, 0, z, args=(Om))[0]
    
#     unit_transform = 3.24e-23
    
#     DM = unit_transform*factor*integral
    
#     return DM