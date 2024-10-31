########################
### Useful functions ###
########################

from config import *


def normalise(lista):
    """
    Function that normalises a list of data.
    
    Parameters
    ----------
    lista : array of data
    
    Returns
    -------
    Array with "probability" of each value, 
    i.e. sum(array) = 1.
    """
    
    lista = np.array(lista)
    
    return lista/np.sum(lista)



def Gaussian(x, x0, s0):
    """
    Function that defines a normalised Gaussian with mean x0 and std s0.
    """
    return 1/(s0*np.sqrt(2*np.pi))*np.exp(-0.5*(x-x0)**2/s0**2)

def posterior_contour_2D(posterior_2D_norm, n=1000):
    """
    Function that generates a contour plot of the 2D posterior distribution.
    
    Parameters
    ----------
    posterior_2D_norm : 2D array of normalised posterior distribution
    n : number of points to sample from the 2D posterior distribution
    
    Returns
    -------
    posterior_2D_norm : 2D array of normalised posterior distribution
    """
    
    t_post = np.linspace(0, posterior_2D_norm.max(), n)
    integral_post = ((posterior_2D_norm > t_post[:, None, None]) * posterior_2D_norm).sum(axis=(1, 2))
    f_post = interpolate.interp1d(integral_post, t_post)
    return f_post(np.array([0.99, 0.95, 0.68]))