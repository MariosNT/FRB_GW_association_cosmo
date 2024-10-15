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