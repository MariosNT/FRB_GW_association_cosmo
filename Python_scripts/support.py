########################
### Useful functions ###
########################

from config import *

def int_limit(fun, init=1e6, error=1e-6, limit='upper', loop_num=10000, step=100, *args, **kwargs):
        
        # find integration limitation for exact function
        
        i=0
        x=init
        while(i<loop_num and fun(x, *args, **kwargs)>error):
            
            # uppper limit times step, lower limit devide step
            x=x*(step**((limit=='upper')*2-1))
            i+=1
            
        if i >= loop_num:
            print(f'Reach to the loop limit while x={x}\n')
            
        return x
        
def normalise(lista, x_array=None):
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
    
    if x_array is not None:
        lista = np.array(lista)
        integral = np.trapz(lista, x_array)
    
        normalised = lista / integral
        normalised = normalised / np.sum(normalised)
    else:
        warnings.warn(
            "Normalizing without x_array: using discrete sum normalization (sum=1). "
            "If this is a continuous PDF on a grid, you must provide z_array "
            "or manually account for dz in subsequent calculations to avoid "
            "grid-spacing-dependent results.",
            UserWarning,
            stacklevel=2
        )
        lista = np.array(lista)
        normalised = lista / np.sum(lista)
    
    return normalised


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

def gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))