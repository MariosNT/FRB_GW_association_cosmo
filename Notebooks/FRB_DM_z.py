import sys
sys.path.append('../Python_scripts')
from config import *
from support import *
from cosmo_support import *
from scipy.integrate import quad_vec
from scipy.optimize import fsolve
from scipy import stats

df1 = pd.read_csv('FRB_Surajit.csv')
df1['DM_ext']=df1['DM_obs (pc cm^-3)']-df1['DM_MW (pc cm^-3)']
df2 = pd.read_excel('FRB_Kritti.xlsx')
df2['FRB']='FRB '+df2['FRB']

data=df1[['FRB','DM_ext','z']]
data=pd.concat([data,df2[['FRB','DM_ext','z']]],ignore_index=True)
data = data[data['FRB'] != 'FRB 20190520B']

# Assuming your dataframe is named df
# 1. Find rows where DM_ext < 0
dropped_rows = data[data['DM_ext'] < 0]

# 2. Print FRB values of dropped rows if any exist
if not dropped_rows.empty:
    print("Dropped rows:")
    print(dropped_rows[['FRB', 'DM_ext']])
else:
    print("No rows with DM_ext < 0 found to drop")

# 3. Keep only rows where DM_ext >= 0
data = data[data['DM_ext'] >= 0]

print(f'Now there are {len(data)} rows in the dataframe')

def pdf_DM_host(DM, e_mu, sigma_host):
    # e^\mu with 20-200 pc cm^{-3} and \sigma_{host} in 0.2-2.0
    mu=np.log(e_mu)
    pdf=np.exp(-(np.log(DM)-mu)**2/(2*sigma_host**2))/(sigma_host*np.sqrt(2*np.pi)*DM)
    
    return pdf

def f_sigma_DM(F, z):
    return F*z**-0.5

def pdf_DM_cosmo(DM, C_0, A, F, z, alpha=3):
    beta=(alpha+1)/(alpha-1)
    sigma_DM=f_sigma_DM(F, z)
    pdf=A*DM**(-beta)*np.exp(-(DM**(-alpha)-C_0)**2/(2*alpha**2*sigma_DM**2))
    return pdf

def x_pdf_DM_cosmo(DM, C_0, A, F, z, alpha=3):
    beta=(alpha+1)/(alpha-1)
    sigma_DM=f_sigma_DM(F, z)
    pdf=A*DM**(1-beta)*np.exp(-(DM**(-alpha)-C_0)**2/(2*alpha**2*sigma_DM**2))
    
    return pdf

def to_C_0(C_0, F, z, alpha=3):
    #result, error = quad(integrand, 0, np.inf, args=(A, beta, alpha, C0, sigma_DM))
    A=1
    x_pdf, error = quad_vec(lambda x: x_pdf_DM_cosmo(x, C_0, A, F, z, alpha), 0, np.inf)
    pdf, error = quad_vec(lambda x: pdf_DM_cosmo(x, C_0, A, F, z, alpha), 0, np.inf)
    #print(f'x_pdf={x_pdf},pdf={pdf}')
    try:
        ratio=x_pdf/pdf
    except ZeroDivisionError:
        print(f'zero division error for x_pdf={x_pdf},pdf={pdf}')
        ratio=50
    except Exception as e:
        print(f'unknown error for x_pdf={x_pdf},pdf={pdf},error={e}')
    return ratio

vec_C_0=np.vectorize(to_C_0)

def find_C0(F, z, alpha=3, initial_guess=1.0):
    """
    use fsolve find C_0 when to_C_0 = 1
    
    parameters:
    F: float - F 
    z: float - z 
    alpha: float - alpha
    initial_guess: float - C_0 initial guess 1.0
    
    return:
    float: C_0
    """
    def objective_function(C_0):
        # target function：to_C_0 - 1 = 0
        return to_C_0(C_0[0], F, z, alpha) - 1
    
    try:
        # use fsolve solve，initial_guess vector
        solution = fsolve(objective_function, [initial_guess], full_output=True)
        
        if solution[2] == 1:  # Check if the solution is found
            return solution[0][0]  # return first element of the solution vector
        else:
            print(f"find_C0 warning：F={F}, z={z}")
            return None
            
    except Exception as e:
        print(f"find_C0 error，F={F}, z={z}, error: {e}")
        return None
    
def find_A(C_0, F, z, alpha=3):
    pdf, error = quad_vec(lambda x: pdf_DM_cosmo(x, C_0, 1, F, z, alpha), 0, np.inf)
    
    try:
        return 1/pdf
            
    except Exception as e:
        print(f"find_A error，pdf={pdf}, C_0={C_0}, F={F}, z={z}, error: {e}")
        return None
    
DM_halo=30  # In paper is 50, but Bing suggest smaller values
# DM_ISM=30 # we already have DM_MW in data

data['DM_']=data['DM_ext']-DM_halo

dropped_rows = data[data['DM_'] < 0]

# 2. Print FRB values of dropped rows if any exist
if not dropped_rows.empty:
    print("Dropped rows:")
    print(dropped_rows[['FRB', 'DM_']])
else:
    print("No rows with DM_ < 0 found to drop")

# 3. Keep only rows where DM_ext >= 0
data = data[data['DM_'] >= 0]

def DM_IGM_O_bh_70(z, O_bh_70, Om=OMEGA_MATTER ,w=-1):
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
    
    O_bH_0=O_bh_70*70

    factor = 3*C_LIGHT*KM_2_MPC*O_bH_0*f_IGM/(8*PI*G_NEWTON*M_PROTON)*(7/8)
    integral = quad(dDM_integrand_w, 0, z, args=(Om, w))[0]
    
    unit_transform = DM_2_PCCM3
    
    DM = unit_transform*factor*integral
    
    return DM

def calculate_dm_probability(DM_frb,z,F,O_bh_70,sigma_host,e_mu):
    """
    Calculate the probability of DM_FRB given various cosmological parameters
    
    Parameters:
    -----------
    DM_frb : float
        The observed DM_FRB value
    z : float
        Redshift
    F : float
        Structure factor parameter
    O_bh_70 : float
        Baryon density parameter
    sigma_host : float
        Standard deviation of host DM distribution
    e_mu : float
        Mean of host DM distribution
    
    Returns:
    --------
    float
        The calculated probability
    """
    def integrand(DM_host):
        # Calculate p_host at the current dm_host value
        p_host_val = pdf_DM_host(DM_host, e_mu, sigma_host)
        
        # Calculate p_cosmic at (DM_FRB - DM_host)
        C_0=find_C0(F=F,z=z)
        A=find_A(C_0=C_0,F=F,z=z)
        # Calculate delta parameter
        delta=DM_frb - DM_host/(1+z)
        delta=delta/DM_IGM_O_bh_70(z=z,O_bh_70=O_bh_70)
        p_cosmic_val = pdf_DM_cosmo(DM=delta, C_0=C_0, A=A, F=F,z=z)
        
        return p_host_val * p_cosmic_val
    
    # Perform the integration from 0 to DM_FRB
    result, error = quad_vec(integrand, 0, DM_frb)
    
    return result

F_array=np.linspace(0.01,0.5,20)
O_bh_70_array=np.linspace(0.015,0.095,10)
sigma_host_array=np.linspace(0.2,2.0,10)
e_mu_array=np.linspace(20,200,100)

F_4D, O_bh_70_4D, sigma_host_4D, e_mu_4D = np.meshgrid(F_array, O_bh_70_array, sigma_host_array, e_mu_array)

F_h, h_F = np.meshgrid(F_array, O_bh_70_array)
F_sigma, sigma_F = np.meshgrid(F_array, sigma_host_array)
F_e_mu, e_mu_F = np.meshgrid(F_array, e_mu_array)

h_sigma, sigma_h = np.meshgrid(O_bh_70_array, sigma_host_array)
h_e_mu, e_mu_h = np.meshgrid(O_bh_70_array, e_mu_array)

sigma_e_mu, e_mu_sigma = np.meshgrid(sigma_host_array, e_mu_array)

# Initialize posterior_4D array for storing results
posterior_4D = np.zeros_like(F_4D)

# Iterate through all parameter combinations
for i, F in enumerate(F_array):
    for j, O_bh70 in enumerate(O_bh_70_array):
        for k, sigma_host in enumerate(sigma_host_array):
            for l, e_mu in enumerate(e_mu_array):
                # Initialize the total probability for current parameter combination
                prob_total = 1.0  # or 0.0 if you want to sum instead of multiply
                
                # Calculate and accumulate probabilities for each FRB
                for _, row in data.iterrows():
                    prob = calculate_dm_probability(
                        DM_frb=row['DM_'],     # Use DM_ column as DM_frb
                        z=row['z'],            # Use z column as redshift
                        F=F,                   # Structure factor from parameter space
                        O_bh_70=O_bh70,        # Baryon density from parameter space
                        sigma_host=sigma_host, # Host DM standard deviation from parameter space
                        e_mu=e_mu             # Host DM mean from parameter space
                    )
                    # Accumulate probability (multiply or add)
                    prob_total *= prob  # or prob_total += prob
                
                # Store the accumulated result in posterior_4D
                posterior_4D[i,j,k,l] = prob_total
                
            print(f"i,j,k,l = {i},{j},{k},{l}",end='\r')

def calculate_D(posterior_4D):
    """
    Calculate likelihood ratio test statistic D
    
    Parameters:
    -----------
    posterior_4D : numpy.ndarray
        4D array containing likelihood values (L in the equation)
    
    Returns:
    --------
    D_4D : numpy.ndarray
        4D array containing D statistic values calculated as -2ln(L/L_max)
    """
    L_max = np.max(posterior_4D)
    D_4D = -2 * (np.log(posterior_4D) - np.log(L_max))
    return D_4D

def get_sigma_thresholds():
    """
    Calculate threshold values for 1σ, 2σ, and 3σ confidence levels using chi-square distribution
    
    Returns:
    --------
    list
        Three threshold values corresponding to:
        1σ (68.27%), 2σ (95.45%), 3σ (99.73%) confidence levels
    """
    confidence_levels = [0.6827, 0.9545, 0.9973]
    thresholds = [stats.chi2.ppf(cl, df=4) for cl in confidence_levels]
    return thresholds

def plot_contours(D_4D, param1_grid, param2_grid, param1_name, param2_name, 
                 param_indices):
    """
    Plot 2D contours showing 1σ, 2σ, and 3σ confidence regions for given parameters
    
    Parameters:
    -----------
    D_4D : numpy.ndarray
        4D array of D statistic values
    param1_grid, param2_grid : numpy.ndarray
        2D meshgrid arrays for the two parameters to plot
    param1_name, param2_name : str
        Names of parameters for axis labels
    param_indices : tuple
        Indices of the two parameters in the original 4D array (0-based)
        These indices determine which dimensions to marginalize over
    """
    # Determine which dimensions need to be marginalized
    all_dims = set(range(4))
    plot_dims = set(param_indices)
    marginalize_dims = tuple(all_dims - plot_dims)
    
    # Marginalize over other dimensions by taking the minimum D value
    # This effectively finds the best-fit values for the marginalized parameters
    D_2D = np.min(D_4D, axis=marginalize_dims)
    
    # Transpose D_2D if necessary to match the parameter grid orientations
    # This ensures the axes of D_2D match the orientations of param1_grid and param2_grid
    if param_indices[0] > param_indices[1]:
        D_2D = D_2D.T
    
    # Get thresholds for different sigma levels
    thresholds = get_sigma_thresholds()
    
    # Create the figure
    plt.figure(figsize=(8, 6))
    
    # Plot filled contours for each confidence level
    colors = ['lightblue', 'lightgreen', 'pink']
    labels = ['1σ (68.27%)', '2σ (95.45%)', '3σ (99.73%)']
    
    for i, (threshold, color, label) in enumerate(zip(thresholds, colors, labels)):
        # Draw contour lines with different line styles
        plt.contour(param1_grid, param2_grid, D_2D, levels=[threshold], 
                   colors='k', linestyles=['-', '--', ':'][i])
        # Fill regions within contours
        plt.contourf(param1_grid, param2_grid, D_2D, levels=[0, threshold],
                    colors=[color], alpha=0.3, label=label)
    
    plt.xlabel(param1_name)
    plt.ylabel(param2_name)
    plt.title('Confidence Regions (1σ, 2σ, 3σ)')
    plt.legend()

# Define parameter order in the 4D array
# This maps parameter names to their corresponding indices in D_4D
PARAM_ORDER = {
    'F': 0,
    'Ω_bh_70': 1,
    'σ_host': 2,
    'μ': 3
}

# Main execution code
D_4D = calculate_D(posterior_4D)

# Define parameter pairs to plot with their corresponding indices
parameter_pairs = [
    # (param1_grid, param2_grid, param1_name, param2_name, (param1_index, param2_index))
    (F_h, h_F, 'F', 'Ω_bh_70', (PARAM_ORDER['F'], PARAM_ORDER['Ω_bh_70'])),
    (F_sigma, sigma_F, 'F', 'σ_host', (PARAM_ORDER['F'], PARAM_ORDER['σ_host'])),
    (F_e_mu, e_mu_F, 'F', 'μ', (PARAM_ORDER['F'], PARAM_ORDER['μ'])),
    (h_sigma, sigma_h, 'Ω_bh_70', 'σ_host', (PARAM_ORDER['Ω_bh_70'], PARAM_ORDER['σ_host'])),
    (h_e_mu, e_mu_h, 'Ω_bh_70', 'μ', (PARAM_ORDER['Ω_bh_70'], PARAM_ORDER['μ'])),
    (sigma_e_mu, e_mu_sigma, 'σ_host', 'μ', (PARAM_ORDER['σ_host'], PARAM_ORDER['μ']))
]

# Plot all parameter combinations
for param1_grid, param2_grid, param1_name, param2_name, param_indices in parameter_pairs:
    plot_contours(D_4D, param1_grid, param2_grid, param1_name, param2_name, param_indices)
    plt.savefig(f"{param1_name}_{param2_name}.png")