import sys
import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from scipy.optimize import fsolve
from scipy import stats
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import Tuple, List, Optional

class FRBAnalysis:
    def __init__(self, dm_halo: float = 30):
        """Initialize FRB analysis with constants."""
        self.dm_halo = dm_halo
        # Define parameter ranges
        self.f_array = np.linspace(0.01, 0.5, 20)
        self.o_bh_70_array = np.linspace(0.015, 0.095, 10)
        self.sigma_host_array = np.linspace(0.2, 2.0, 10)
        self.e_mu_array = np.linspace(20, 200, 100)
        
    def load_data(self, frb_surajit_path: str, frb_kritti_path: str) -> pd.DataFrame:
        """Load and preprocess FRB data from both sources."""
        # Load Surajit's data
        df1 = pd.read_csv(frb_surajit_path)
        df1['DM_ext'] = df1['DM_obs (pc cm^-3)'] - df1['DM_MW (pc cm^-3)']
        
        # Load Kritti's data
        df2 = pd.read_excel(frb_kritti_path)
        df2['FRB'] = 'FRB ' + df2['FRB']
        
        # Combine datasets
        data = pd.concat([
            df1[['FRB', 'DM_ext', 'z']],
            df2[['FRB', 'DM_ext', 'z']]
        ], ignore_index=True)
        
        # Remove specific FRB and negative DM_ext values
        data = data[data['FRB'] != 'FRB 20190520B']
        data = data[data['DM_ext'] >= 0]
        
        # Calculate DM_
        data['DM_'] = data['DM_ext'] - self.dm_halo
        data = data[data['DM_'] >= 0]
        
        return data

    @staticmethod
    def pdf_dm_host(dm: float, e_mu: float, sigma_host: float) -> float:
        """Calculate PDF for host DM contribution."""
        mu = np.log(e_mu)
        return np.exp(-(np.log(dm) - mu)**2 / (2 * sigma_host**2)) / (sigma_host * np.sqrt(2 * np.pi) * dm)

    @staticmethod
    def pdf_dm_cosmo(dm: float, c_0: float, a: float, f: float, z: float, alpha: float = 3) -> float:
        """Calculate PDF for cosmological DM contribution."""
        beta = (alpha + 1) / (alpha - 1)
        sigma_dm = f * z**-0.5
        return a * dm**(-beta) * np.exp(-(dm**(-alpha) - c_0)**2 / (2 * alpha**2 * sigma_dm**2))

    def find_c0(self, f: float, z: float, alpha: float = 3) -> Optional[float]:
        """Find C_0 parameter using numerical solver."""
        def x_moment_ratio(c_0: float) -> float:
            """Calculate ratio of first moment to zeroth moment."""
            x_pdf, _ = quad_vec(lambda x: x * self.pdf_dm_cosmo(x, c_0, 1, f, z, alpha), 0, np.inf)
            pdf, _ = quad_vec(lambda x: self.pdf_dm_cosmo(x, c_0, 1, f, z, alpha), 0, np.inf)
            return x_pdf/pdf - 1
        
        try:
            solution = fsolve(x_moment_ratio, [1.0], full_output=True)
            return solution[0][0] if solution[2] == 1 else None
        except Exception as e:
            print(f"C_0 calculation failed for F={f}, z={z}: {e}")
            return None

    def find_a(self, c_0: float, f: float, z: float, alpha: float = 3) -> Optional[float]:
        """Find normalization factor A."""
        try:
            pdf, _ = quad_vec(lambda x: self.pdf_dm_cosmo(x, c_0, 1, f, z, alpha), 0, np.inf)
            return 1/pdf
        except Exception as e:
            print(f"A calculation failed for C_0={c_0}, F={f}, z={z}: {e}")
            return None

    def calculate_single_probability(self, params: Tuple[int, int, int, int, pd.DataFrame]) -> Tuple[int, int, int, int, float]:
        """Calculate probability for a single parameter combination."""
        i, j, k, l, data = params
        f = self.f_array[i]
        o_bh70 = self.o_bh_70_array[j]
        sigma_host = self.sigma_host_array[k]
        e_mu = self.e_mu_array[l]
        
        prob_total = 1.0
        for _, row in data.iterrows():
            dm_frb = row['DM_']
            z = row['z']
            
            def integrand(dm_host):
                p_host = self.pdf_dm_host(dm_host, e_mu, sigma_host)
                c_0 = self.find_c0(f, z)
                if c_0 is None:
                    return 0
                a = self.find_a(c_0, f, z)
                if a is None:
                    return 0
                delta = (dm_frb - dm_host/(1+z)) / self.dm_igm_o_bh_70(z, o_bh70)
                p_cosmic = self.pdf_dm_cosmo(delta, c_0, a, f, z)
                return p_host * p_cosmic
            
            prob, _ = quad_vec(integrand, 0, dm_frb)
            prob_total *= prob
            
        return i, j, k, l, prob_total

    def calculate_posterior(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate posterior probabilities using multiprocessing."""
        posterior_4d = np.zeros((len(self.f_array), len(self.o_bh_70_array), 
                               len(self.sigma_host_array), len(self.e_mu_array)))
        
        # Prepare parameter combinations
        params = [(i, j, k, l, data) 
                 for i in range(len(self.f_array))
                 for j in range(len(self.o_bh_70_array))
                 for k in range(len(self.sigma_host_array))
                 for l in range(len(self.e_mu_array))]
        
        # Use multiprocessing
        with Pool(processes=cpu_count()) as pool:
            results = pool.map(self.calculate_single_probability, params)
        
        # Fill posterior array with results
        for i, j, k, l, prob in results:
            posterior_4d[i, j, k, l] = prob
            
        return posterior_4d

    def plot_confidence_regions(self, posterior_4d: np.ndarray):
        """Plot confidence regions for all parameter pairs."""
        d_4d = self.calculate_d_statistic(posterior_4d)
        thresholds = self.get_sigma_thresholds()
        
        # Create parameter grids
        grids = self.create_parameter_grids()
        
        # Plot each parameter combination
        for (param1_name, param2_name), (grid1, grid2) in grids.items():
            self.plot_single_contour(d_4d, grid1, grid2, param1_name, param2_name, thresholds)
            plt.savefig(f"{param1_name}_{param2_name}.png")
            plt.close()

    @staticmethod
    def calculate_d_statistic(posterior_4d: np.ndarray) -> np.ndarray:
        """Calculate likelihood ratio test statistic D."""
        l_max = np.max(posterior_4d)
        return -2 * (np.log(posterior_4d) - np.log(l_max))

    @staticmethod
    def get_sigma_thresholds() -> List[float]:
        """Get threshold values for confidence levels."""
        confidence_levels = [0.6827, 0.9545, 0.9973]
        return [stats.chi2.ppf(cl, df=4) for cl in confidence_levels]

def main():
    # Initialize analysis
    analysis = FRBAnalysis()
    
    # Load and preprocess data
    data = analysis.load_data('FRB_Surajit.csv', 'FRB_Kritti.xlsx')
    
    # Calculate posterior probabilities
    posterior_4d = analysis.calculate_posterior(data)
    
    # Plot results
    analysis.plot_confidence_regions(posterior_4d)

if __name__ == '__main__':
    main()