#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>

class FRBProbabilityCalculator {

private:
    // Constants (you might want to define more from the original Python script)
    const double PI = 3.14159265358979323846;

    // Utility functions to mimic numpy/scipy functionality
    double pdf_DM_host(double DM, double e_mu, double sigma_host) {
        double mu = std::log(e_mu);
        return std::exp(-std::pow(std::log(DM) - mu, 2) / (2 * sigma_host * sigma_host)) 
               / (sigma_host * std::sqrt(2 * PI) * DM);
    }

    // double f_sigma_DM(double F, double z){
    //     return F * std::pow(z, -0.5);
    // }

    // pdf_DM_IGM
    double pdf_DM_cosmo(double DM, double C_0, double A, double F, double z) const {
        // consider alpha==beta==3
        return z* A * std::exp(-std::pow(DM , -3)) * std::exp(-std::pow(std::pow(DM, -3) - C_0 , 2))
            / (2 * 9 * F * F); 
    }

    double findC0(double f, double z, double alpha) const {
    // Simplified C0 finding using numerical method
    auto momentRatio = [&](double c_0) {
        auto integrand = [&](double x) { 
            return x * pdf_DM_cosmo(x, c_0, 1.0, f, z); 
        };
        double x_pdf = simpsonsRule(integrand, 0, std::numeric_limits<double>::infinity(), 3);
        double pdf = simpsonsRule([&](double x) { 
            return pdf_DM_cosmo(x, c_0, 1.0, f, z); 
        }, 0, std::numeric_limits<double>::infinity(), 3);
        return x_pdf / pdf - 1.0;
    };

    // Simplified root-finding (replace with more robust method in practice)
    return brentMethod(momentRatio, 0.1, 10.0);
}

    double DM_IGM_O_bh_70(double z, double O_bh_70) const {
        // Constants (you may need to adjust these to match the original implementation)
        const double C_LIGHT = 299792.458;  // km/s
        const double KM_2_MPC = 3.24078e-6;  // km to Mpc conversion
        const double PI = 3.14159265358979323846;
        const double G_NEWTON = 6.67430e-11;  // m^3 kg^-1 s^-2
        const double M_PROTON = 1.672621898e-27;  // kg
        const double DM_2_PCCM3 = 1.0;  // conversion factor
        const double f_IGM = 1.0;  // default value, adjust as needed
        const double OMEGA_MATTER = 0.3;
        const double w = -1.0;

        // Compute O_bH_0 
        double O_bH_0 = O_bh_70 * 70;

        // Compute integral (simplified - you may need to replace with a more accurate integration method)
        double integral = 0.0;
        for (double i = 0; i < z; i += 0.1) {
            // Simple trapezoidal approximation
            integral += 0.1 * (1 + i);
        }

        // Compute factor
        double factor = 3 * C_LIGHT * KM_2_MPC * O_bH_0 * f_IGM / 
                   (8 * PI * G_NEWTON * M_PROTON) * (7.0 / 8.0);

        // Unit transformation
        double unit_transform = DM_2_PCCM3;

        // Compute DM
        double DM = unit_transform * factor * integral;

    return DM;
}

    double calculate_dm_probability(
        double DM_frb, double z, double F, double O_bh_70, 
        double sigma_host, double e_mu
    ) {
        auto integrand = [&](double DM_host) {
            double p_host_val = pdf_DM_host(DM_host, e_mu, sigma_host);
            // Implement the rest of the probability calculation logic
            // This includes finding C_0 and A, calculating delta, etc.
            return p_host_val; // Placeholder
        };

        // Implement numerical integration
        // This would use a method like Simpson's rule or adaptive quadrature
        return 0.0; // Placeholder
    }

public:

    double brentMethod(const std::function<double(double)>& f, double a, double b, 
                   double tolerance = 1e-6, int max_iter = 100) {
    double fa = f(a);
    double fb = f(b);

    if (std::abs(fa) < std::abs(fb)) {
        std::swap(a, b);
        std::swap(fa, fb);
    }

    double c = a;
    double fc = fa;
    bool mflag = true;
    double d = b - a;
    double e = d;

    for (int iter = 0; iter < max_iter; ++iter) {
        if (std::abs(fc) < std::abs(fb)) {
            std::swap(a, c);
            std::swap(fa, fc);
            std::swap(b, c);
            std::swap(fb, fc);
        }

        double s;
        if (std::abs(fa - fc) > tolerance && std::abs(fb - fc) > tolerance) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc)) +
                b * fa * fc / ((fb - fa) * (fb - fc)) +
                c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        bool condition1 = !(s >= (3 * a + b) / 4 && s <= b);
        bool condition2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2;
        bool condition3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2;
        bool condition4 = mflag && std::abs(b - c) < tolerance;
        bool condition5 = !mflag && std::abs(c - d) < tolerance;

        if (condition1 || condition2 || condition3 || condition4 || condition5) {
            s = (a + b) / 2;
            mflag = true;
        } else {
            mflag = false;
        }

        double fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (std::abs(fa) < std::abs(fb)) {
            std::swap(a, b);
            std::swap(fa, fb);
        }

        if (std::abs(b - a) < tolerance) {
            return b;
        }
    }

    return b;
}


    double simpsonsRule(const std::function<double(double)>& func, double a, double b, int n) const {
    // Handle improper integrals with large upper limit
    if (std::isinf(b)) {
        // Adaptive technique for infinite upper limit
        double result = 0.0;
        double x = a;
        double step = 1.0;
        
        while (x < 100) {  // Truncate at a large finite value
            double x_next = x + step;
            result += (step / 6.0) * (
                func(x) + 
                4.0 * func((x + x_next) / 2.0) + 
                func(x_next)
            );
            x = x_next;
            step *= 1.5;  // Exponential step increase
        }
        
        return result;
    }
    
    // Standard Simpson's rule for finite limits
    if (n % 2 != 0) n += 1;  // Ensure even number of subintervals
    
    double h = (b - a) / n;
    double result = func(a) + func(b);
    
    for (int i = 1; i < n; i += 2) {
        double x = a + i * h;
        result += 4.0 * func(x);
    }
    
    for (int i = 2; i < n - 1; i += 2) {
        double x = a + i * h;
        result += 2.0 * func(x);
    }
    
    return result * h / 3.0;
}

    std::vector<std::vector<std::vector<std::vector<double>>>> 
    calculate_posterior_4D(
        const std::vector<double>& F_array,
        const std::vector<double>& O_bh_70_array,
        const std::vector<double>& sigma_host_array,
        const std::vector<double>& e_mu_array,
        const std::vector<std::pair<double, double>>& data
    ) {
        // Initialize 4D posterior array
        std::vector<std::vector<std::vector<std::vector<double>>>> posterior_4D(
            F_array.size(),
            std::vector<std::vector<std::vector<double>>>(
                O_bh_70_array.size(),
                std::vector<std::vector<double>>(
                    sigma_host_array.size(),
                    std::vector<double>(e_mu_array.size(), 0.0)
                )
            )
        );

        // Nested loops to iterate through parameter space
        for (size_t i = 0; i < F_array.size(); ++i) {
            for (size_t j = 0; j < O_bh_70_array.size(); ++j) {
                for (size_t k = 0; k < sigma_host_array.size(); ++k) {
                    for (size_t l = 0; l < e_mu_array.size(); ++l) {
                        double prob_total = 1.0;

                        // Calculate probability for each FRB in the dataset
                        for (const auto& row : data) {
                            double DM_frb = row.first;  // DM value
                            double z = row.second;      // redshift

                            double prob = calculate_dm_probability(
                                DM_frb, z,
                                F_array[i], O_bh_70_array[j],
                                sigma_host_array[k], e_mu_array[l]
                            );

                            prob_total *= prob;
                        }

                        posterior_4D[i][j][k][l] = prob_total;
                    }
                }
            }
        }

        return posterior_4D;
    }

    void save_posterior_to_csv(
        const std::vector<std::vector<std::vector<std::vector<double>>>>& posterior_4D,
        const std::vector<double>& F_array,
        const std::vector<double>& O_bh_70_array,
        const std::vector<double>& sigma_host_array,
        const std::vector<double>& e_mu_array,
        const std::string& filename
    ) {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Unable to open file: " << filename << std::endl;
            return;
        }

        // Write header
        outfile << "F,O_bh_70,sigma_host,e_mu,posterior\n";

        for (size_t i = 0; i < F_array.size(); ++i) {
            for (size_t j = 0; j < O_bh_70_array.size(); ++j) {
                for (size_t k = 0; k < sigma_host_array.size(); ++k) {
                    for (size_t l = 0; l < e_mu_array.size(); ++l) {
                        outfile << F_array[i] << ","
                                << O_bh_70_array[j] << ","
                                << sigma_host_array[k] << ","
                                << e_mu_array[l] << ","
                                << posterior_4D[i][j][k][l] << "\n";
                    }
                }
            }
        }

        outfile.close();
    }
};

int main() {
    FRBProbabilityCalculator calculator;

    // Define parameter ranges (same as in Python script)
    std::vector<double> F_array = {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<double> O_bh_70_array = {0.015, 0.035, 0.055, 0.075, 0.095};
    std::vector<double> sigma_host_array = {0.2, 0.6, 1.0, 1.4, 1.8};
    std::vector<double> e_mu_array = {20, 60, 100, 140, 180};

    // Prepare data (you'll need to load actual data similar to the Python script)
    std::vector<std::pair<double, double>> data = {
        {50.5, 0.3}, // example entry: {DM_frb, z}
        {75.2, 0.5}  // Add more real data entries
    };

    // Calculate posterior
    auto posterior_4D = calculator.calculate_posterior_4D(
        F_array, O_bh_70_array, sigma_host_array, e_mu_array, data
    );

    // Save to CSV
    calculator.save_posterior_to_csv(
        posterior_4D, F_array, O_bh_70_array, 
        sigma_host_array, e_mu_array, "posterior_results.csv"
    );

    return 0;
}