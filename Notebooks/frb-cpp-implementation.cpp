#include "frb_analysis.h"

FRBAnalysis::FRBAnalysis(double dm_halo) {
    // Initialize parameter ranges
    f_array = linspace(0.01, 0.5, 20);
    o_bh_70_array = linspace(0.015, 0.095, 10);
    sigma_host_array = linspace(0.2, 2.0, 10);
    e_mu_array = linspace(20, 200, 100);
}

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


double FRBAnalysis::simpsonsRule(const std::function<double(double)>& func, double a, double b, int n) const {
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


double FRBAnalysis::pdfDMHost(double dm, double e_mu, double sigma_host) const {
    double mu = std::log(e_mu);
    return std::exp(-std::pow(std::log(dm) - mu, 2) / (2 * std::pow(sigma_host, 2))) 
           / (sigma_host * std::sqrt(2 * PI) * dm);
}

double FRBAnalysis::pdfDMCosmo(double dm, double c_0, double a, double f, double z, double alpha) const {
    double beta = (alpha + 1) / (alpha - 1);
    double sigma_dm = f * std::pow(z, -0.5);
    return a * std::pow(dm, -beta) * std::exp(-std::pow(std::pow(dm, -alpha) - c_0, 2) 
           / (2 * std::pow(alpha, 2) * std::pow(sigma_dm, 2)));
}

double FRBAnalysis::findC0(double f, double z, double alpha) const {
    // Simplified C0 finding using numerical method
    auto momentRatio = [&](double c_0) {
        auto integrand = [&](double x) { 
            return x * pdfDMCosmo(x, c_0, 1.0, f, z, alpha); 
        };
        double x_pdf = simpsonsRule(integrand, 0, std::numeric_limits<double>::infinity());
        double pdf = simpsonsRule([&](double x) { 
            return pdfDMCosmo(x, c_0, 1.0, f, z, alpha); 
        }, 0, std::numeric_limits<double>::infinity());
        return x_pdf / pdf - 1.0;
    };

    // Simplified root-finding (replace with more robust method in practice)
    return brentMethod(momentRatio, 0.1, 10.0);
}

double FRBAnalysis::dmIGMOBH70(double z, double O_bh_70) const {
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

std::vector<double> FRBAnalysis::calculatePosteriorProbabilities(const FRBData& data) {
    std::vector<double> posterior;

    for (const auto& f : f_array) {
        for (const auto& o_bh70 : o_bh_70_array) {
            for (const auto& sigma_host : sigma_host_array) {
                for (const auto& e_mu : e_mu_array) {
                    double prob_total = 1.0;
                    
                    for (size_t i = 0; i < data.dm_ext.size(); ++i) {
                        double dm_frb = data.dm_ext[i];
                        double z = data.redshift[i];

                        auto integrand = [&](double dm_host) {
                            double p_host = pdfDMHost(dm_host, e_mu, sigma_host);
                            double c_0 = findC0(f, z);
                            double a = findA(c_0, f, z);
                            double delta = (dm_frb - dm_host / (1.0 + z)) / dmIGMOBH70(z, o_bh70);
                            double p_cosmic = pdfDMCosmo(delta, c_0, a, f, z);
                            return p_host * p_cosmic;
                        };

                        double prob = simpsonsRule(integrand, 0, dm_frb);
                        prob_total *= prob;
                    }
                    
                    posterior.push_back(prob_total);
                }
            }
        }
    }

    return posterior;
}
