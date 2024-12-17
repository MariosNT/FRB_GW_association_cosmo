#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <memory>
#include <limits>
#include <functional>

class FRBProbabilityCalculator {

//private:

public:
    // Constants (you might want to define more from the original Python script)
    const double PI = 3.14159265358979323846;

    // Utility functions to mimic numpy/scipy functionality
    double PDF_DM_host(double DM, double e_mu, double sigma_host) {
        double mu = std::log(e_mu);

        if (DM == 0){
            DM=1e-6;
        }
        long double result = std::exp(-std::pow(std::log(DM) - mu, 2) / (2.0 * sigma_host * sigma_host)) 
               / (sigma_host * std::sqrt(2.0 * PI) * DM);

        // std::cout << "Private Variable: "<< DM;
        // std::cout << std::endl;

        if (std::isnan(result)) {
            throw std::runtime_error("PDF_DM_host in NaN");
        }
        
        return result;
    }

    // double f_sigma_DM(double F, double z){
    //     return F * std::pow(z, -0.5);
    // }

    // pdf_DM_IGM
    double pdf_DM_cosmo(double DM, double C_0, double A, double F, double z) const {
        // consider alpha==beta==3

        if (DM == 0){
            DM=1e-100;
        }

        if (std::isinf(C_0) || std::isnan(C_0) || std::isinf(A) || std::isnan(A)) {
        std::cerr << "Invalid C_0 or A values" << std::endl;
        return 0.0;
    }

        double result = A * std::pow(DM , -3) * std::exp(-std::pow(std::pow(DM, -3) - C_0 , 2) * z
            / (2 * 9 * F * F));

        
        if (std::isnan(result) || std::isnan(result)) {
            std::cout << "Private Variable A: "<< A;
            std::cout << "Private Variable exp: "<< std::exp(-std::pow(std::pow(DM, -3) - C_0 , 2)/ (2 * 9 * F * F));
            std::cout << "Private Variable in exp: "<< -std::pow(std::pow(DM, -3) - C_0 , 2)/ (2 * 9 * F * F);
            std::cout << std::endl;
            throw std::runtime_error("pdf_DM_cosmo in NaN");
        }

        return result; 

    }

    double find_pdf_DM_cosmo(double F, double z, double c_0=1.0, double tolerance = 1e-10) const{
        int i=1;
        while(pdf_DM_cosmo(std::pow(10,i), 1.0 ,1.0 ,F,z)>=tolerance){
            i++;
        }
        return std::pow(10,i);
    }

    double findC0(double f, double z) const {

        auto function = [&](double c_0) {
        // double result = simpsonsRule([&](double x){
        double result = rombergIntegration([&](double x){
            return (x - 1.0) * pdf_DM_cosmo(x, c_0, 1.0, f, z);
        }, 0, find_pdf_DM_cosmo(f,z,c_0));
        //std::numeric_limits<double>::infinity());

        return result;
    };

    // std::cout << "function:"<< function(1.0)<< std::endl;
    // Wider search range and more robust root-finding
    try {
        return bisection(function, 0.01, 20.0);
    }
    catch (const std::exception& e) {
        std::cerr << "Bisection method failed: " << e.what() << std::endl;
        throw std::runtime_error("Unable to find C0");
    }
}

//     double findC0(double f, double z) const {
//     // // Simplified C0 finding using numerical method
//     // auto average_ratio = [&](double c_0) {
//     //     // auto x_pdf = [&](double x) { 
//     //     //     return x * pdf_DM_cosmo(x, c_0, 1.0, f, z);
//     //     // };
//     //     double average = simpsonsRule([&](double x){
//     //         return x * pdf_DM_cosmo(x, c_0, 1.0, f, z);
//     //     }, 0, std::numeric_limits<double>::infinity(), 3);
//     //     double normal = simpsonsRule([&](double x) { 
//     //         return pdf_DM_cosmo(x, c_0, 1.0, f, z); 
//     //     }, 0, std::numeric_limits<double>::infinity(), 3);

//     //     double result = average / normal - 1.0;

//     auto function = [&](double c_0) {
//         // double result = simpsonsRule([&](double x){
//         double result = rombergIntegration([&](double x){
//             return (x - 1.0) * pdf_DM_cosmo(x, c_0, 1.0, f, z);
//         //}, 0, std::numeric_limits<double>::infinity(), 3);
//         }, 0, std::numeric_limits<double>::infinity());

//         //std::cout << "Private Variable normal: "<< result;
//         //std::cout << std::endl;
//         if (std::isnan(result)) {
//             throw std::runtime_error("Finding C0 in NaN");
//         }

//         return result;
//     };

//     // Simplified root-finding (replace with more robust method in practice)
//     // return brentMethod(function, 0.1, 10.0);
//     return bisection(function, 0.1, 10.0);
// }

    double findA(double C_0, double F, double z){
        //double integral = simpsonsRule([&](double x) {
        double integral = rombergIntegration([&](double x){
            return pdf_DM_cosmo(x, C_0, 1.0, F, z); 
        //}, 0, std::numeric_limits<double>::infinity());
        }, 0, find_pdf_DM_cosmo(F,z,C_0));

        double result = 1.0/integral;
        if (std::isnan(result)) {
            throw std::runtime_error("Finding A in NaN");
        }

        return result;
    };

    double dDM_integrand_w(double z, double Om, double w) const {
            return (1.0+z)/std::sqrt(Om*std::pow(1.0+z,3)+(1-Om)*std::pow((1+z),(3*(1+w))));
        };
    
    double DM_IGM_O_bh_70(double z, double O_bh_70) const {
        // Constants (you may need to adjust these to match the original implementation)
        const double C_LIGHT = 299792458.0;  // m/s
        const double KM_2_MPC = 3.24e-20; //3.24078e-20;  // km to Mpc conversion
        const double G_NEWTON = 6.67430e-11;  // m^3 kg^-1 s^-2
        const double M_PROTON = 1.67262192369e-27; //1.672621898e-27;  // kg
        const double DM_2_PCCM3 = 3.24e-23;  // conversion factor
        const double f_IGM = 0.83;  // default value, adjust as needed
        const double OMEGA_MATTER = 0.2865;
        const double w = -1.0;

        // Compute O_bH_0 
        double O_bH_0 = O_bh_70 * 70;

        // Compute integral (simplified - you may need to replace with a more accurate integration method)
        //long double integral = simpsonsRule([&](double x) { 
        double integral = rombergIntegration([&](double x) { 
            return dDM_integrand_w(x, OMEGA_MATTER, w); 
        //}, 0, z, 3);
        }, 0, z);

        // Compute factor
        double factor = 3 * C_LIGHT * KM_2_MPC * O_bH_0 * f_IGM / 
                   (8 * PI * G_NEWTON * M_PROTON) * (7.0 / 8.0);

        // Unit transformation
        double unit_transform = DM_2_PCCM3;

        // Compute DM
        double DM = unit_transform * factor * integral;

        if (std::isnan(DM)) {
            throw std::runtime_error("Finding DM_IGM_O_bh_70 in NaN");
        }

    return DM;
}

    // double PDF_DM_cosmo(double DM, double O_bh_70, double F, double z){
    //     double C_0 = findC0(F, z);
    //     double A = findA(C_0, F, z);
    //     double delta = DM/DM_IGM_O_bh_70(z, O_bh_70);
    //     //std::cout << "Private Variable delta: "<< delta;
    //     //std::cout << std::endl;
    //     return pdf_DM_cosmo(delta, C_0, A, F, z);
    // };

//public:

    double calculate_dm_probability(
        double DM_ext, double z, double F, double O_bh_70, double sigma_host, double e_mu
    ) {

        double C_0 = findC0(F, z);
        double A = findA(C_0, F, z);
        double DM_std = DM_IGM_O_bh_70(z,O_bh_70);
        double delta;

        //double integrand = simpsonsRule([&](double DM_host) {
        double integrand = rombergIntegration([&](double DM_host) {
            double p_host_val = PDF_DM_host(DM_host, e_mu, sigma_host);
            delta = (DM_ext - DM_host/(1+z))/DM_std;
            //double p_cosmo_val = PDF_DM_cosmo(DM_ext - DM_host/(1.0+z), O_bh_70, F, z);
            double p_cosmo_val = pdf_DM_cosmo(delta, C_0, A, F, z);
            return p_host_val * p_cosmo_val;
        //}, 0, DM_ext, 3);
        }, 0, DM_ext);

        // Implement numerical integration
        // This would use a method like Simpson's rule or adaptive quadrature
        return integrand;
    }

    double bisection(const std::function<double(double)>& f, double a, double b, double tolerance = 1e-6) const{
        double fa = f(a);
        double fb = f(b);

        if (fa * fb >=0) {
            std::cout << "fa: "<< fa;
            std::cout << "fb: "<< fb;
            throw std::runtime_error("Choose larger scale for bisection method");
        }

        double c = (a + b) / 2.0;
        double fc = f(c);

        while (std::abs(b-a)>=tolerance){
            if (fc==0){
                return c;
            }
            else if(fc * fa <= 0){
                b=c;
            }
            else{
                a=c;
            }
            c=(a + b) / 2.0;
            fc = f(c);
        }
        return c;

//     double brentMethod(const std::function<double(double)>& f, double a, double b, 
//                    double tolerance = 1e-6, int max_iter = 100) const{
//     double fa = f(a);
//     double fb = f(b);

//     if (std::abs(fa) < std::abs(fb)) {
//         std::swap(a, b);
//         std::swap(fa, fb);
//     }

//     double c = a;
//     double fc = fa;
//     bool mflag = true;
//     double d = b - a;
//     double e = d;

//     for (int iter = 0; iter < max_iter; ++iter) {
//         if (std::abs(fc) < std::abs(fb)) {
//             std::swap(a, c);
//             std::swap(fa, fc);
//             std::swap(b, c);
//             std::swap(fb, fc);
//         }

//         double s;
//         if (std::abs(fa - fc) > tolerance && std::abs(fb - fc) > tolerance) {
//             // Inverse quadratic interpolation
//             s = a * fb * fc / ((fa - fb) * (fa - fc)) +
//                 b * fa * fc / ((fb - fa) * (fb - fc)) +
//                 c * fa * fb / ((fc - fa) * (fc - fb));
//         } else {
//             // Secant method
//             s = b - fb * (b - a) / (fb - fa);
//         }

//         bool condition1 = !(s >= (3 * a + b) / 4 && s <= b);
//         bool condition2 = mflag && std::abs(s - b) >= std::abs(b - c) / 2;
//         bool condition3 = !mflag && std::abs(s - b) >= std::abs(c - d) / 2;
//         bool condition4 = mflag && std::abs(b - c) < tolerance;
//         bool condition5 = !mflag && std::abs(c - d) < tolerance;

//         if (condition1 || condition2 || condition3 || condition4 || condition5) {
//             s = (a + b) / 2;
//             mflag = true;
//         } else {
//             mflag = false;
//         }

//         double fs = f(s);
//         d = c;
//         c = b;
//         fc = fb;

//         if (fa * fs < 0) {
//             b = s;
//             fb = fs;
//         } else {
//             a = s;
//             fa = fs;
//         }

//         if (std::abs(fa) < std::abs(fb)) {
//             std::swap(a, b);
//             std::swap(fa, fb);
//         }

//         if (std::abs(b - a) < tolerance) {
//             return b;
//         }
//     }

//     return b;

    }


//     double simpsonsRule(const std::function<double(double)>& func, double a, double b, int n) const {
//     // Handle improper integrals with large upper limit

//     if (a==0){
//             a=1e-6;
//         }

//     if (std::isinf(b)) {
//         // Adaptive technique for infinite upper limit
//         double result = 0.0;
//         double x = a;
//         double step = 1.0;

//         while (x < 1000) {  // Truncate at a large finite value
//             double x_next = x + step;
//             result += (step / 6.0) * (
//                 func(x) + 
//                 4.0 * func((x + x_next) / 2.0) + 
//                 func(x_next)
//             );
//             x = x_next;
//             step *= 1.5;  // Exponential step increase
//         }
        
//         return result;
//     }
    
//     // Standard Simpson's rule for finite limits
//     if (n % 2 != 0) n += 1;  // Ensure even number of subintervals
    
//     double h = (b - a) / n;
//     double result = func(a) + func(b);
    
//     for (int i = 1; i < n; i += 2) {
//         double x = a + i * h;
//         result += 4.0 * func(x);
//     }
    
//     for (int i = 2; i < n - 1; i += 2) {
//         double x = a + i * h;
//         result += 2.0 * func(x);
//     }
    
//     return result * h / 3.0;
// }

double rombergIntegration(const std::function<double(double)>& f, 
                          double a, 
                          double b, 
                          double tolerance = 1e-8, 
                          int max_iterations = 20) const {
    //Handle infinite upper limit by using a change of variable

    auto transformedF = [f, a, b](double t) {
        if (b == std::numeric_limits<double>::infinity()) {
            // variable replace：x = a + t / (1 - t)
            // Jacobian: dx/dt = 1 / (1-t)²
            // Integration x [a, +∞] to t [0,1]
            double x = a + t / (1 - t);
            return f(x) / ((1 - t) * (1 - t));
        }
        else{
            // Integration x [a, b] to t [0,1]
            double x = a + t * (b - a);
            return f(x) * (b-a);
        }
    };

    //max_iterations × max_iterations all zero matrix
    std::vector<std::vector<double>> R(max_iterations, std::vector<double>(max_iterations, 0.0)); 
    
    // Trapezoidal rule for first iteration
    R[0][0] = (transformedF(0.0) + transformedF(1.0)) / 2.0;
    //R[0][0] = (b-a) * (f(a) + f(b)) / 2.0;


    for (int i = 1; i < max_iterations; i++) {
        // Composite trapezoidal rule
        //double h = (b-a) * std::pow(0.5, i);
        double h = std::pow(0.5, i);
        double sum = 0.0;
        int k_max = std::pow(2, i-1);
        
        for (int j = 1; j <= k_max; j++) {
            double c = (2.0*j-1.0) * h;//+ a;
            sum += transformedF(c); // f(c);
        }
        
        R[i][0] = 0.5 * R[i-1][0] + h * sum;

        // Richardson extrapolation
        for (int j = 1; j <= i; j++) {
            R[i][j] = (std::pow(4.0, j) * R[i][j-1] - R[i-1][j-1]) / 
                      (std::pow(4.0, j) - 1);
        }

        // // Check for convergence
        // if (std::abs(R[i][i] - R[i][i-1]) < tolerance) {
        //     return R[i][i];
        // }
    }

    //std::cout << "max_iterations may not enough for tolerance setting"<<std::endl;
    return R[max_iterations-1][max_iterations-1];


    // // Boundary for infinite limit integration
    // auto transformedIntegrationBounds = [&](double t) {
    //     return b == std::numeric_limits<double>::infinity() ? 1.0 : b;
    // };

    // std::vector<std::vector<double>> R(max_iterations, std::vector<double>(max_iterations, 0.0));
    
    // // Trapezoidal rule for first iteration
    // R[0][0] = (transformedIntegrationBounds(1.0) - 0.0) * 
    //           (transformedF(0.0) + transformedF(1.0)) / 2.0;

    // for (int i = 1; i < max_iterations; ++i) {
    //     // Composite trapezoidal rule
    //     double h = std::pow(0.5, i);
    //     double sum = 0.0;
    //     int steps = 1 << i;
        
    //     for (int j = 1; j < steps; j += 2) {
    //         double t = j * h;
    //         sum += transformedF(t);
    //     }
        
    //     R[i][0] = 0.5 * R[i-1][0] + h * sum;

    //     // Richardson extrapolation
    //     for (int j = 1; j <= i; ++j) {
    //         R[i][j] = (std::pow(4.0, j) * R[i][j-1] - R[i-1][j-1]) / 
    //                   (std::pow(4.0, j) - 1);
    //     }

    //     // Check for convergence
    //     if (std::abs(R[i][i] - R[i][i-1]) < tolerance) {
    //         return R[i][i];
    //     }
    // }

    // return R[max_iterations-1][max_iterations-1];
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


double test_f(double f, double z) {

    double result = rombergIntegration([&](double x){
            return (x - 1.0) * pdf_DM_cosmo(x, 3.0, 1.0, f, z);
        }, 0, 10000, 1e-8, 20);
        //std::numeric_limits<double>::infinity());

        return result;
}

};

int main(){

    FRBProbabilityCalculator calculator;
    double integration;
    double p1;

    p1=calculator.calculate_dm_probability(300,0.2,0.1,0.05,0.25,50);
    std::cout << "p1="<<p1<<std::endl;

    integration=calculator.test_f(0.1,0.2);
    //integration=calculator.findC0(0.1,0.2);
    std::cout << "integration="<<integration<<std::endl;

    return 0;
}

// int main() {
//     FRBProbabilityCalculator calculator;

//     // test
//     // double simpsons_result = calculator.simpsonsRule(
//     //     [&](double x){ return 1.0/std::pow(x,2); }, 0.5, std::numeric_limits<double>::infinity(), 5
//     // );

//     // std::cout << "Simpson's result: " << simpsons_result << std::endl;
//     // double p;
//     // try{
//     //     p = calculator.calculate_dm_probability(100,0.2,0.1,0.05,0.25,50);
//     // }
//     // catch (const std::exception& e) {
//     //     std::cout << "Error: " << e.what() << std::endl;
//     // }
//     // printf("%f\n", p);

//     // Define parameter ranges (same as in Python script)
//     std::vector<double> F_array = {0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5};
//     std::vector<double> O_bh_70_array = {0.015, 0.035, 0.055, 0.075, 0.095};
//     std::vector<double> sigma_host_array = {0.2, 0.6, 1.0, 1.4, 1.8};
//     std::vector<double> e_mu_array = {20, 60, 100, 140, 180};
//     std::vector<std::vector<std::vector<std::vector<double>>>> posterior_4D;

//     // Prepare data (you'll need to load actual data similar to the Python script)
//     std::vector<std::pair<double, double>> data = {
//         {50.5, 0.3}, // example entry: {DM_frb, z}
//         {75.2, 0.5}  // Add more real data entries
//     };

//     // Calculate posterior
//     try{
//         posterior_4D = calculator.calculate_posterior_4D(
//             F_array, O_bh_70_array, sigma_host_array, e_mu_array, data
//         );
//     }
//     catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }

//     // Save to CSV
//     calculator.save_posterior_to_csv(
//         posterior_4D, F_array, O_bh_70_array, 
//         sigma_host_array, e_mu_array, "posterior_results.csv"
//     );

//     return 0;
// }