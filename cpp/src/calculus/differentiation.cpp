#include "morphosml/calculus/differentiation.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace morphosml {
namespace calculus {

double derivative(const std::function<double(double)>& f, double x, int order, double h) {
    if (order < 1 || order > 3) {
        throw std::invalid_argument("Derivative order must be 1, 2, or 3");
    }

    if (h <= 0.0) {
        // Machine-epsilon balanced step size
        double abs_x = std::abs(x);
        double scale = (abs_x > 1.0) ? abs_x : 1.0;
        if (order == 1) {
            h = scale * 1e-5;
        } else if (order == 2) {
            h = scale * 1e-4;
        } else {
            h = scale * 1e-3;
        }
    }

    if (order == 1) {
        // 5-point central stencil: O(h^4) accuracy
        double fm2 = f(x - 2.0 * h);
        double fm1 = f(x - h);
        double fp1 = f(x + h);
        double fp2 = f(x + 2.0 * h);
        return (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * h);
    } else if (order == 2) {
        // 5-point central stencil for second derivative: O(h^4) accuracy
        double fm2 = f(x - 2.0 * h);
        double fm1 = f(x - h);
        double f0  = f(x);
        double fp1 = f(x + h);
        double fp2 = f(x + 2.0 * h);
        return (-fp2 + 16.0 * fp1 - 30.0 * f0 + 16.0 * fm1 - fm2) / (12.0 * h * h);
    } else {
        // Central stencil for third derivative: O(h^2) accuracy
        double fm2 = f(x - 2.0 * h);
        double fm1 = f(x - h);
        double fp1 = f(x + h);
        double fp2 = f(x + 2.0 * h);
        return (fp2 - 2.0 * fp1 + 2.0 * fm1 - fm2) / (2.0 * h * h * h);
    }
}

std::vector<double> gradient(const std::function<double(const std::vector<double>&)>& f,
                             const std::vector<double>& x,
                             double h) {
    const size_t n = x.size();
    std::vector<double> grad(n);
    std::vector<double> x_perturbed = x;

    for (size_t i = 0; i < n; ++i) {
        double xi = x[i];
        double hi = h * ((std::abs(xi) > 1.0) ? std::abs(xi) : 1.0);

        // 5-point stencil along axis i
        x_perturbed[i] = xi + 2.0 * hi;
        double fp2 = f(x_perturbed);

        x_perturbed[i] = xi + hi;
        double fp1 = f(x_perturbed);

        x_perturbed[i] = xi - hi;
        double fm1 = f(x_perturbed);

        x_perturbed[i] = xi - 2.0 * hi;
        double fm2 = f(x_perturbed);

        // Reset
        x_perturbed[i] = xi;

        grad[i] = (-fp2 + 8.0 * fp1 - 8.0 * fm1 + fm2) / (12.0 * hi);
    }

    return grad;
}

Matrix jacobian(const std::function<std::vector<double>(const std::vector<double>&)>& f,
                const std::vector<double>& x,
                double h) {
    const size_t n = x.size();
    std::vector<double> f0 = f(x);
    const size_t m = f0.size();

    Matrix J(m, n);
    std::vector<double> x_perturbed = x;

    for (size_t j = 0; j < n; ++j) {
        double xj = x[j];
        double hj = h * ((std::abs(xj) > 1.0) ? std::abs(xj) : 1.0);

        x_perturbed[j] = xj + 2.0 * hj;
        std::vector<double> fp2 = f(x_perturbed);

        x_perturbed[j] = xj + hj;
        std::vector<double> fp1 = f(x_perturbed);

        x_perturbed[j] = xj - hj;
        std::vector<double> fm1 = f(x_perturbed);

        x_perturbed[j] = xj - 2.0 * hj;
        std::vector<double> fm2 = f(x_perturbed);

        x_perturbed[j] = xj;

        for (size_t i = 0; i < m; ++i) {
            J(i, j) = (-fp2[i] + 8.0 * fp1[i] - 8.0 * fm1[i] + fm2[i]) / (12.0 * hj);
        }
    }

    return J;
}

Matrix hessian(const std::function<double(const std::vector<double>&)>& f,
               const std::vector<double>& x,
               double h) {
    const size_t n = x.size();
    Matrix H(n, n);
    std::vector<double> x_p = x;
    double f0 = f(x);

    std::vector<double> h_vec(n);
    for (size_t i = 0; i < n; ++i) {
        h_vec[i] = h * ((std::abs(x[i]) > 1.0) ? std::abs(x[i]) : 1.0);
    }

    for (size_t i = 0; i < n; ++i) {
        // Diagonal: second partial derivative d^2 f / dx_i^2
        x_p[i] = x[i] + h_vec[i];
        double f_pi = f(x_p);

        x_p[i] = x[i] - h_vec[i];
        double f_mi = f(x_p);

        x_p[i] = x[i];
        H(i, i) = (f_pi - 2.0 * f0 + f_mi) / (h_vec[i] * h_vec[i]);

        // Off-diagonal: cross partial derivatives d^2 f / (dx_i dx_j)
        for (size_t j = i + 1; j < n; ++j) {
            x_p[i] = x[i] + h_vec[i];
            x_p[j] = x[j] + h_vec[j];
            double f_pp = f(x_p);

            x_p[j] = x[j] - h_vec[j];
            double f_pm = f(x_p);

            x_p[i] = x[i] - h_vec[i];
            x_p[j] = x[j] + h_vec[j];
            double f_mp = f(x_p);

            x_p[j] = x[j] - h_vec[j];
            double f_mm = f(x_p);

            // Reset
            x_p[i] = x[i];
            x_p[j] = x[j];

            double val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_vec[i] * h_vec[j]);
            H(i, j) = val;
            H(j, i) = val; // Symmetry
        }
    }

    return H;
}

} // namespace calculus
} // namespace morphosml

