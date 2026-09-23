#pragma once

#include <vector>
#include <functional>
#include "morphosml/matrix.hpp"

namespace morphosml {
namespace calculus {

// Single-variable numerical derivative with up to O(h^4) stencils
double derivative(const std::function<double(double)>& f, double x, int order = 1, double h = 0.0);

// Multivariate gradient: returns grad f(x) as std::vector<double>
std::vector<double> gradient(const std::function<double(const std::vector<double>&)>& f,
                             const std::vector<double>& x,
                             double h = 1e-5);

// Jacobian matrix for vector-valued function f: R^n -> R^m (returns Matrix of shape (m, n))
Matrix jacobian(const std::function<std::vector<double>(const std::vector<double>&)>& f,
                const std::vector<double>& x,
                double h = 1e-5);

// Hessian matrix for scalar-valued function f: R^n -> R (returns symmetric Matrix of shape (n, n))
Matrix hessian(const std::function<double(const std::vector<double>&)>& f,
               const std::vector<double>& x,
               double h = 1e-4);

} // namespace calculus
} // namespace morphosml
