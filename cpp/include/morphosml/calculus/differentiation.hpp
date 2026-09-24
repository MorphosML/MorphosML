#pragma once

#include <vector>
#include <functional>
#include "morphosml/matrix.hpp"

namespace morphosml {
namespace calculus {

/**
 * @brief Computes the numerical derivative of a single-variable function \(f: \mathbb{R} \to \mathbb{R}\).
 *
 * @details
 * Employs high-precision central finite difference stencils:
 * - **1st Order (\(f'\))**: 5-point central stencil with truncation error \(\mathcal{O}(h^4)\):
 *   \f[
 *   f'(x) \approx \frac{-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)}{12h}
 *   \f]
 * - **2nd Order (\(f''\))**: 5-point central stencil with \(\mathcal{O}(h^4)\) accuracy:
 *   \f[
 *   f''(x) \approx \frac{-f(x + 2h) + 16f(x + h) - 30f(x) + 16f(x - h) - f(x - 2h)}{12h^2}
 *   \f]
 * - **3rd Order (\(f'''\))**: 5-point central stencil with \(\mathcal{O}(h^2)\) accuracy:
 *   \f[
 *   f'''(x) \approx \frac{f(x + 2h) - 2f(x + h) + 2f(x - h) - f(x - 2h)}{2h^3}
 *   \f]
 *
 * @param f Continuous scalar function \(f(x)\).
 * @param x Evaluation point.
 * @param order Derivative order (1, 2, or 3).
 * @param h Step size. If 0.0, optimal machine-epsilon step sizes are automatically chosen:
 *          \(h = \max(|x|, 1.0) \times \epsilon^{1/(order + 1)}\).
 * @return double Approximated derivative value.
 * @throws std::invalid_argument If `order` is not 1, 2, or 3.
 */
double derivative(const std::function<double(double)>& f, double x, int order = 1, double h = 0.0);

/**
 * @brief Computes the multivariate gradient vector \(\nabla f(\mathbf{x})\) of \(f: \mathbb{R}^n \to \mathbb{R}\).
 *
 * @details
 * Calculates partial derivatives \(\frac{\partial f}{\partial x_i}\) using 4th-order central differences:
 * \f[
 * \nabla f(\mathbf{x}) = \left[ \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right]^T
 * \f]
 *
 * @param f Multivariate scalar function.
 * @param x Evaluation coordinate vector \(\mathbf{x} \in \mathbb{R}^n\).
 * @param h Perturbation step size (default: 1e-5).
 * @return std::vector<double> Gradient vector of dimension \(n\).
 */
std::vector<double> gradient(const std::function<double(const std::vector<double>&)>& f,
                             const std::vector<double>& x,
                             double h = 1e-5);

/**
 * @brief Computes the Jacobian matrix \(J \in \mathbb{R}^{m \times n}\) of \(\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m\).
 *
 * @details
 * Evaluates the matrix of all first-order partial derivatives:
 * \f[
 * J_{i, j} = \frac{\partial f_i}{\partial x_j}(\mathbf{x})
 * \f]
 *
 * @param f Vector-valued function returning \(\mathbb{R}^m\) given input in \(\mathbb{R}^n\).
 * @param x Evaluation coordinate vector in \(\mathbb{R}^n\).
 * @param h Perturbation step size (default: 1e-5).
 * @return Matrix Jacobian matrix of shape \((m, n)\).
 */
Matrix jacobian(const std::function<std::vector<double>(const std::vector<double>&)>& f,
                const std::vector<double>& x,
                double h = 1e-5);

/**
 * @brief Computes the Hessian matrix \(H \in \mathbb{R}^{n \times n}\) of \(f: \mathbb{R}^n \to \mathbb{R}\).
 *
 * @details
 * Evaluates the symmetric matrix of second-order partial derivatives:
 * \f[
 * H_{i, j} = \frac{\partial^2 f}{\partial x_i \partial x_j}(\mathbf{x})
 * \f]
 * Diagonal elements \(H_{i, i}\) use the 5-point second-derivative stencil. Off-diagonal elements
 * \(H_{i, j}\) use the symmetric cross-derivative formula:
 * \f[
 * \frac{\partial^2 f}{\partial x_i \partial x_j} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i + h\mathbf{e}_j) - f(\mathbf{x} + h\mathbf{e}_i - h\mathbf{e}_j) - f(\mathbf{x} - h\mathbf{e}_i + h\mathbf{e}_j) + f(\mathbf{x} - h\mathbf{e}_i - h\mathbf{e}_j)}{4h^2}
 * \f]
 *
 * @param f Multivariate scalar function.
 * @param x Evaluation coordinate vector in \(\mathbb{R}^n\).
 * @param h Perturbation step size (default: 1e-4).
 * @return Matrix Symmetric Hessian matrix of shape \((n, n)\).
 */
Matrix hessian(const std::function<double(const std::vector<double>&)>& f,
               const std::vector<double>& x,
               double h = 1e-4);

} // namespace calculus
} // namespace morphosml
