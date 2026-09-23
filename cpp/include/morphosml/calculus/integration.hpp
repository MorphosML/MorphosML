#pragma once

#include <functional>
#include <string>
#include <cstddef>

namespace morphosml {
namespace calculus {

/**
 * @brief Approximates the definite integral \(\int_a^b f(x)\,dx\) using the Composite Trapezoidal rule.
 *
 * @param f Integrand function.
 * @param a Lower bound of integration.
 * @param b Upper bound of integration.
 * @param n Number of subintervals (default: 1000).
 * @return double Approximate integral value.
 */
double trapezoidal(const std::function<double(double)>& f, double a, double b, size_t n = 1000);

/**
 * @brief Approximates the definite integral \(\int_a^b f(x)\,dx\) using Composite Simpson's 1/3 rule (\(\mathcal{O}(h^4)\)).
 *
 * @param f Integrand function.
 * @param a Lower bound of integration.
 * @param b Upper bound of integration.
 * @param n Number of subintervals (must be even; adjusted automatically if odd).
 * @return double Approximate integral value.
 */
double simpson(const std::function<double(double)>& f, double a, double b, size_t n = 1000);

/**
 * @brief Approximates the definite integral \(\int_a^b f(x)\,dx\) using Composite Simpson's 3/8 rule.
 *
 * @param f Integrand function.
 * @param a Lower bound of integration.
 * @param b Upper bound of integration.
 * @param n Number of subintervals (must be divisible by 3).
 * @return double Approximate integral value.
 */
double simpson_38(const std::function<double(double)>& f, double a, double b, size_t n = 999);

/**
 * @brief Approximates the definite integral \(\int_a^b f(x)\,dx\) using Gauss-Legendre Quadrature.
 *
 * @param f Integrand function.
 * @param a Lower bound of integration.
 * @param b Upper bound of integration.
 * @param n_points Number of quadrature points (supports 5 or 10 points).
 * @return double High-precision numerical quadrature approximation.
 * @throws std::invalid_argument If `n_points` is not 5 or 10.
 */
double gauss_legendre(const std::function<double(double)>& f, double a, double b, size_t n_points = 5);

/**
 * @brief Unified 1D numerical integration dispatcher.
 *
 * @param f Integrand function.
 * @param a Lower bound of integration.
 * @param b Upper bound of integration.
 * @param method Integration algorithm: `"simpson"`, `"trapezoidal"`, `"simpson_38"`, or `"gauss_legendre"`.
 * @param n Number of subintervals (ignored for `"gauss_legendre"` which uses 5 points).
 * @return double Approximate definite integral.
 * @throws std::invalid_argument If `method` is unknown.
 */
double integrate(const std::function<double(double)>& f,
                 double a,
                 double b,
                 const std::string& method = "simpson",
                 size_t n = 1000);

/**
 * @brief Computes 2D numerical double integration \(\int_{y_a}^{y_b} \int_{x_a}^{x_b} f(x, y)\,dx\,dy\).
 *
 * @param f 2D integrand function \(f(x, y)\).
 * @param x_a Lower bound for \(x\).
 * @param x_b Upper bound for \(x\).
 * @param y_a Lower bound for \(y\).
 * @param y_b Upper bound for \(y\).
 * @param nx Number of subintervals along \(x\) axis (default: 100).
 * @param ny Number of subintervals along \(y\) axis (default: 100).
 * @return double Numerical approximation of the 2D volume integral.
 */
double integrate_2d(const std::function<double(double, double)>& f,
                    double x_a,
                    double x_b,
                    double y_a,
                    double y_b,
                    size_t nx = 100,
                    size_t ny = 100);

} // namespace calculus
} // namespace morphosml
