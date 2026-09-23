#pragma once

#include <functional>
#include <string>

namespace morphosml {
namespace calculus {

// 1D Numerical Integration Methods
double trapezoidal(const std::function<double(double)>& f, double a, double b, size_t n = 1000);
double simpson(const std::function<double(double)>& f, double a, double b, size_t n = 1000);
double simpson_38(const std::function<double(double)>& f, double a, double b, size_t n = 999);
double gauss_legendre(const std::function<double(double)>& f, double a, double b, size_t n_points = 5);

// Unified 1D Dispatcher
double integrate(const std::function<double(double)>& f,
                 double a,
                 double b,
                 const std::string& method = "simpson",
                 size_t n = 1000);

// 2D Numerical Double Integration: integral_{y_a}^{y_b} integral_{x_a}^{x_b} f(x, y) dx dy
double integrate_2d(const std::function<double(double, double)>& f,
                    double x_a,
                    double x_b,
                    double y_a,
                    double y_b,
                    size_t nx = 100,
                    size_t ny = 100);

} // namespace calculus
} // namespace morphosml
