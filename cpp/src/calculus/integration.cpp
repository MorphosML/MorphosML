#include "morphosml/calculus/integration.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>

namespace morphosml {
namespace calculus {

double trapezoidal(const std::function<double(double)>& f, double a, double b, size_t n) {
    if (n == 0) throw std::invalid_argument("Number of intervals n must be positive");
    if (a == b) return 0.0;

    double h = (b - a) / static_cast<double>(n);
    double sum = 0.5 * (f(a) + f(b));

    for (size_t i = 1; i < n; ++i) {
        sum += f(a + static_cast<double>(i) * h);
    }

    return sum * h;
}

double simpson(const std::function<double(double)>& f, double a, double b, size_t n) {
    if (n == 0) throw std::invalid_argument("Number of intervals n must be positive");
    if (a == b) return 0.0;

    // Simpson's 1/3 requires an even number of intervals
    if (n % 2 != 0) {
        n += 1;
    }

    double h = (b - a) / static_cast<double>(n);
    double sum = f(a) + f(b);

    for (size_t i = 1; i < n; ++i) {
        double x = a + static_cast<double>(i) * h;
        if (i % 2 == 1) {
            sum += 4.0 * f(x);
        } else {
            sum += 2.0 * f(x);
        }
    }

    return (h / 3.0) * sum;
}

double simpson_38(const std::function<double(double)>& f, double a, double b, size_t n) {
    if (n == 0) throw std::invalid_argument("Number of intervals n must be positive");
    if (a == b) return 0.0;

    // Simpson's 3/8 requires n to be a multiple of 3
    if (n % 3 != 0) {
        n += (3 - (n % 3));
    }

    double h = (b - a) / static_cast<double>(n);
    double sum = f(a) + f(b);

    for (size_t i = 1; i < n; ++i) {
        double x = a + static_cast<double>(i) * h;
        if (i % 3 == 0) {
            sum += 2.0 * f(x);
        } else {
            sum += 3.0 * f(x);
        }
    }

    return (3.0 * h / 8.0) * sum;
}

double gauss_legendre(const std::function<double(double)>& f, double a, double b, size_t n_points) {
    if (a == b) return 0.0;

    // Gauss-Legendre quadrature on standard interval [-1, 1] mapped to [a, b]
    // x = 0.5 * ((b - a) * t + (b + a))
    // dx = 0.5 * (b - a) dt
    struct Node {
        double root;
        double weight;
    };

    std::vector<Node> nodes;
    if (n_points <= 5) {
        // 5-point Gauss-Legendre
        nodes = {
            { 0.0, 0.5688888888888889 },
            { -0.5384693101056831, 0.4786286704993665 },
            {  0.5384693101056831, 0.4786286704993665 },
            { -0.9061798459386640, 0.2369268850561891 },
            {  0.9061798459386640, 0.2369268850561891 }
        };
    } else {
        // 10-point Gauss-Legendre
        nodes = {
            { -0.1488743389816312, 0.2955242247147529 },
            {  0.1488743389816312, 0.2955242247147529 },
            { -0.4333953941292472, 0.2692667193099963 },
            {  0.4333953941292472, 0.2692667193099963 },
            { -0.6794095682990244, 0.2190863625159820 },
            {  0.6794095682990244, 0.2190863625159820 },
            { -0.8650633666889845, 0.1494513491505806 },
            {  0.8650633666889845, 0.1494513491505806 },
            { -0.9739065285171717, 0.0666713443086881 },
            {  0.9739065285171717, 0.0666713443086881 }
        };
    }

    double mid = 0.5 * (a + b);
    double half_width = 0.5 * (b - a);

    double sum = 0.0;
    for (const auto& node : nodes) {
        double x = mid + half_width * node.root;
        sum += node.weight * f(x);
    }

    return sum * half_width;
}

double integrate(const std::function<double(double)>& f,
                 double a,
                 double b,
                 const std::string& method,
                 size_t n) {
    if (method == "simpson" || method == "simpson_13") {
        return simpson(f, a, b, n);
    } else if (method == "trapezoidal") {
        return trapezoidal(f, a, b, n);
    } else if (method == "simpson_38") {
        return simpson_38(f, a, b, n);
    } else if (method == "gauss_legendre" || method == "gauss") {
        return gauss_legendre(f, a, b, (n > 5 ? 10 : 5));
    }
    throw std::invalid_argument("Unknown integration method: " + method +
                                ". Supported: 'simpson', 'trapezoidal', 'simpson_38', 'gauss_legendre'");
}

double integrate_2d(const std::function<double(double, double)>& f,
                    double x_a,
                    double x_b,
                    double y_a,
                    double y_b,
                    size_t nx,
                    size_t ny) {
    auto inner_integral = [&](double y) {
        auto fx = [&](double x) { return f(x, y); };
        return simpson(fx, x_a, x_b, nx);
    };

    return simpson(inner_integral, y_a, y_b, ny);
}

} // namespace calculus
} // namespace morphosml
