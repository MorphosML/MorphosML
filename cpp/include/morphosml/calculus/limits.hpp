#pragma once

#include <functional>
#include <string>

namespace morphosml {
namespace calculus {

struct LimitResult {
    double value;
    bool exists;
    bool is_infinite;
    std::string direction;
};

// Evaluates numerical limit lim_{x -> x_target} f(x)
// direction: "left" (x -> a^-), "right" (x -> a^+), "both" (two-sided)
LimitResult evaluate_limit(const std::function<double(double)>& f,
                           double x_target,
                           const std::string& direction = "both",
                           double tol = 1e-6);

// Convenience wrapper returning double directly (throws std::runtime_error if limit does not exist)
double limit(const std::function<double(double)>& f,
             double x_target,
             const std::string& direction = "both",
             double tol = 1e-6);

} // namespace calculus
} // namespace morphosml
