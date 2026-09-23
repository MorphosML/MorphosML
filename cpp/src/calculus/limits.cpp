#include "morphosml/calculus/limits.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace morphosml {
namespace calculus {

static LimitResult evaluate_one_sided(const std::function<double(double)>& f,
                                      double x_target,
                                      bool is_right,
                                      double tol) {
    const double sign = is_right ? 1.0 : -1.0;
    std::vector<double> deltas = {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12};
    std::vector<double> vals;
    vals.reserve(deltas.size());

    const double inf = std::numeric_limits<double>::infinity();

    for (double delta : deltas) {
        double x = x_target + sign * delta;
        try {
            double y = f(x);
            if (std::isnan(y)) {
                continue;
            }
            vals.push_back(y);
        } catch (...) {
            continue;
        }
    }

    LimitResult res;
    res.direction = is_right ? "right" : "left";
    res.exists = false;
    res.is_infinite = false;
    res.value = std::numeric_limits<double>::quiet_NaN();

    if (vals.size() < 2) {
        return res;
    }

    // Check for divergence toward +/- infinity:
    // If values are rapidly growing in magnitude and exceed 1e6
    double last_val = vals.back();
    double prev_val = vals[vals.size() - 2];
    if (std::abs(last_val) > 1e6 && std::abs(last_val) > 3.0 * std::abs(prev_val)) {
        res.is_infinite = true;
        res.value = (last_val > 0) ? inf : -inf;
        res.exists = false;
        return res;
    }

    // Check convergence across consecutive steps
    for (size_t i = vals.size() - 1; i >= 1; --i) {
        if (std::abs(vals[i] - vals[i - 1]) <= tol * (1.0 + std::abs(vals[i]))) {
            res.exists = true;
            res.value = vals[i];
            return res;
        }
    }

    // Fallback: return the closest evaluated point if variance is small
    if (std::abs(vals.back() - vals[vals.size() - 2]) <= 1e-3) {
        res.exists = true;
        res.value = vals.back();
        return res;
    }

    return res;
}

LimitResult evaluate_limit(const std::function<double(double)>& f,
                           double x_target,
                           const std::string& direction,
                           double tol) {
    if (direction == "left") {
        return evaluate_one_sided(f, x_target, false, tol);
    } else if (direction == "right") {
        return evaluate_one_sided(f, x_target, true, tol);
    } else if (direction == "both") {
        LimitResult left = evaluate_one_sided(f, x_target, false, tol);
        LimitResult right = evaluate_one_sided(f, x_target, true, tol);

        LimitResult res;
        res.direction = "both";
        res.exists = false;
        res.is_infinite = false;
        res.value = std::numeric_limits<double>::quiet_NaN();

        if (left.is_infinite && right.is_infinite && (left.value == right.value)) {
            res.is_infinite = true;
            res.value = left.value;
            return res;
        }

        if (left.exists && right.exists) {
            if (std::abs(left.value - right.value) <= 10.0 * tol * (1.0 + std::abs(left.value))) {
                res.exists = true;
                res.value = 0.5 * (left.value + right.value);
                return res;
            }
        }

        return res;
    }

    throw std::invalid_argument("Unknown limit direction: " + direction + ". Use 'left', 'right', or 'both'.");
}

double limit(const std::function<double(double)>& f,
             double x_target,
             const std::string& direction,
             double tol) {
    LimitResult res = evaluate_limit(f, x_target, direction, tol);
    if (res.is_infinite) {
        return res.value; // +/- inf
    }
    if (!res.exists) {
        throw std::runtime_error("Limit does not exist at x = " + std::to_string(x_target) + " (direction: " + direction + ")");
    }
    return res.value;
}

} // namespace calculus
} // namespace morphosml
