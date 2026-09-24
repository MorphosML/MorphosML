#pragma once

#include <functional>
#include <string>

namespace morphosml {
namespace calculus {

/**
 * @brief Structured outcome of a numerical limit evaluation \(\lim_{x \to a} f(x)\).
 */
struct LimitResult {
    double value;           ///< Calculated limit value (or `std::numeric_limits<double>::infinity()`).
    bool exists;            ///< True if limit converged within tolerance.
    bool is_infinite;       ///< True if function monotonically diverged to \(\pm \infty\).
    std::string direction;  ///< Evaluated direction (`"left"`, `"right"`, or `"both"`).
};

/**
 * @brief Evaluates the numerical limit \(\lim_{x \to x_{\text{target}}} f(x)\).
 *
 * @details
 * Evaluates directional sequences approaching \(x_{\text{target}}\) using geometric decay
 * step sizes (\(h_k = 10^{-1}, 10^{-2}, \dots, 10^{-9}\)).
 * Features:
 * - Detects removable singularities (e.g., \(\lim_{x \to 0} \frac{\sin x}{x} = 1\)) even if \(f(0)\) is undefined.
 * - Distinguishes between finite convergence and monotonic infinite divergence (\(\pm \infty\)).
 * - For two-sided limits (`"both"`), checks left and right limit agreement within `tol`.
 *
 * @param f Continuous or piecewise continuous function \(f(x)\).
 * @param x_target Point of interest \(a\).
 * @param direction Evaluation direction: `"both"` (\(x \to a\)), `"left"` (\(x \to a^-\)), or `"right"` (\(x \to a^+\)).
 * @param tol Relative convergence tolerance (default: 1e-6).
 * @return LimitResult Structure containing computed value, existence flag, infinity flag, and direction.
 */
LimitResult evaluate_limit(const std::function<double(double)>& f,
                           double x_target,
                           const std::string& direction = "both",
                           double tol = 1e-6);

/**
 * @brief Convenience function returning the numerical limit as a `double`.
 *
 * @param f Function to evaluate.
 * @param x_target Point of interest.
 * @param direction Direction (`"both"`, `"left"`, or `"right"`).
 * @param tol Convergence tolerance.
 * @return double Limit value.
 * @throws std::runtime_error If the limit does not exist (left and right limits differ).
 */
double limit(const std::function<double(double)>& f,
             double x_target,
             const std::string& direction = "both",
             double tol = 1e-6);

} // namespace calculus
} // namespace morphosml
