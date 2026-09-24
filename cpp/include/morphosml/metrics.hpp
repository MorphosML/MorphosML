#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <cstddef>

namespace morphosml {

/**
 * @brief Computes classification accuracy between ground truth and predicted labels.
 *
 * \f[
 * \text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i = \hat{y}_i)
 * \f]
 *
 * @param y_true Ground truth class labels vector.
 * @param y_hat Predicted class labels vector.
 * @return double Accuracy score in \([0.0, 1.0]\).
 * @throws std::invalid_argument If inputs are empty or have mismatched sizes.
 */
inline double accuracy(const std::vector<int>& y_true, const std::vector<int>& y_hat) {
    if (y_true.size() != y_hat.size() || y_true.empty()) {
        throw std::invalid_argument("Input vectors must be non-empty and of matching size");
    }
    size_t correct = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_hat[i]) correct++;
    }
    return static_cast<double>(correct) / y_true.size();
}

/**
 * @brief Computes continuous Mean Squared Error (MSE).
 *
 * \f[
 * \text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2
 * \f]
 *
 * @param y_true True continuous target values.
 * @param y_hat Predicted continuous target values.
 * @return double Mean squared error.
 * @throws std::invalid_argument If inputs are empty or have mismatched sizes.
 */
inline double mse(const std::vector<double>& y_true, const std::vector<double>& y_hat) {
    if (y_true.size() != y_hat.size() || y_true.empty()) {
        throw std::invalid_argument("Input vectors must be non-empty and of matching size");
    }
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_hat[i];
        sum += diff * diff;
    }
    return sum / static_cast<double>(y_true.size());
}

/**
 * @brief Computes Mean Squared Error for discrete integer targets.
 *
 * @param y_true True integer targets.
 * @param y_hat Predicted integer targets.
 * @return double Mean squared error.
 * @throws std::invalid_argument If inputs are empty or have mismatched sizes.
 */
inline double mse(const std::vector<int>& y_true, const std::vector<int>& y_hat) {
    if (y_true.size() != y_hat.size() || y_true.empty()) {
        throw std::invalid_argument("Input vectors must be non-empty and of matching size");
    }
    double sum = 0.0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        double diff = static_cast<double>(y_true[i]) - static_cast<double>(y_hat[i]);
        sum += diff * diff;
    }
    return sum / static_cast<double>(y_true.size());
}

/**
 * @brief Computes the Coefficient of Determination (\(R^2\) score).
 *
 * \f[
 * R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}
 * \f]
 *
 * @param y_true True continuous target values.
 * @param y_hat Predicted continuous target values.
 * @return double \(R^2\) score in \((-\infty, 1.0]\) (1.0 indicates perfect fit).
 * @throws std::invalid_argument If inputs are empty or have mismatched sizes.
 */
inline double r2_score(const std::vector<double>& y_true, const std::vector<double>& y_hat) {
    if (y_true.size() != y_hat.size() || y_true.empty()) {
        throw std::invalid_argument("Input vectors must be non-empty and of matching size");
    }
    const size_t n = y_true.size();
    double sum_true = 0.0;
    for (double val : y_true) {
        sum_true += val;
    }
    const double mean_true = sum_true / static_cast<double>(n);

    double ss_res = 0.0;
    double ss_tot = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double res = y_true[i] - y_hat[i];
        ss_res += res * res;
        double tot = y_true[i] - mean_true;
        ss_tot += tot * tot;
    }

    if (ss_tot == 0.0) {
        return (ss_res == 0.0) ? 1.0 : 0.0;
    }

    return 1.0 - (ss_res / ss_tot);
}

} // namespace morphosml