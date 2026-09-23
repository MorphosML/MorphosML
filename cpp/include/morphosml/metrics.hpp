#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace morphosml {

// Classification Accuracy
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

// Mean Squared Error (double precision for regression)
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

// Backward-compatible integer MSE
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

// R² Score (Coefficient of Determination)
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