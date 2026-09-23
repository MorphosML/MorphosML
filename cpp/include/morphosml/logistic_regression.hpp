#pragma once

#include "matrix.hpp"
#include "vector.hpp"
#include "metrics.hpp"
#include <vector>

namespace morphosml {

class LogisticRegression {
private:
    Vector weights_;
    double bias_ = 0.0;
    double learning_rate_;
    size_t epochs_;
    bool fit_intercept_;
    bool is_fitted_ = false;

public:
    static double sigmoid(double z) noexcept;

    explicit LogisticRegression(double learning_rate = 0.01, size_t epochs = 1000, bool fit_intercept = true);

    void fit(const Matrix& X, const std::vector<int>& y);
    std::vector<int> predict(const Matrix& X, double threshold = 0.5) const;
    std::vector<double> predict_proba(const Matrix& X) const;
    double score(const Matrix& X, const std::vector<int>& y) const;
    double compute_loss(const Matrix& X, const std::vector<int>& y) const;

    const Vector& weights() const { return weights_; }
    double bias() const noexcept { return bias_; }
    double learning_rate() const noexcept { return learning_rate_; }
    size_t epochs() const noexcept { return epochs_; }
    bool is_fitted() const noexcept { return is_fitted_; }
};

} // namespace morphosml

