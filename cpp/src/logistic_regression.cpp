#include "morphosml/logistic_regression.hpp"
#include "morphosml/simd/simd_ops.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace morphosml {

double LogisticRegression::sigmoid(double z) noexcept {
    // Numerically stable sigmoid with clamping
    constexpr double CLAMP_LIMIT = 250.0;
    if (z > CLAMP_LIMIT) return 1.0;
    if (z < -CLAMP_LIMIT) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

LogisticRegression::LogisticRegression(double learning_rate, size_t epochs, bool fit_intercept)
    : learning_rate_(learning_rate), epochs_(epochs), fit_intercept_(fit_intercept) {
    if (learning_rate_ <= 0.0) {
        throw std::invalid_argument("Learning rate must be strictly positive");
    }
}

void LogisticRegression::fit(const Matrix& X, const std::vector<int>& y) {
    const size_t m = X.rows();
    const size_t n = X.cols();

    if (m == 0 || n == 0) {
        throw std::invalid_argument("Feature matrix X cannot be empty");
    }
    if (m != y.size()) {
        throw std::invalid_argument("Number of rows in X must match length of target vector y");
    }

    // Verify binary labels {0, 1}
    for (int label : y) {
        if (label != 0 && label != 1) {
            throw std::invalid_argument("LogisticRegression only supports binary labels (0 or 1)");
        }
    }

    weights_ = Vector(n);
    bias_ = 0.0;

    std::vector<double> errors(m);
    const double inv_m = 1.0 / static_cast<double>(m);
    const double step = learning_rate_ * inv_m;
    const double* w_ptr = weights_.data().data();

#if defined(MORPHOSML_HAS_OPENMP)
    int num_threads = (m > 256) ? omp_get_max_threads() : 1;
    std::vector<double> thread_dw(num_threads * n, 0.0);
#endif
    std::vector<double> dw(n, 0.0);

    // Gradient Descent optimization
    for (size_t epoch = 0; epoch < epochs_; ++epoch) {
        // Forward pass
#if defined(MORPHOSML_HAS_OPENMP)
        #pragma omp parallel for schedule(static) if(m > 256)
#endif
        for (size_t i = 0; i < m; ++i) {
            double z = bias_ + simd::dot(X.row_ptr(i), w_ptr, n);
            double p = sigmoid(z);
            errors[i] = p - static_cast<double>(y[i]);
        }

        double sum_error = 0.0;
        for (size_t i = 0; i < m; ++i) {
            sum_error += errors[i];
        }

        // Backward pass: streaming contiguous row access with AVX2 FMA
#if defined(MORPHOSML_HAS_OPENMP)
        std::memset(thread_dw.data(), 0, num_threads * n * sizeof(double));
        std::memset(dw.data(), 0, n * sizeof(double));

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            double* local_dw = thread_dw.data() + tid * n;

            #pragma omp for schedule(static)
            for (size_t i = 0; i < m; ++i) {
                simd::vec_fmadd(X.row_ptr(i), errors[i], local_dw, n);
            }
        }

        for (int t = 0; t < num_threads; ++t) {
            simd::vec_add(dw.data(), thread_dw.data() + t * n, dw.data(), n);
        }
#else
        std::memset(dw.data(), 0, n * sizeof(double));
        for (size_t i = 0; i < m; ++i) {
            simd::vec_fmadd(X.row_ptr(i), errors[i], dw.data(), n);
        }
#endif

        for (size_t j = 0; j < n; ++j) {
            weights_[j] -= step * dw[j];
        }

        if (fit_intercept_) {
            bias_ -= step * sum_error;
        }
    }

    is_fitted_ = true;
}

std::vector<double> LogisticRegression::predict_proba(const Matrix& X) const {
    if (!is_fitted_) {
        throw std::runtime_error("LogisticRegression model is not fitted yet");
    }
    if (X.cols() != weights_.size()) {
        throw std::invalid_argument("Number of columns in X must match number of fitted features");
    }

    const size_t m = X.rows();
    const size_t n = X.cols();
    std::vector<double> probabilities(m);
    const double* w_ptr = weights_.data().data();

#if defined(MORPHOSML_HAS_OPENMP)
    #pragma omp parallel for schedule(static) if(m > 256)
#endif
    for (size_t i = 0; i < m; ++i) {
        double z = bias_ + simd::dot(X.row_ptr(i), w_ptr, n);
        probabilities[i] = sigmoid(z);
    }

    return probabilities;
}

std::vector<int> LogisticRegression::predict(const Matrix& X, double threshold) const {
    auto probs = predict_proba(X);
    std::vector<int> preds(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        preds[i] = (probs[i] >= threshold) ? 1 : 0;
    }
    return preds;
}

double LogisticRegression::score(const Matrix& X, const std::vector<int>& y) const {
    auto preds = predict(X);
    return accuracy(y, preds);
}

double LogisticRegression::compute_loss(const Matrix& X, const std::vector<int>& y) const {
    auto probs = predict_proba(X);
    constexpr double EPS = 1e-15;
    double total_bce = 0.0;
    const size_t m = y.size();

    for (size_t i = 0; i < m; ++i) {
        double p = std::clamp(probs[i], EPS, 1.0 - EPS);
        double target = static_cast<double>(y[i]);
        total_bce += target * std::log(p) + (1.0 - target) * std::log(1.0 - p);
    }

    return -total_bce / static_cast<double>(m);
}

} // namespace morphosml

