#include "morphosml/linear_regression.hpp"
#include "morphosml/simd/simd_ops.hpp"
#include <stdexcept>
#include <cmath>

namespace morphosml {

LinearRegression::LinearRegression(double learning_rate, size_t epochs, bool fit_intercept)
    : learning_rate_(learning_rate), epochs_(epochs), fit_intercept_(fit_intercept) {
    if (learning_rate_ <= 0.0) {
        throw std::invalid_argument("Learning rate must be strictly positive");
    }
}

void LinearRegression::fit(const Matrix& X, const std::vector<double>& y) {
    const size_t m = X.rows();
    const size_t n = X.cols();

    if (m == 0 || n == 0) {
        throw std::invalid_argument("Feature matrix X cannot be empty");
    }
    if (m != y.size()) {
        throw std::invalid_argument("Number of rows in X must match length of target vector y");
    }

    // Initialize weights to 0.0
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
        // Forward pass: compute predictions and errors
#if defined(MORPHOSML_HAS_OPENMP)
        #pragma omp parallel for schedule(static) if(m > 256)
#endif
        for (size_t i = 0; i < m; ++i) {
            double y_pred = bias_ + simd::dot(X.row_ptr(i), w_ptr, n);
            errors[i] = y_pred - y[i];
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

std::vector<double> LinearRegression::predict(const Matrix& X) const {
    if (!is_fitted_) {
        throw std::runtime_error("LinearRegression model is not fitted yet");
    }
    if (X.cols() != weights_.size()) {
        throw std::invalid_argument("Number of columns in X must match number of fitted features");
    }

    const size_t m = X.rows();
    const size_t n = X.cols();
    std::vector<double> predictions(m);
    const double* w_ptr = weights_.data().data();

#if defined(MORPHOSML_HAS_OPENMP)
    #pragma omp parallel for schedule(static) if(m > 256)
#endif
    for (size_t i = 0; i < m; ++i) {
        predictions[i] = bias_ + simd::dot(X.row_ptr(i), w_ptr, n);
    }

    return predictions;
}

double LinearRegression::score(const Matrix& X, const std::vector<double>& y) const {
    auto preds = predict(X);
    return r2_score(y, preds);
}

double LinearRegression::compute_cost(const Matrix& X, const std::vector<double>& y) const {
    auto preds = predict(X);
    return mse(y, preds);
}

} // namespace morphosml
