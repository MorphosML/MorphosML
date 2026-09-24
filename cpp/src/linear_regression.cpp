#include "morphosml/linear_regression.hpp"
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

    // Gradient Descent optimization
    for (size_t epoch = 0; epoch < epochs_; ++epoch) {
        // Forward pass: compute predictions and errors
        double sum_error = 0.0;
        for (size_t i = 0; i < m; ++i) {
            const double* x_row = X.row_ptr(i);
            double y_pred = bias_;
            for (size_t j = 0; j < n; ++j) {
                y_pred += x_row[j] * weights_[j];
            }
            double err = y_pred - y[i];
            errors[i] = err;
            sum_error += err;
        }

        // Backward pass: compute gradients
        const double inv_m = 1.0 / static_cast<double>(m);
        for (size_t j = 0; j < n; ++j) {
            double dw_j = 0.0;
            for (size_t i = 0; i < m; ++i) {
                dw_j += errors[i] * X(i, j);
            }
            weights_[j] -= learning_rate_ * (dw_j * inv_m);
        }

        if (fit_intercept_) {
            bias_ -= learning_rate_ * (sum_error * inv_m);
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

    for (size_t i = 0; i < m; ++i) {
        const double* x_row = X.row_ptr(i);
        double val = bias_;
        for (size_t j = 0; j < n; ++j) {
            val += x_row[j] * weights_[j];
        }
        predictions[i] = val;
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
