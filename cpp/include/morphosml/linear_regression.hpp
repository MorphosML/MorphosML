#pragma once

#include "matrix.hpp"
#include "vector.hpp"
#include "metrics.hpp"
#include <vector>

namespace morphosml {

/**
 * @brief Multivariate Linear Regression trained via Batch Gradient Descent.
 *
 * @details
 * Models continuous target variables \(\hat{y}\) as a linear combination of input features \(\mathbf{x}\):
 * \f[
 * \hat{y} = \mathbf{w}^T \mathbf{x} + b
 * \f]
 *
 * Optimization minimizes the Mean Squared Error (MSE) objective:
 * \f[
 * J(\mathbf{w}, b) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})^2
 * \f]
 * Analytical parameter updates are executed sequentially over `epochs`:
 * \f[
 * \mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{1}{N} X^T (\hat{\mathbf{y}} - \mathbf{y}), \quad
 * b \leftarrow b - \alpha \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})
 * \f]
 */
class LinearRegression {
private:
    Vector weights_;        ///< Learned weight coefficients vector.
    double bias_ = 0.0;     ///< Learned intercept (bias) scalar.
    double learning_rate_;  ///< Gradient descent step size \(\alpha\).
    size_t epochs_;         ///< Maximum optimization iterations.
    bool fit_intercept_;    ///< Whether to fit an intercept term \(b\).
    bool is_fitted_ = false;///< Flag indicating if `fit()` has executed.

public:
    /**
     * @brief Constructs a LinearRegression model with hyperparameters.
     *
     * @param learning_rate Gradient descent learning rate \(\alpha\) (default: 0.01).
     * @param epochs Maximum number of training epochs (default: 1000).
     * @param fit_intercept Whether to calculate intercept \(b\) (default: true).
     */
    explicit LinearRegression(double learning_rate = 0.01, size_t epochs = 1000, bool fit_intercept = true);

    /**
     * @brief Fits the linear model to the training dataset \((X, \mathbf{y})\).
     *
     * @param X Training feature matrix of shape \((N, D)\).
     * @param y Target continuous vector of dimension \(N\).
     * @throws std::invalid_argument If \(X.\text{rows}() \neq y.\text{size}()\) or inputs are empty.
     */
    void fit(const Matrix& X, const std::vector<double>& y);

    /**
     * @brief Generates continuous target predictions for query matrix \(X\).
     *
     * @param X Feature matrix of shape \((M, D)\).
     * @return std::vector<double> Predicted continuous values vector of dimension \(M\).
     * @throws std::runtime_error If the model has not been fitted.
     * @throws std::invalid_argument If feature dimension \(D\) does not match `weights.size()`.
     */
    std::vector<double> predict(const Matrix& X) const;

    /**
     * @brief Computes the Coefficient of Determination (\(R^2\) score) on query data.
     *
     * @param X Evaluation feature matrix.
     * @param y True continuous target values.
     * @return double \(R^2\) score in \((-\infty, 1.0]\).
     */
    double score(const Matrix& X, const std::vector<double>& y) const;

    /**
     * @brief Computes current Mean Squared Error cost \(J(\mathbf{w}, b)\).
     *
     * @param X Feature matrix.
     * @param y Target vector.
     * @return double Computed MSE cost.
     */
    double compute_cost(const Matrix& X, const std::vector<double>& y) const;

    /**
     * @brief Returns learned weight vector \(\mathbf{w}\).
     * @return const Vector& Feature weights.
     */
    const Vector& weights() const { return weights_; }

    /**
     * @brief Returns learned intercept term \(b\).
     * @return double Bias scalar.
     */
    double bias() const noexcept { return bias_; }

    /**
     * @brief Returns learning rate \(\alpha\).
     * @return double Learning rate.
     */
    double learning_rate() const noexcept { return learning_rate_; }

    /**
     * @brief Returns maximum training epochs.
     * @return size_t Epoch count.
     */
    size_t epochs() const noexcept { return epochs_; }

    /**
     * @brief Returns whether the model has been fitted.
     * @return bool True if fitted, false otherwise.
     */
    bool is_fitted() const noexcept { return is_fitted_; }
};

} // namespace morphosml