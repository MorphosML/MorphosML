#pragma once

#include "model.hpp"
#include "matrix.hpp"
#include "vector.hpp"
#include "metrics.hpp"
#include <vector>

namespace morphosml {

/**
 * @brief Binary Logistic Regression trained via Gradient Descent with Cross-Entropy Loss.
 *
 * @details
 * Models the posterior probability \(\hat{y} = P(y = 1 \mid \mathbf{x})\) using the logistic sigmoid function:
 * \f[
 * z = \mathbf{w}^T \mathbf{x} + b, \quad \sigma(z) = \frac{1}{1 + e^{-z}}
 * \f]
 *
 * Minimizes the binary cross-entropy (log-loss) objective:
 * \f[
 * \mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]
 * \f]
 */
class LogisticRegression : public Classifier {
private:
    Vector weights_;        ///< Learned weight coefficients vector.
    double bias_ = 0.0;     ///< Learned bias intercept scalar.
    double learning_rate_;  ///< Gradient descent step size \(\alpha\).
    size_t epochs_;         ///< Maximum optimization iterations.
    bool fit_intercept_;    ///< Whether to include intercept \(b\).
    bool is_fitted_ = false;///< Flag indicating if `fit()` has executed.

public:
    /**
     * @brief Computes numerically stable standard logistic sigmoid function \(\sigma(z)\).
     *
     * @param z Real-valued scalar input.
     * @return double Probability value in \([0, 1]\).
     */
    static double sigmoid(double z) noexcept;

    /**
     * @brief Constructs a LogisticRegression model with hyperparameters.
     *
     * @param learning_rate Gradient descent learning rate \(\alpha\) (default: 0.01).
     * @param epochs Maximum number of training iterations (default: 1000).
     * @param fit_intercept Whether to calculate intercept \(b\) (default: true).
     */
    explicit LogisticRegression(double learning_rate = 0.01, size_t epochs = 1000, bool fit_intercept = true);

    /**
     * @brief Fits the logistic regression classifier to binary training data \((X, \mathbf{y})\).
     *
     * @param X Training feature matrix of shape \((N, D)\).
     * @param y Binary label vector containing values in \(\{0, 1\}\).
     * @throws std::invalid_argument If dimensions do not match, inputs are empty, or labels \(\notin \{0, 1\}\).
     */
    void fit(const Matrix& X, const std::vector<int>& y) override;

    /**
     * @brief Predicts discrete binary class labels \(\{0, 1\}\) using default threshold 0.5.
     *
     * @param X Feature matrix of shape \((M, D)\).
     * @return std::vector<int> Predicted class labels in \(\{0, 1\}\).
     * @throws std::runtime_error If the model has not been fitted.
     */
    std::vector<int> predict(const Matrix& X) const override {
        return predict(X, 0.5);
    }

    /**
     * @brief Predicts discrete binary class labels \(\{0, 1\}\) given a decision threshold.
     *
     * @param X Feature matrix of shape \((M, D)\).
     * @param threshold Decision threshold (default: 0.5).
     * @return std::vector<int> Predicted class labels in \(\{0, 1\}\).
     * @throws std::runtime_error If the model has not been fitted.
     */
    std::vector<int> predict(const Matrix& X, double threshold) const;

    /**
     * @brief Predicts class-1 probability estimates \(\hat{y} \in [0, 1]\).
     *
     * @param X Feature matrix of shape \((M, D)\).
     * @return std::vector<double> Probability vector of length \(M\).
     * @throws std::runtime_error If the model has not been fitted.
     */
    std::vector<double> predict_proba(const Matrix& X) const;

    /**
     * @brief Calculates classification accuracy on evaluation data.
     *
     * @param X Evaluation feature matrix.
     * @param y Ground truth binary labels.
     * @return double Accuracy score in \([0.0, 1.0]\).
     */
    double score(const Matrix& X, const std::vector<int>& y) const;

    /**
     * @brief Computes binary cross-entropy loss on query data.
     *
     * @param X Evaluation feature matrix.
     * @param y Ground truth binary labels.
     * @return double Binary cross-entropy loss value.
     */
    double compute_loss(const Matrix& X, const std::vector<int>& y) const;

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
    bool is_fitted() const noexcept override { return is_fitted_; }
};

} // namespace morphosml
