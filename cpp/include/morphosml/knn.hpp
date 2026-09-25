#pragma once

#include "model.hpp"
#include "matrix.hpp"
#include <vector>
#include <queue>
#include <unordered_map>

namespace morphosml {

/**
 * @brief k-Nearest Neighbors (k-NN) classification model.
 *
 * @details
 * Non-parametric lazy learning classifier that predicts class membership for query samples
 * by identifying the \(k\) nearest training exemplars under Euclidean distance:
 * \f[
 * d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{j=1}^D (x_j - z_j)^2}
 * \f]
 * Ties in majority voting are resolved deterministically based on neighbor proximity.
 */
class KNN : public Classifier {
private:
    Matrix X_train_;             ///< Stored training feature matrix.
    std::vector<int> y_train_;   ///< Stored training class labels.
    int k_;                      ///< Number of nearest neighbors to consider.

    /**
     * @brief Internal helper struct storing neighbor distance and dataset index.
     */
    struct DistanceIndex {
        double distance;
        int index;
        bool operator<(const DistanceIndex& other) const {
            return distance < other.distance;
        }
    };

public:
    /**
     * @brief Constructs a KNN classifier with parameter \(k\).
     * @param k Number of nearest neighbors (default: 3).
     */
    explicit KNN(int k = 3) : k_(k) {}

    /**
     * @brief Memorizes training points and labels.
     *
     * @param X Training feature matrix of shape \((N, D)\).
     * @param y Training label vector of length \(N\).
     */
    void fit(const Matrix& X, const std::vector<int>& y) override;

    /**
     * @brief Predicts discrete class labels for each row in feature matrix \(X\).
     *
     * @param X Feature matrix of shape \((M, D)\).
     * @return std::vector<int> Predicted class labels vector.
     */
    std::vector<int> predict(const Matrix& X) const override;

    /**
     * @brief Checks whether the model has stored training samples.
     * @return bool True if fitted.
     */
    bool is_fitted() const noexcept override { return X_train_.rows() > 0; }

    /**
     * @brief Returns hyperparameter \(k\).
     * @return int Neighborhood size.
     */
    int get_k() const { return k_; }

private:
    /**
     * @brief Predicts label for a single query vector.
     */
    int predict_single(const Vector& x) const;

    /**
     * @brief Predicts label for a contiguous memory buffer without allocations.
     */
    int predict_single(const double* x_ptr, size_t n_features) const;

    /**
     * @brief Computes Euclidean distance between two vectors.
     */
    double euclidean_distance(const Vector& a, const Vector& b) const;
};

} // namespace morphosml