#pragma once

#include "matrix.hpp"
#include <vector>

namespace morphosml {

/**
 * @brief Abstract base interface for discrete supervised learning models in MorphosML.
 */
class BaseModel {
public:
    virtual ~BaseModel() = default;

    /**
     * @brief Fits model to training features and integer class labels.
     * @param X Training feature matrix of shape (N, D).
     * @param y Ground truth labels vector of length N.
     */
    virtual void fit(const Matrix& X, const std::vector<int>& y) = 0;

    /**
     * @brief Predicts discrete class labels for query feature matrix.
     * @param X Evaluation feature matrix.
     * @return std::vector<int> Predicted class labels vector.
     */
    virtual std::vector<int> predict(const Matrix& X) = 0;
};

} // namespace morphosml
