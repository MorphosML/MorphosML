#pragma once

#include "matrix.hpp"
#include <vector>

namespace morphosml {

/**
 * @brief Common base interface for all models in MorphosML.
 */
class BaseModel {
public:
    virtual ~BaseModel() = default;

    /**
     * @brief Checks if the model has been fitted to data.
     */
    virtual bool is_fitted() const noexcept = 0;
};

/**
 * @brief Abstract base class for discrete supervised classification models.
 */
class Classifier : public BaseModel {
public:
    virtual ~Classifier() = default;

    /**
     * @brief Fits model to training features and integer class labels.
     */
    virtual void fit(const Matrix& X, const std::vector<int>& y) = 0;

    /**
     * @brief Predicts discrete class labels for query feature matrix.
     */
    virtual std::vector<int> predict(const Matrix& X) const = 0;
};

/**
 * @brief Abstract base class for continuous supervised regression models.
 */
class Regressor : public BaseModel {
public:
    virtual ~Regressor() = default;

    /**
     * @brief Fits model to training features and continuous targets.
     */
    virtual void fit(const Matrix& X, const std::vector<double>& y) = 0;

    /**
     * @brief Predicts continuous targets for query feature matrix.
     */
    virtual std::vector<double> predict(const Matrix& X) const = 0;
};

} // namespace morphosml
