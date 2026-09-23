#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cstddef>
#include <utility>

namespace morphosml {

/**
 * @brief Dense 1D mathematical vector for linear algebra, gradient updates, and feature weights.
 *
 * @details
 * `Vector` represents a 1D sequence of double-precision floating-point numbers equipped with
 * Euclidean vector space operations: inner (dot) product, \f$L_2\f$ norm, unit normalization,
 * element-wise addition/subtraction, and scalar multiplication.
 */
class Vector {
private:
    std::vector<double> data_; ///< Internal contiguous element storage.

public:
    /**
     * @brief Constructs an empty 0-dimensional Vector.
     */
    Vector() = default;

    // Rule of 5: Default copy and move semantics
    Vector(const Vector&) = default;
    Vector& operator=(const Vector&) = default;
    Vector(Vector&&) noexcept = default;
    Vector& operator=(Vector&&) noexcept = default;

    /**
     * @brief Constructs a Vector of length `n` initialized to `0.0`.
     * @param n Dimension of the vector.
     */
    explicit Vector(size_t n) : data_(n, 0.0) {}

    /**
     * @brief Constructs a Vector from an initializer list of doubles.
     * @param list Initializer list (e.g., `{1.0, 2.0, 3.0}`).
     */
    Vector(std::initializer_list<double> list) : data_(list) {}

    /**
     * @brief Constructs a Vector by copying an existing `std::vector<double>`.
     * @param vec Source vector.
     */
    Vector(const std::vector<double>& vec) : data_(vec) {}

    /**
     * @brief Constructs a Vector by moving an existing `std::vector<double>`.
     * @param vec Source vector to move from (avoids heap allocation).
     */
    Vector(std::vector<double>&& vec) noexcept : data_(std::move(vec)) {}

    /**
     * @brief Returns the number of dimensions/elements in the vector.
     * @return size_t Element count.
     */
    size_t size() const noexcept { return data_.size(); }

    /**
     * @brief Checks if the vector is empty.
     * @return bool True if empty, false otherwise.
     */
    bool empty() const noexcept { return data_.empty(); }

    /**
     * @brief Fast, unchecked mutable element access.
     * @param i 0-based element index.
     * @return double& Reference to element `i`.
     */
    double& operator[](size_t i) noexcept { return data_[i]; }

    /**
     * @brief Fast, unchecked const element access.
     * @param i 0-based element index.
     * @return const double& Const reference to element `i`.
     */
    const double& operator[](size_t i) const noexcept { return data_[i]; }

    /**
     * @brief Safe mutable element access with bounds checking.
     * @param i 0-based element index.
     * @return double& Reference to element `i`.
     * @throws std::out_of_range If `i >= size()`.
     */
    double& at(size_t i) {
        if (i >= data_.size()) throw std::out_of_range("Vector index out of range");
        return data_[i];
    }

    /**
     * @brief Safe const element access with bounds checking.
     * @param i 0-based element index.
     * @return const double& Const reference to element `i`.
     * @throws std::out_of_range If `i >= size()`.
     */
    const double& at(size_t i) const {
        if (i >= data_.size()) throw std::out_of_range("Vector index out of range");
        return data_[i];
    }

    /**
     * @brief Element-wise addition \f$\mathbf{u} + \mathbf{v}\f$.
     * @param other Vector of identical size.
     * @return Vector Resulting sum vector.
     * @throws std::invalid_argument If dimensions do not match.
     */
    Vector operator+(const Vector& other) const;

    /**
     * @brief Element-wise subtraction \f$\mathbf{u} - \mathbf{v}\f$.
     * @param other Vector of identical size.
     * @return Vector Resulting difference vector.
     * @throws std::invalid_argument If dimensions do not match.
     */
    Vector operator-(const Vector& other) const;

    /**
     * @brief Scalar multiplication \f$\alpha \mathbf{v}\f$.
     * @param scalar Floating-point multiplier.
     * @return Vector Scaled vector.
     */
    Vector operator*(double scalar) const;

    /**
     * @brief Computes the Euclidean inner (dot) product \f$\mathbf{u} \cdot \mathbf{v} = \sum_{i} u_i v_i\f$.
     * @param other Vector of identical size.
     * @return double Scalar inner product.
     * @throws std::invalid_argument If dimensions do not match.
     */
    double dot(const Vector& other) const;

    /**
     * @brief Computes the Euclidean \f$L_2\f$ norm \f$\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}\f$.
     * @return double Vector length/magnitude.
     */
    double norm() const;

    /**
     * @brief Returns a unit-length normalized vector \f$\frac{\mathbf{v}}{\|\mathbf{v}\|_2}\f$.
     * @return Vector Unit vector with \f$\|\hat{\mathbf{v}}\|_2 = 1.0\f$.
     * @throws std::runtime_error If the norm is zero.
     */
    Vector normalized() const;

    /**
     * @brief Prints the vector components to standard output.
     */
    void print() const;

    /**
     * @brief Returns a const reference to the underlying `std::vector<double>`.
     * @return const std::vector<double>& Internal storage buffer.
     */
    const std::vector<double>& data() const noexcept { return data_; }

    /**
     * @brief Checks element-wise equality between two vectors.
     */
    bool operator==(const Vector& other) const noexcept {
        return data_ == other.data_;
    }

    /**
     * @brief Checks inequality between two vectors.
     */
    bool operator!=(const Vector& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Exports the internal data as a `std::vector<double>`.
     * @return std::vector<double> Vector data copy.
     */
    std::vector<double> to_std() const { return data_; }
};

} // namespace morphosml