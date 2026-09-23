#pragma once

#include <vector>
#include <stdexcept>
#include "vector.hpp"
#include <random>
#include <iostream>
#include <cstddef>
#include <cstring>

namespace morphosml {

/**
 * @brief Sets the global deterministic random seed used across MorphosML C++ routines.
 *
 * @param seed Unsigned 64-bit integer seed value.
 */
void set_seed(uint64_t seed);

/**
 * @brief Retrieves the current global deterministic random seed value.
 *
 * @return uint64_t Current seed value.
 */
uint64_t get_seed();

/**
 * @brief High-Performance Computing (HPC) 2D dense matrix with contiguous 1D memory layout.
 *
 * @details
 * `Matrix` is the fundamental dense linear algebra primitive in MorphosML. Unlike standard
 * `std::vector<std::vector<double>>` representations (which cause pointer chasing, cache line
 * invalidations, and allocator overhead), `morphosml::Matrix` stores all elements in a single
 * contiguous 1D row-major buffer:
 *
 * \f[
 * \text{Index}(i, j) = i \times \text{cols} + j
 * \f]
 *
 * ### High-Performance Features:
 * - **L1/L2 Cache Locality**: Sequential row elements are adjacent in physical memory, maximizing
 *   hardware prefetcher hit rates and SIMD throughput (AVX2/AVX-512/NEON).
 * - **Optimized Multiplication**: Matrix multiplication employs an \f$(i, k, j)\f$ loop ordering
 *   to ensure inner-loop access patterns are stride-1 across memory (`result[i, j] += A[i, k] * B[k, j]`).
 * - **Zero-Copy Python Buffer Protocol**: Seamlessly exchanges memory buffers with NumPy, PyTorch,
 *   and Polars without intermediate heap allocations.
 * - **Rule of 5**: Move constructors and move assignment operators transfer ownership of the
 *   underlying flat buffer in \f$O(1)\f$ time without copying.
 */
class Matrix {
private:
    std::vector<double> data_; ///< Contiguous 1D flat row-major buffer of size (n_rows * n_cols).
    size_t n_rows;             ///< Number of rows.
    size_t n_cols;             ///< Number of columns.

public:
    /**
     * @brief Constructs an empty 0x0 Matrix.
     */
    Matrix() : n_rows(0), n_cols(0) {}

    /**
     * @brief Constructs a Matrix of dimensions (rows, cols) initialized to a constant value.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param init_val Initial value for all elements (default: 0.0).
     */
    Matrix(size_t rows, size_t cols, double init_val = 0.0);

    /**
     * @brief Constructs a Matrix from nested initializer lists.
     *
     * @param list Nested initializer list of double values (e.g., `{{1.0, 2.0}, {3.0, 4.0}}`).
     */
    Matrix(std::initializer_list<std::initializer_list<double>> list);

    /**
     * @brief Constructs a Matrix from a 2D vector of vectors.
     *
     * @param data 2D vector to convert into a contiguous flat buffer.
     */
    Matrix(const std::vector<std::vector<double>>& data);

    /**
     * @brief Constructs a Matrix directly from dimensions and an existing flat 1D buffer.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param flat_data 1D vector containing (rows * cols) elements in row-major order.
     * @throws std::invalid_argument If `flat_data.size() != rows * cols`.
     */
    Matrix(size_t rows, size_t cols, std::vector<double> flat_data);

    // Rule of 5: Default copy and move semantics
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    /**
     * @brief Returns the number of rows.
     * @return size_t Row count.
     */
    size_t rows() const noexcept { return n_rows; }

    /**
     * @brief Returns the number of columns.
     * @return size_t Column count.
     */
    size_t cols() const noexcept { return n_cols; }

    /**
     * @brief Returns the total number of elements (\f$\text{rows} \times \text{cols}\f$).
     * @return size_t Total element count.
     */
    size_t size() const noexcept { return n_rows * n_cols; }

    /**
     * @brief Checks if the matrix contains 0 elements.
     * @return bool True if empty, false otherwise.
     */
    bool empty() const noexcept { return data_.empty(); }

    /**
     * @brief Mutable element access with bounds checking.
     *
     * @param i 0-based row index.
     * @param j 0-based column index.
     * @return double& Reference to element at coordinate (i, j).
     * @throws std::out_of_range If i >= rows or j >= cols.
     */
    double& operator()(size_t i, size_t j);

    /**
     * @brief Const element access with bounds checking.
     *
     * @param i 0-based row index.
     * @param j 0-based column index.
     * @return const double& Const reference to element at coordinate (i, j).
     * @throws std::out_of_range If i >= rows or j >= cols.
     */
    const double& operator()(size_t i, size_t j) const;

    /**
     * @brief Returns a mutable pointer to the beginning of the contiguous buffer.
     * @return double* Raw data pointer.
     */
    double* data() noexcept { return data_.data(); }

    /**
     * @brief Returns a const pointer to the beginning of the contiguous buffer.
     * @return const double* Raw data pointer.
     */
    const double* data() const noexcept { return data_.data(); }

    /**
     * @brief Returns a mutable pointer to the start of row `i`.
     *
     * @param i 0-based row index.
     * @return double* Pointer to element `(i, 0)`.
     */
    double* row_ptr(size_t i) noexcept { return data_.data() + i * n_cols; }

    /**
     * @brief Returns a const pointer to the start of row `i`.
     *
     * @param i 0-based row index.
     * @return const double* Pointer to element `(i, 0)`.
     */
    const double* row_ptr(size_t i) const noexcept { return data_.data() + i * n_cols; }

    /**
     * @brief Extracts row `i` as a MorphosML `Vector`.
     *
     * @param i 0-based row index.
     * @return Vector Owning vector copy of row `i`.
     * @throws std::out_of_range If i >= rows.
     */
    Vector row(size_t i) const;

    /**
     * @brief Computes the transpose matrix \f$M^T\f$.
     * @return Matrix Transposed matrix of shape (cols, rows).
     */
    Matrix transpose() const;

    /**
     * @brief Element-wise matrix addition \f$A + B\f$.
     * @param other Matrix with identical dimensions.
     * @return Matrix Element-wise sum.
     * @throws std::invalid_argument If dimensions do not match.
     */
    Matrix operator+(const Matrix& other) const;

    /**
     * @brief Element-wise matrix subtraction \f$A - B\f$.
     * @param other Matrix with identical dimensions.
     * @return Matrix Element-wise difference.
     * @throws std::invalid_argument If dimensions do not match.
     */
    Matrix operator-(const Matrix& other) const;

    /**
     * @brief Cache-optimized matrix multiplication \f$A \times B\f$.
     *
     * @details
     * Implemented using an \f$(i, k, j)\f$ loop structure to ensure sequential, contiguous memory
     * reads from both operands and sequential writes into the destination row, eliminating CPU cache misses.
     *
     * @param other Matrix with dimensions (this->cols(), K).
     * @return Matrix Result matrix with dimensions (this->rows(), K).
     * @throws std::invalid_argument If this->cols() != other.rows().
     */
    Matrix operator*(const Matrix& other) const;

    /**
     * @brief Scalar multiplication \f$A \times \alpha\f$.
     * @param scalar Floating-point scalar multiplier.
     * @return Matrix Scaled matrix.
     */
    Matrix operator*(double scalar) const;

    /**
     * @brief Matrix-vector product \f$A \mathbf{v}\f$.
     * @param vec Vector with dimension equal to this->cols().
     * @return Vector Result vector with dimension equal to this->rows().
     * @throws std::invalid_argument If dimensions do not match.
     */
    Vector operator*(const Vector& vec) const;

    /**
     * @brief Creates an \f$n \times n\f$ identity matrix \f$I_n\f$.
     * @param n Size of the square identity matrix.
     * @return Matrix Identity matrix with 1.0 along the main diagonal.
     */
    static Matrix identity(size_t n);

    /**
     * @brief Creates an \f$r \times c\f$ matrix populated entirely with zeros.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return Matrix Zero-initialized matrix.
     */
    static Matrix zeros(size_t rows, size_t cols);

    /**
     * @brief Creates an \f$r \times c\f$ matrix initialized with uniform random values in \([0, 1)\).
     * Uses the global deterministic seed.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @return Matrix Uniformly distributed random matrix.
     */
    static Matrix random(size_t rows, size_t cols);

    /**
     * @brief Creates an \f$r \times c\f$ matrix initialized with uniform random values using a custom seed.
     *
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param seed Custom 64-bit seed.
     * @return Matrix Uniformly distributed random matrix.
     */
    static Matrix random(size_t rows, size_t cols, uint64_t seed);

    /**
     * @brief Prints matrix contents to standard output in tabular format.
     */
    void print() const;

    /**
     * @brief Returns a const reference to the underlying contiguous `std::vector<double>`.
     * @return const std::vector<double>& Contiguous flat buffer.
     */
    const std::vector<double>& raw_data() const noexcept { return data_; }

    /**
     * @brief Converts the contiguous matrix into a nested `std::vector<std::vector<double>>`.
     * @return std::vector<std::vector<double>> Nested 2D representation.
     */
    std::vector<std::vector<double>> to_std() const;
};

} // namespace morphosml
