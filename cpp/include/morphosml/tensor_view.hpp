#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include "morphosml/matrix.hpp"

namespace morphosml {

/**
 * @brief Non-owning, zero-copy 2D tensor view over contiguous or strided float64 memory.
 *
 * @details
 * `TensorView` provides high-performance, zero-overhead 2D indexing semantics over an
 * externally owned buffer of `double` elements. It is engineered specifically for High-Performance
 * Computing (HPC) workflows where copying large arrays across Python/C++ boundaries or between
 * memory-mapped disk buffers is unacceptable.
 *
 * ### Key Architectural Properties:
 * - **Zero-Copy & Zero-Allocation**: The view does not allocate or deallocate memory on the heap.
 *   It points directly to existing buffers (e.g., NumPy `ndarray` via Python buffer protocol,
 *   POSIX `mmap` virtual memory addresses, or raw C arrays).
 * - **Strided Navigation**: Supports row-major striding (`stride`), allowing zero-copy sub-matrix
 *   slices, batch windows, or padded buffers without requiring memory repacking.
 * - **Sub-Nanosecond Access**: The call operator `operator()(size_t r, size_t c)` computes
 *   `r * stride + c` directly inline, enabling aggressive compiler auto-vectorization and loop pipelining.
 * - **Safe Lifetime Contract**: The caller must ensure that the backing memory buffer remains valid
 *   and unmodified for the entire duration of the `TensorView` lifetime. If owning storage is
 *   required, use `to_matrix()`.
 *
 * ### Example Usage (C++):
 * @code
 * const double raw_buffer[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
 * morphosml::TensorView view(raw_buffer, 2, 3);
 * double val = view(1, 2); // 6.0
 * morphosml::Matrix mat = view.to_matrix(); // Explicit copy to owning matrix
 * @endcode
 */
class TensorView {
private:
    const double* data_ = nullptr; ///< Non-owning pointer to the start of contiguous memory.
    size_t rows_ = 0;              ///< Number of rows in the 2D view.
    size_t cols_ = 0;              ///< Number of columns in each logical row.
    size_t stride_ = 0;            ///< Step size (number of elements) between successive row starts.

public:
    /**
     * @brief Constructs an empty, null `TensorView`.
     */
    TensorView() = default;

    /**
     * @brief Constructs a `TensorView` with a custom row stride.
     *
     * @param data Non-owning pointer to the external contiguous float64 memory buffer.
     * @param rows Total number of rows in the logical view.
     * @param cols Total number of columns per row.
     * @param stride Stride in number of elements (must be >= `cols` for valid row-major layouts).
     */
    TensorView(const double* data, size_t rows, size_t cols, size_t stride)
        : data_(data), rows_(rows), cols_(cols), stride_(stride) {}

    /**
     * @brief Constructs a dense contiguous `TensorView` where `stride == cols`.
     *
     * @param data Non-owning pointer to the external contiguous float64 memory buffer.
     * @param rows Total number of rows in the view.
     * @param cols Total number of columns in the view.
     */
    TensorView(const double* data, size_t rows, size_t cols)
        : data_(data), rows_(rows), cols_(cols), stride_(cols) {}

    /**
     * @brief Constructs a `TensorView` directly wrapping an existing `Matrix`.
     *
     * @param mat Matrix to view without copying.
     */
    TensorView(const Matrix& mat)
        : data_(mat.data()), rows_(mat.rows()), cols_(mat.cols()), stride_(mat.cols()) {}

    /**
     * @brief Accesses an element at index (r, c) with bounds checking.
     *
     * @param r 0-based row index.
     * @param c 0-based column index.
     * @return double Value at coordinate `(r, c)`.
     * @throws std::out_of_range If `r >= rows` or `c >= cols`.
     * @note Performance: Inlined with zero virtual dispatch.
     */
    inline double operator()(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("TensorView index out of bounds");
        }
        return data_[r * stride_ + c];
    }

    /**
     * @brief Returns a direct pointer to the start of a specified row.
     *
     * @param r 0-based row index.
     * @return const double* Pointer to the first element of row `r`.
     * @throws std::out_of_range If `r >= rows`.
     */
    const double* row_ptr(size_t r) const {
        if (r >= rows_) {
            throw std::out_of_range("TensorView row out of bounds");
        }
        return data_ + (r * stride_);
    }

    /**
     * @brief Returns the underlying raw pointer to the memory buffer.
     * @return const double* Pointer to the base memory address.
     */
    const double* data() const noexcept { return data_; }

    /**
     * @brief Returns the number of rows.
     * @return size_t Row count.
     */
    size_t rows() const noexcept { return rows_; }

    /**
     * @brief Returns the number of columns.
     * @return size_t Column count.
     */
    size_t cols() const noexcept { return cols_; }

    /**
     * @brief Returns the row stride (in number of elements).
     * @return size_t Stride count.
     */
    size_t stride() const noexcept { return stride_; }

    /**
     * @brief Returns the total number of logical elements (`rows * cols`).
     * @return size_t Logical element count.
     */
    size_t size() const noexcept { return rows_ * cols_; }

    /**
     * @brief Checks if the view is empty (0 rows or 0 columns).
     * @return bool True if empty, false otherwise.
     */
    bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }

    /**
     * @brief Materializes and copies the view into an owning contiguous MorphosML `Matrix`.
     *
     * @details
     * Copies elements row-by-row respecting `stride_`. Use this method whenever data
     * must persist beyond the lifetime of the underlying NumPy/mmap buffer, or when
     * in-place matrix mutation is needed.
     *
     * @return Matrix Owning 2D matrix containing an identical copy of the view's data.
     */
    Matrix to_matrix() const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            const double* src = row_ptr(i);
            double* dst = result.row_ptr(i);
            std::memcpy(dst, src, cols_ * sizeof(double));
        }
        return result;
    }

    /**
     * @brief Returns a zero-copy row slice spanning `num_rows` starting at `start_row`.
     */
    TensorView slice(size_t start_row, size_t num_rows) const {
        if (start_row + num_rows > rows_) {
            throw std::out_of_range("TensorView slice exceeds row bounds");
        }
        return TensorView(data_ + start_row * stride_, num_rows, cols_, stride_);
    }

    /**
     * @brief Returns a zero-copy 2D subview spanning `num_rows` and `num_cols`.
     */
    TensorView subview(size_t start_row, size_t num_rows, size_t start_col, size_t num_cols) const {
        if (start_row + num_rows > rows_ || start_col + num_cols > cols_) {
            throw std::out_of_range("TensorView subview exceeds bounds");
        }
        return TensorView(data_ + start_row * stride_ + start_col, num_rows, num_cols, stride_);
    }

    /**
     * @brief Extracts row `r` as an owning MorphosML Vector.
     */
    Vector row(size_t r) const {
        if (r >= rows_) {
            throw std::out_of_range("TensorView row index out of bounds");
        }
        std::vector<double> row_data(cols_);
        std::memcpy(row_data.data(), row_ptr(r), cols_ * sizeof(double));
        return Vector(std::move(row_data));
    }

    /**
     * @brief Checks element-wise equality between two tensor views.
     */
    bool operator==(const TensorView& other) const noexcept {
        if (rows_ != other.rows_ || cols_ != other.cols_) return false;
        for (size_t r = 0; r < rows_; ++r) {
            for (size_t c = 0; c < cols_; ++c) {
                if ((*this)(r, c) != other(r, c)) return false;
            }
        }
        return true;
    }

    /**
     * @brief Checks inequality between two tensor views.
     */
    bool operator!=(const TensorView& other) const noexcept {
        return !(*this == other);
    }
};

} // namespace morphosml
