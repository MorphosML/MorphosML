#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include "morphosml/matrix.hpp"

namespace morphosml {

class TensorView {
private:
    const double* data_ = nullptr;
    size_t rows_ = 0;
    size_t cols_ = 0;
    size_t stride_ = 0; // Number of elements between consecutive rows

public:
    TensorView() = default;

    TensorView(const double* data, size_t rows, size_t cols, size_t stride)
        : data_(data), rows_(rows), cols_(cols), stride_(stride) {}

    TensorView(const double* data, size_t rows, size_t cols)
        : data_(data), rows_(rows), cols_(cols), stride_(cols) {}

    // Inline element access (sub-nanosecond, cache-friendly)
    inline double operator()(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("TensorView index out of bounds");
        }
        return data_[r * stride_ + c];
    }

    const double* row_ptr(size_t r) const {
        if (r >= rows_) {
            throw std::out_of_range("TensorView row out of bounds");
        }
        return data_ + (r * stride_);
    }

    const double* data() const noexcept { return data_; }
    size_t rows() const noexcept { return rows_; }
    size_t cols() const noexcept { return cols_; }
    size_t stride() const noexcept { return stride_; }
    size_t size() const noexcept { return rows_ * cols_; }
    bool empty() const noexcept { return rows_ == 0 || cols_ == 0; }

    // Convert non-owning view to an owning MorphosML Matrix
    Matrix to_matrix() const {
        Matrix result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            const double* src = row_ptr(i);
            double* dst = result.row_ptr(i);
            std::memcpy(dst, src, cols_ * sizeof(double));
        }
        return result;
    }
};

} // namespace morphosml
