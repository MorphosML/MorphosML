#pragma once

#include <vector>
#include <stdexcept>
#include "vector.hpp"
#include <random>
#include <iostream>
#include <cstddef>
#include <cstring>

namespace morphosml {

// Global deterministic seed control for reproducibility (Issue #9 & #13)
void set_seed(uint64_t seed);
uint64_t get_seed();

class Matrix {
private:
    // HPC Contiguous 1D flat row-major buffer: [i * n_cols + j]
    std::vector<double> data_;
    size_t n_rows;
    size_t n_cols;

public:
    // Constructors
    Matrix() : n_rows(0), n_cols(0) {}
    Matrix(size_t rows, size_t cols, double init_val = 0.0);
    Matrix(std::initializer_list<std::initializer_list<double>> list);
    Matrix(const std::vector<std::vector<double>>& data);
    Matrix(size_t rows, size_t cols, std::vector<double> flat_data);

    // Rule of 5: Default copy and move semantics
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix(Matrix&&) noexcept = default;
    Matrix& operator=(Matrix&&) noexcept = default;

    // Dimensions
    size_t rows() const noexcept { return n_rows; }
    size_t cols() const noexcept { return n_cols; }
    size_t size() const noexcept { return n_rows * n_cols; }
    bool empty() const noexcept { return data_.empty(); }

    // HPC Element & Pointer Access
    double& operator()(size_t i, size_t j);
    const double& operator()(size_t i, size_t j) const;

    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

    double* row_ptr(size_t i) noexcept { return data_.data() + i * n_cols; }
    const double* row_ptr(size_t i) const noexcept { return data_.data() + i * n_cols; }

    Vector row(size_t i) const;

    // Core HPC Operations
    Matrix transpose() const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const; // Cache-optimized i-k-j loop
    Matrix operator*(double scalar) const;
    Vector operator*(const Vector& vec) const;

    // Static Factories
    static Matrix identity(size_t n);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix random(size_t rows, size_t cols);
    static Matrix random(size_t rows, size_t cols, uint64_t seed);

    // Utility & Interoperability
    void print() const;
    const std::vector<double>& raw_data() const noexcept { return data_; }
    std::vector<std::vector<double>> to_std() const;
};

} // namespace morphosml
