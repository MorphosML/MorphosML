#pragma once // define once

#include <vector>
#include <stdexcept>
#include "vector.hpp"
#include <random>
#include <iostream>

namespace morphosml {

class Matrix {
    private:
    std::vector<std::vector<double>> data_;
    size_t n_rows,n_cols;
    
    void validate() const {
        if (data_.size() != n_rows * n_cols) {
        throw std::logic_error("Invalid matrix dimensions");
    }}
    
    Matrix augmented_solve(const Vector& b) const;
    void swap_rows(size_t i , size_t j) const;

    public: 

    // Constructors
    Matrix() : n_rows(0), n_cols(0) {}
    Matrix(size_t n_rows, size_t n_cols);
    Matrix(std::initializer_list<std::initializer_list<double>> list);

    // Accessors
    size_t rows() const {
        return n_rows;
    }
    size_t cols() const {
        return n_cols;
    }
     
    double& operator()(size_t i , size_t t);
    const double& operator()(size_t i , size_t t) const;

    // Core operations

    Matrix transpose() const;
    Matrix operator*(const Matrix& other) const;
    Vector operator*(const Vector& vec) const;

    // Static factories
    static Matrix identity(size_t n);
    static Matrix zeros(size_t rows, size_t cols);
    static Matrix random(size_t rows, size_t cols);


    // Utility
    void print() const;
    std::vector<std::vector<double>> to_std() const {return data_ ; } // Fixed semicolon

    // System solvers (0.3.0)

};
}
