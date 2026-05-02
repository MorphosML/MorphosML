#pragma once // define once

#include <vector>
#include <stdexcept>
#include "vector.hpp"

namespace morphosml {

class Matrix {
    private:
    std::vector<std::vector<double>> data_;
    size_t rows,cols;
    
    void validate() const {
        if (data_.size() != rows * cols) {
        throw std::logic_error("Invalid matrix dimensions");
    }}
    
    Matrix augmented_solve(const Vector& b) const;
    void swap_rows(size_t i , size_t j) const;

    public: 

    // Constructors
    Matrix() : rows(0), cols(0) {}
    Matrix(size_t rows, size_t cols);
    Matrix(std::initializer_list<std::initializer_list<double>> list);

    // Accessors
    size_t rows() const {
        return rows;
    }
    size_t cols() const {
        return cols;
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

};
}
