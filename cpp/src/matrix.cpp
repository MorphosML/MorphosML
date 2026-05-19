#include "morphosml/matrix.hpp"

namespace morphosml {
    Matrix::Matrix(size_t rows, size_t cols) : n_rows(rows),n_cols(cols), data_(rows,std::vector<double>(cols,0.0)) {}

    Matrix::Matrix(std::initializer_list<std::initializer_list<double>> list){
        if (list.size() == 0){
            throw std::invalid_argument("Matrix cannot be empty");
        }

        n_rows = list.size();
        n_cols = list.begin()->size(); // same as list[0].size()
        data_.resize(n_rows);

        size_t i = 0;

        if (n_cols == 0){
            throw std::invalid_argument("Matrix cannot be empty");
        }
        
        for (const auto& n_rows : list){
            if(n_rows.size() != n_cols){
                throw std::invalid_argument("Inconsistent row sizes");
        }

        data_[i++] = n_rows;
        
    }}

    double& Matrix::operator()(size_t i ,size_t j){
        if (i >= n_rows || j >= n_cols){
            throw std::out_of_range("Matrix index out of bounds");
        }
        return data_[i][j];
    }

    const double& Matrix::operator()(size_t i , size_t j) const{
        if (i >= n_rows || j >= n_cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[i][j];}

    Matrix Matrix::transpose() const {
    Matrix result(n_cols, n_rows); // the result have the tranpose structure
    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j < n_cols; ++j) {
            result(j, i) = data_[i][j];
        }
    }
    return result;}

    Matrix Matrix::operator+(const Matrix& other) const {

        if (n_rows != other.n_rows || n_cols != other.n_cols) {
            throw std::invalid_argument(
                "Matrices must have the same dimensions."
            );
        }

        Matrix result(n_rows, n_cols);

        for (size_t i = 0; i < n_rows; ++i) {

            for (size_t j = 0; j < n_cols; ++j) {

                result(i, j) =
                    data_[i][j] + other(i, j);
            }
        }

        return result;
    }

    Matrix Matrix::operator-(const Matrix& other) const {

        if (n_rows != other.n_rows || n_cols != other.n_cols) {
            throw std::invalid_argument("Matrices should be same dimensions.");
        }

        Matrix result(n_rows,n_cols);

        for (size_t i = 0; i < n_rows; i++) {

            for (size_t j = 0; j < n_cols; ++ j) {

                result(i,j) = 
                    data_[i][j] - other(i,j);
            }
        } 
        return result;
    } // continue from here

    Matrix Matrix::operator*(const Matrix& other) const {
        if (n_cols != other.n_rows) {
            throw std::invalid_argument("Number of cols from the first");
        }
        Matrix result(n_rows, other.n_cols); 

        for (size_t i = 0; i < n_rows; i++ ) {
            for (size_t j = 0; j < other.n_cols; j++) {
                double sum = 0.0;
                for (size_t k = 0; k < n_cols; ++k) {
                sum += data_[i][k] * other(k, j);
            }
            result(i, j) = sum;
            }
        }
        return result;
    }

    Vector Matrix::operator*(const Vector& vec) const {
    if (n_cols != vec.size()) {
        throw std::invalid_argument("Matrix columns must match vector size");
    }

    Vector result(n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < n_cols; ++j) {
            sum += data_[i][j] * vec[j];
        }
        result[i] = sum;
    }
    return result;
}
    void Matrix::print() const {
    std::cout << "[";
    for (size_t i = 0; i < n_rows; ++i) {
        if (i > 0) std::cout << " ";
        std::cout << "[";
        for (size_t j = 0; j < n_cols; ++j) {
            std::cout << data_[i][j];
            if (j < n_cols - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (i < n_rows - 1) std::cout << ",\n";
    }
    std::cout << "]" << std::endl;
}
    Matrix Matrix::identity(size_t n) {
    Matrix result(n, n);
    for (size_t i = 0; i < n; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}
    Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols);
}

    Matrix Matrix::random(size_t rows, size_t cols) {
    Matrix result(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0); // variation between -1.0 and 1 
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = dis(gen);
        }
    }
    return result;
}

}


