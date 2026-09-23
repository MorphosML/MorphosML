#include "morphosml/matrix.hpp"
#include <cstring>
#include <iomanip>

namespace morphosml {

// Global seed state
static uint64_t g_global_seed = 42;
static bool g_global_seed_configured = false;

void set_seed(uint64_t seed) {
    g_global_seed = seed;
    g_global_seed_configured = true;
}

uint64_t get_seed() {
    return g_global_seed;
}

Matrix::Matrix(size_t rows, size_t cols, double init_val)
    : data_(rows * cols, init_val), n_rows(rows), n_cols(cols) {}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> list) {
    n_rows = list.size();
    if (n_rows == 0) {
        throw std::invalid_argument("Matrix cannot be empty");
    }
    n_cols = list.begin()->size();
    if (n_cols == 0) {
        throw std::invalid_argument("Matrix columns cannot be empty");
    }

    data_.resize(n_rows * n_cols);
    size_t i = 0;
    for (const auto& row_list : list) {
        if (row_list.size() != n_cols) {
            throw std::invalid_argument("Inconsistent row sizes");
        }
        std::copy(row_list.begin(), row_list.end(), data_.begin() + (i * n_cols));
        ++i;
    }
}

Matrix::Matrix(const std::vector<std::vector<double>>& data) {
    n_rows = data.size();
    n_cols = (n_rows == 0) ? 0 : data[0].size();
    data_.resize(n_rows * n_cols);

    for (size_t i = 0; i < n_rows; ++i) {
        if (data[i].size() != n_cols) {
            throw std::invalid_argument("Inconsistent row sizes in nested vector");
        }
        std::memcpy(data_.data() + (i * n_cols), data[i].data(), n_cols * sizeof(double));
    }
}

Matrix::Matrix(size_t rows, size_t cols, std::vector<double> flat_data)
    : data_(std::move(flat_data)), n_rows(rows), n_cols(cols) {
    if (data_.size() != n_rows * n_cols) {
        throw std::invalid_argument("Flat data size does not match rows * cols dimensions");
    }
}

double& Matrix::operator()(size_t i, size_t j) {
    if (i >= n_rows || j >= n_cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[i * n_cols + j];
}

const double& Matrix::operator()(size_t i, size_t j) const {
    if (i >= n_rows || j >= n_cols) {
        throw std::out_of_range("Matrix index out of bounds");
    }
    return data_[i * n_cols + j];
}

Vector Matrix::row(size_t i) const {
    if (i >= n_rows) {
        throw std::out_of_range("Row index out of bounds");
    }
    const double* ptr = row_ptr(i);
    std::vector<double> v_data(ptr, ptr + n_cols);
    return Vector(std::move(v_data));
}

Matrix Matrix::transpose() const {
    Matrix result(n_cols, n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        const double* r = row_ptr(i);
        for (size_t j = 0; j < n_cols; ++j) {
            result(j, i) = r[j];
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (n_rows != other.n_rows || n_cols != other.n_cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition");
    }
    Matrix result(n_rows, n_cols);
    const size_t total = data_.size();
    for (size_t i = 0; i < total; ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (n_rows != other.n_rows || n_cols != other.n_cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction");
    }
    Matrix result(n_rows, n_cols);
    const size_t total = data_.size();
    for (size_t i = 0; i < total; ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(n_rows, n_cols);
    const size_t total = data_.size();
    for (size_t i = 0; i < total; ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

// HPC Cache-friendly i-k-j loop reordering
Matrix Matrix::operator*(const Matrix& other) const {
    if (n_cols != other.n_rows) {
        throw std::invalid_argument("Matrix inner dimensions must agree for multiplication");
    }
    Matrix result(n_rows, other.n_cols, 0.0);
    const size_t p = other.n_cols;

    for (size_t i = 0; i < n_rows; ++i) {
        const double* a_row = this->row_ptr(i);
        double* c_row = result.row_ptr(i);
        for (size_t k = 0; k < n_cols; ++k) {
            const double r = a_row[k];
            const double* b_row = other.row_ptr(k);
            // Sequential inner loop allows SIMD auto-vectorization
            for (size_t j = 0; j < p; ++j) {
                c_row[j] += r * b_row[j];
            }
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
        const double* r = row_ptr(i);
        double sum = 0.0;
        for (size_t j = 0; j < n_cols; ++j) {
            sum += r[j] * vec[j];
        }
        result[i] = sum;
    }
    return result;
}

Matrix Matrix::identity(size_t n) {
    Matrix result(n, n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

Matrix Matrix::zeros(size_t rows, size_t cols) {
    return Matrix(rows, cols, 0.0);
}

Matrix Matrix::random(size_t rows, size_t cols) {
    if (g_global_seed_configured) {
        return random(rows, cols, g_global_seed);
    }
    std::random_device rd;
    return random(rows, cols, rd());
}

Matrix Matrix::random(size_t rows, size_t cols, uint64_t seed) {
    Matrix result(rows, cols);
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    const size_t total = rows * cols;
    for (size_t i = 0; i < total; ++i) {
        result.data_[i] = dis(gen);
    }
    return result;
}

void Matrix::print() const {
    std::cout << "[";
    for (size_t i = 0; i < n_rows; ++i) {
        if (i > 0) std::cout << " ";
        std::cout << "[";
        for (size_t j = 0; j < n_cols; ++j) {
            std::cout << (*this)(i, j);
            if (j < n_cols - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (i < n_rows - 1) std::cout << ",\n";
    }
    std::cout << "]" << std::endl;
}

std::vector<std::vector<double>> Matrix::to_std() const {
    std::vector<std::vector<double>> res(n_rows, std::vector<double>(n_cols));
    for (size_t i = 0; i < n_rows; ++i) {
        std::memcpy(res[i].data(), data_.data() + (i * n_cols), n_cols * sizeof(double));
    }
    return res;
}

} // namespace morphosml
