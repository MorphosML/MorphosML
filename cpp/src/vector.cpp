#include "morphosml/vector.hpp"
#include "morphosml/simd/simd_ops.hpp"

namespace morphosml {

Vector Vector::operator+(const Vector& other) const {
    if (size() != other.size()){
        throw std::invalid_argument("Vectors need to be of the same size for these operations");
    }
    Vector result(size());
    simd::vec_add(data_.data(), other.data_.data(), result.data_.data(), size());
    return result;
} 

Vector Vector::operator-(const Vector& other) const {
    if (size() != other.size()){
        throw std::invalid_argument("Vectors need to be of the same size for these operations");
    }
    Vector result(size());
    simd::vec_sub(data_.data(), other.data_.data(), result.data_.data(), size());
    return result;
} 

Vector Vector::operator*(double scalar) const {
    Vector result(size());
    simd::vec_scale(data_.data(), scalar, result.data_.data(), size());
    return result;
} // scaling operation

double Vector::dot(const Vector& other) const {
    if (size() != other.size()){
        throw std::invalid_argument("Vectors need to be of the same size for these operations"); 
    }
    return simd::dot(data_.data(), other.data_.data(), size());
}

double Vector::norm() const {
    return std::sqrt(dot(*this));
}

Vector Vector::normalized() const {
    double n = norm();
    if (n < 1e-10){
        throw std::runtime_error("Cannot normalize a zero vector");
    }

    return (*this) * (1.0/n);
} 

void Vector::print() const {
       std::cout << "[";
    for (size_t i = 0; i < size(); ++i) {
        std::cout << data_[i];
        if (i < size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

} // namespace morphosml