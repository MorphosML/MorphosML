#include "morphosml/vector.hpp"

namespace morphosml {

Vector Vector::operator+(const Vector& other) const {
    if (size() != other.size()){
        throw std::invalid_argument("Vectors need to be of the same size for these operations");
    }
    Vector result(size());

    for (size_t i = 0; i < size(); i++){
        result[i] = data_[i] + other[i];
    }
    return result;
} 

Vector Vector::operator-(const Vector& other) const {
    if (size() != other.size()){
        throw std::invalid_argument("Vectors need to be of the same size for these operations");
    }
    Vector result(size());

    for (size_t i = 0; i < size(); i++){
        result[i] = data_[i] - other[i];
    }
    return result;
} 

Vector Vector::operator*(double scalar) const {

    Vector result(size());
    for (size_t i = 0; i < size(); i++){
        result[i] = data_[i] * scalar;
    }
    return result;
    
} // scaling operation

double Vector::dot(const Vector& other) const {
    if (size() != other.size()){
        throw std::invalid_argument("Vectors need to be of the same size for these operations"); 
    }
    
    double result = 0.0; 

    for (size_t i = 0; i < size(); i++) {
        result += data_[i] * other[i];
    }
    return result;
    // dot product needs to result in the multiplication then sum of the products.
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