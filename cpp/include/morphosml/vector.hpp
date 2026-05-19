#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace morphosml {

   class Vector {
    private:
        std::vector<double> data_;
    
    public:
        // Same as Vector() {}
        Vector() = default; 

        // Copy
        Vector(const Vector&) = default;
        Vector& operator=(const Vector&) = default;

        // Move
        Vector(Vector&&) noexcept = default;
        Vector& operator=(Vector&&) noexcept = default;

        // Constructor initializes data_ with n elements, setting them to 0.0
        explicit Vector(size_t n) :  data_(n,0.0) {}

        // to_std()    
        std::vector<double> to_std() const {
            return data_;
        }

        Vector(std::initializer_list<double> list) : data_(list) {}

        Vector(const std::vector<double>& vec) : data_(vec) {}

        // Basic operations
        size_t size() const {return data_.size();} // return size
         
        // Fast access
        double& operator[](size_t i) { return data_[i]; }
        const double& operator[](size_t i) const { return data_[i]; }

        // Safety acess
        double& at(size_t i) {
            if ( i>= data_.size() ) throw std::out_of_range("Index out of range");
            return data_[i];
        } // can modify ||  vec[0] = 10.0;  // allowed 

        const double& at(size_t i ) const {
            if (i >= data_.size()) throw std::out_of_range("Index out of range");
            return data_[i] ;
        } // read only
        
        // Vector operations 
        Vector operator+(const Vector& other) const;
        Vector operator-(const Vector& other) const;
        Vector operator*(double scalar) const;

        double dot(const Vector& other) const;
        double norm() const;
        Vector normalized() const;

        // void utility 
        void print() const;
        
        // Returns a copy of the internal data as a std::vector
        const std::vector<double>& data() const { return data_; } 

        // Avoids copying when returning large vectors
        Vector(std::vector<double>&& vec) noexcept 
            : data_(std::move(vec)) {} 
    };
    }// namespace morphosml
#endif