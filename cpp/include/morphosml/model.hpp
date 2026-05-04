#pragma once

#include "matrix.hpp"
#include <vector>


namespace morphosml
{
    class BaseModel {
        public:
        virtual ~BaseModel() = default;
        virtual void fit(const Matrix& X, const std::vector<int>& y)= 0;
        virtual std::vector<int> predict(const Matrix& X) const = 0;
    };
} // morphosml
