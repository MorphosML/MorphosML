#pragma once

#include "model.hpp"
#include "matrix.hpp"
#include <vector>

namespace morphosml {
    class linear_regression : public BaseModel {
        private : 
        Matrix x_train;
    };
    
    
}