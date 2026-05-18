#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

namespace morphosml {
    inline double accuracy(const std::vector<int>& y_true,const std::vector<int>& y_hat) {
        size_t correct = 0;

        for (size_t i=0; i < y_true.size(); i++) {
            if (y_true[i] == y_hat[i]) correct++;
        }
        return static_cast<double>(correct) / y_true.size();
    }

    inline double mse(const std::vector<int>& y_true, const std::vector<int>& y_hat) {
        double sum = 0.0;
        
        for (size_t i = 0; i < y_true.size(); ++ i ) {
            double diff = y_true[i] - y_hat[i];
            sum += diff * diff;
        }
        return sum / y_true.size();
    }

    // inline double precision() {}

    // inline double recall() {}

} // namespace morphosml