#pragma once

#include "model.hpp"
#include "matrix.hpp"
#include <vector>
#include <queue>
#include <unordered_map>

namespace morphosml {
    class KNN : public BaseModel {
        private:
        Matrix X_train_;
        std::vector<int> y_train_;
        int k_;

        struct DistanceIndex {
            double distance;
            int index;
            bool operator<(const DistanceIndex& other){
                return distance < other.distance;
            }
        };
    public:
        // Constructor
        explicit KNN(int k=3) : k_(k){}

        void fit(const Matrix& X, const std::vector<int>& y) override;
        std::vector<int> predict(const Matrix& X) const override;

    private:
        int predict_single(const Vector& x) const;
        double euclidean_distance(const Vector& a, const Vector& b) const;

    };
}