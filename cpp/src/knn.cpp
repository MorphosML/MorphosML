#include "morphosml/knn.hpp"
#include "morphosml/simd/simd_ops.hpp"
#include <cmath>

namespace morphosml {

    // Train the KNN model by storing the training data
    void KNN::fit(const Matrix& X, const std::vector<int>& y) {

        // Check if the number of rows matches the number of labels
        if (X.rows() != y.size()) {
            throw std::invalid_argument(
                "Number of samples and labels must be the same."
            );
        }

        // Store training samples
        X_train_ = X;

        // Store training labels
        y_train_ = y;
    }

    // Compute Euclidean distance between two vectors using SIMD
    double KNN::euclidean_distance(
        const Vector& a,
        const Vector& b
    ) const {
        return std::sqrt(simd::squared_distance(a.data().data(), b.data().data(), a.size()));
    }

    // Predict a single sample from raw pointer (zero intermediate allocations)
    int KNN::predict_single(const double* x_ptr, size_t n_features) const {
        std::priority_queue<DistanceIndex> heap;
        const size_t n_train = X_train_.rows();
        const size_t k_val = static_cast<size_t>(k_);

        // Compare input sample against ALL training samples using SIMD squared distance
        for (size_t i = 0; i < n_train; i++) {
            const double* train_row = X_train_.row_ptr(i);
            double dist_sq = simd::squared_distance(x_ptr, train_row, n_features);

            if (heap.size() < k_val) {
                heap.push({dist_sq, static_cast<int>(i)});
            } else if (dist_sq < heap.top().distance) {
                heap.pop();
                heap.push({dist_sq, static_cast<int>(i)});
            }
        }

        // Store vote counts for each class label
        std::unordered_map<int, int> votes;

        // Count labels from K nearest neighbors
        while (!heap.empty()) {
            votes[y_train_[heap.top().index]]++;
            heap.pop();
        }

        // Find class with highest vote count
        int best_label = -1;
        int best_count = -1;

        for (const auto& [label, count] : votes) {
            if (count > best_count) {
                best_count = count;
                best_label = label;
            }
        }

        return best_label;
    }

    // Predict a single sample Vector
    int KNN::predict_single(const Vector& x) const {
        return predict_single(x.data().data(), x.size());
    }

    // Predict multiple samples with OpenMP parallelization
    std::vector<int> KNN::predict(const Matrix& X) const {

        // Ensure model has been trained
        if (X_train_.rows() == 0) {
            throw std::runtime_error("Model not fitted yet");
        }

        const size_t n_samples = X.rows();
        const size_t n_features = X.cols();
        std::vector<int> predictions(n_samples);

#if defined(MORPHOSML_HAS_OPENMP)
        #pragma omp parallel for schedule(dynamic) if(n_samples > 8)
#endif
        for (size_t i = 0; i < n_samples; i++) {
            predictions[i] = predict_single(X.row_ptr(i), n_features);
        }

        return predictions;
    }

} // namespace morphosml