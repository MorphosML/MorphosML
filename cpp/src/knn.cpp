#include "morphosml/knn.hpp"
#include <cmath>
#include <stdexcept>

namespace morphosml {

    KNN::KNN(int k) : k_(k) {
        if (k_ <= 0) {
            throw std::invalid_argument("Number of neighbors k must be strictly positive.");
        }
    }

    // Train the KNN model by storing the training data
    void KNN::fit(const Matrix& X, const std::vector<int>& y) {
        if (X.rows() == 0 || X.cols() == 0) {
            throw std::invalid_argument("Training feature matrix X cannot be empty.");
        }

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

    // Compute Euclidean distance between two vectors
    double KNN::euclidean_distance(
        const Vector& a,
        const Vector& b
    ) const {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must have the same dimension for Euclidean distance.");
        }

        double sum = 0.0;

        // Iterate through each feature
        for (size_t i = 0; i < a.size(); i++) {

            // Difference between coordinates
            double diff = a[i] - b[i];

            // Sum squared differences
            sum += diff * diff;
        }

        // Return square root of the sum
        return std::sqrt(sum);
    }

    // Predict a single sample
    int KNN::predict_single(const Vector& x) const {
        if (X_train_.rows() == 0) {
            throw std::runtime_error("Model not fitted yet");
        }
        if (x.size() != X_train_.cols()) {
            throw std::invalid_argument("Input sample feature dimension does not match fitted model.");
        }

        // Max heap storing the K nearest neighbors
        //
        // The farthest neighbor among the K closest stays on top
        // so we can efficiently remove it if we find a closer one.
        std::priority_queue<DistanceIndex> heap;

        // Compare input sample against ALL training samples
        for (size_t i = 0; i < X_train_.rows(); i++) {

            // Get one training sample row (HPC contiguous zero-extra-allocation)
            Vector x_train = X_train_.row(i);

            // Compute distance from x to training sample
            double dist = euclidean_distance(x, x_train);

            // If heap is not full yet, push directly
            if (heap.size() < static_cast<size_t>(k_)) {

                heap.push({dist, i});

            // Otherwise replace the farthest neighbor if current is closer
            } else if (dist < heap.top().distance) {

                // Remove current farthest neighbor
                heap.pop();

                // Add closer neighbor
                heap.push({dist, i});
            }
        }

        // Store vote counts for each class label
        //
        // key   -> class label
        // value -> number of votes
        std::unordered_map<int, int> votes;

        // Count labels from K nearest neighbors
        while (!heap.empty()) {

            // Increase vote count for this label
            votes[y_train_[heap.top().index]]++;

            heap.pop();
        }

        // Variables to track most voted class
        int best_label = -1;
        int best_count = -1;

        // Find class with highest vote count
        for (const auto& [label, count] : votes) {

            if (count > best_count) {

                best_count = count;
                best_label = label;
            }
        }

        // Return predicted class
        return best_label;
    }

    // Predict multiple samples
    std::vector<int> KNN::predict(const Matrix& X) const {

        // Ensure model has been trained
        if (X_train_.rows() == 0) {
            throw std::runtime_error("Model not fitted yet");
        }
        if (X.cols() != X_train_.cols()) {
            throw std::invalid_argument("Input feature count does not match fitted model.");
        }

        // Store predictions
        std::vector<int> predictions;

        // Reserve memory for efficiency
        predictions.reserve(X.rows());

        // Predict each row independently
        for (size_t i = 0; i < X.rows(); i++) {

            // Extract sample row (HPC contiguous zero-extra-allocation)
            Vector x = X.row(i);

            // Predict and store result
            predictions.push_back(predict_single(x));
        }

        return predictions;
    }

}