#include "morphosml/knn.hpp"
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

    // Compute Euclidean distance between two vectors
    double KNN::euclidean_distance(
        const Vector& a,
        const Vector& b
    ) const {

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

        // Max heap storing the K nearest neighbors
        //
        // The farthest neighbor among the K closest stays on top
        // so we can efficiently remove it if we find a closer one.
        std::priority_queue<DistanceIndex> heap;

        // Compare input sample against ALL training samples
        for (size_t i = 0; i < X_train_.rows(); i++) {

            // Get one training sample row
            Vector x_train(X_train_.to_std()[i]);

            // Compute distance from x to training sample
            double dist = euclidean_distance(x, x_train);

            // If heap is not full yet, push directly
            if (heap.size() < static_cast<size_t>(k_)) {

                heap.push({dist, static_cast<int>(i)});

            // Otherwise replace the farthest neighbor if current is closer
            } else if (dist < heap.top().distance) {

                // Remove current farthest neighbor
                heap.pop();

                // Add closer neighbor
                heap.push({dist, static_cast<int>(i)});
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
    std::vector<int> KNN::predict(const Matrix& X) {

        // Ensure model has been trained
        if (X_train_.rows() == 0) {
            throw std::runtime_error("Model not fitted yet");
        }

        // Store predictions
        std::vector<int> predictions;

        // Reserve memory for efficiency
        predictions.reserve(X.rows());

        // Predict each row independently
        for (size_t i = 0; i < X.rows(); i++) {

            // Extract sample row
            Vector x(X.to_std()[i]);

            // Predict and store result
            predictions.push_back(predict_single(x));
        }

        return predictions;
    }

}