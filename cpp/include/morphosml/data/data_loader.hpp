#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstddef>

namespace morphosml {
namespace data {

template <typename BatchType>
class PrefetchQueue {
private:
    std::queue<BatchType> queue_;
    std::mutex mtx_;
    std::condition_variable cv_consumer_;
    std::condition_variable cv_producer_;
    size_t capacity_;
    std::atomic<bool> stopped_{false};

public:
    explicit PrefetchQueue(size_t capacity = 2) : capacity_(capacity) {}

    ~PrefetchQueue() {
        stop();
    }

    void push(BatchType batch) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_producer_.wait(lock, [this]() { return queue_.size() < capacity_ || stopped_.load(); });
        if (stopped_.load()) return;
        queue_.push(std::move(batch));
        cv_consumer_.notify_one();
    }

    bool pop(BatchType& out_batch) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_consumer_.wait(lock, [this]() { return !queue_.empty() || stopped_.load(); });
        if (queue_.empty()) return false;
        out_batch = std::move(queue_.front());
        queue_.pop();
        cv_producer_.notify_one();
        return true;
    }

    void stop() {
        stopped_.store(true);
        cv_producer_.notify_all();
        cv_consumer_.notify_all();
    }

    bool is_empty() {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.empty();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }
};

} // namespace data
} // namespace morphosml

