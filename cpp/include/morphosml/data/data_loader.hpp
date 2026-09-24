#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cstddef>
#include <utility>

namespace morphosml {
namespace data {

/**
 * @brief Thread-safe bounded producer-consumer queue for asynchronous double-buffering.
 *
 * @tparam BatchType Type of batch object held in the queue (e.g., `TensorView` or `Matrix`).
 *
 * @details
 * `PrefetchQueue` implements a bounded FIFO queue designed to overlap I/O disk reading or
 * decompression with CPU/GPU model computation ("zero bubbles"):
 * - **Producer (I/O thread)**: Enqueues batch \f$N + 1\f$. Blocks if the queue reaches `capacity_`.
 * - **Consumer (Trainer thread)**: Dequeues batch \f$N\f$. Blocks if the queue is empty.
 * - **Thread Safety**: Synchronized using `std::mutex` and two `std::condition_variable`s.
 * - **Clean Teardown**: Atomically signals cancellation via `stop()`, awakening any blocked threads.
 */
template <typename BatchType>
class PrefetchQueue {
private:
    std::queue<BatchType> queue_;             ///< Internal FIFO queue.
    std::mutex mtx_;                          ///< Mutex protecting queue access.
    std::condition_variable cv_consumer_;     ///< Notified when an item is pushed or queue stopped.
    std::condition_variable cv_producer_;     ///< Notified when space becomes available or queue stopped.
    size_t capacity_;                         ///< Maximum number of buffered batches (typically 2-3).
    std::atomic<bool> stopped_{false};        ///< Atomic flag indicating queue shutdown.

public:
    /**
     * @brief Constructs a bounded prefetch queue with a designated capacity.
     * @param capacity Maximum batch buffer capacity (default: 2 for double buffering).
     */
    explicit PrefetchQueue(size_t capacity = 2) : capacity_(capacity) {}

    /**
     * @brief Destructor stops queue and wakes all blocked threads.
     */
    ~PrefetchQueue() {
        stop();
    }

    /**
     * @brief Pushes a new batch into the queue, blocking if capacity is reached.
     * @param batch Batch item to push (moved into the internal queue).
     */
    void push(BatchType batch) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_producer_.wait(lock, [this]() { return queue_.size() < capacity_ || stopped_.load(); });
        if (stopped_.load()) return;
        queue_.push(std::move(batch));
        cv_consumer_.notify_one();
    }

    /**
     * @brief Pops a batch from the queue, blocking until a batch is available or queue stopped.
     * @param[out] out_batch Output reference where popped batch is moved.
     * @return bool True if a batch was successfully retrieved; false if queue is empty and stopped.
     */
    bool pop(BatchType& out_batch) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_consumer_.wait(lock, [this]() { return !queue_.empty() || stopped_.load(); });
        if (queue_.empty()) return false;
        out_batch = std::move(queue_.front());
        queue_.pop();
        cv_producer_.notify_one();
        return true;
    }

    /**
     * @brief Signals queue shutdown, waking all waiting producers and consumers.
     */
    void stop() {
        stopped_.store(true);
        cv_producer_.notify_all();
        cv_consumer_.notify_all();
    }

    /**
     * @brief Checks whether the queue is currently empty.
     * @return bool True if empty.
     */
    bool is_empty() {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.empty();
    }

    /**
     * @brief Returns current number of items buffered in the queue.
     * @return size_t Current item count.
     */
    size_t size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }
};

} // namespace data
} // namespace morphosml
