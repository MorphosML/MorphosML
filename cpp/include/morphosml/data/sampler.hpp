#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace morphosml {
namespace data {

/**
 * @brief Idempotent, deterministic pseudo-random sampler using SplitMix64.
 *
 * @details
 * `IdempotentSampler` provides provably reproducible dataset shuffling permutations without
 * relying on non-deterministic system entropy sources (`std::random_device`).
 *
 * ### Idempotency Principle:
 * For any given tuple \f$(\text{seed}, \text{epoch}, N)\f$, the generated index permutation
 * is guaranteed to be mathematically identical across runs, process restarts, or worker threads:
 * \f[
 * \Pi = \text{Permutation}(\text{Hash}(\text{seed} + \text{epoch}), N)
 * \f]
 *
 * Shuffling is performed in-place using the Fisher-Yates shuffle algorithm driven by a 64-bit
 * SplitMix64 state generator, ensuring uniform distribution with period \f$2^{64}\f$.
 */
class IdempotentSampler {
private:
    uint64_t base_seed_; ///< Base random seed.

public:
    /**
     * @brief Constructs an IdempotentSampler with a specific random seed.
     * @param seed Base unsigned 64-bit integer seed (default: 42).
     */
    explicit IdempotentSampler(uint64_t seed = 42) : base_seed_(seed) {}

    /**
     * @brief Generates a deterministic permutation of sample indices \([0, \text{total\_samples} - 1]\).
     *
     * @param total_samples Total number of elements/rows to shuffle.
     * @param epoch Training epoch index (mixes the seed deterministically per epoch).
     * @return std::vector<size_t> Vector containing the shuffled indices.
     */
    std::vector<size_t> generate_indices(size_t total_samples, uint32_t epoch = 0) const;

    /**
     * @brief Retrieves the current base seed.
     * @return uint64_t Base seed value.
     */
    uint64_t seed() const noexcept { return base_seed_; }

    /**
     * @brief Updates the base seed.
     * @param seed New 64-bit seed value.
     */
    void set_seed(uint64_t seed) noexcept { base_seed_ = seed; }
};

} // namespace data
} // namespace morphosml
