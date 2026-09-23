#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace morphosml {
namespace data {

class IdempotentSampler {
private:
    uint64_t base_seed_;

public:
    explicit IdempotentSampler(uint64_t seed = 42) : base_seed_(seed) {}

    // Deterministic SplitMix64 Fisher-Yates permutation
    std::vector<size_t> generate_indices(size_t total_samples, uint32_t epoch = 0) const;

    uint64_t seed() const noexcept { return base_seed_; }
    void set_seed(uint64_t seed) noexcept { base_seed_ = seed; }
};

} // namespace data
} // namespace morphosml

