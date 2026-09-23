#include "morphosml/data/sampler.hpp"
#include <numeric>
#include <algorithm>

namespace morphosml {
namespace data {

static inline uint64_t splitmix64(uint64_t state) {
    state += 0x9e3779b97f4a7c15ULL;
    state = (state ^ (state >> 30)) * 0xbf58476d1ce4e5b9ULL;
    state = (state ^ (state >> 27)) * 0x94d049bb133111ebULL;
    return state ^ (state >> 31);
}

std::vector<size_t> IdempotentSampler::generate_indices(size_t total_samples, uint32_t epoch) const {
    std::vector<size_t> indices(total_samples);
    std::iota(indices.begin(), indices.end(), 0);

    if (total_samples <= 1) {
        return indices;
    }

    // Blend base seed with epoch using golden ratio prime
    uint64_t state = base_seed_ ^ (static_cast<uint64_t>(epoch) * 0x517cc1b727220a95ULL + 0x9e3779b97f4a7c15ULL);

    // Deterministic Fisher-Yates shuffle
    for (size_t i = total_samples - 1; i > 0; --i) {
        state = splitmix64(state);
        size_t j = state % (i + 1);
        std::swap(indices[i], indices[j]);
    }

    return indices;
}

} // namespace data
} // namespace morphosml

