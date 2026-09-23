#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <sstream>
#include <stdexcept>

namespace morphosml {
namespace data {

/**
 * @brief Lightweight, serializable cursor representing an exact ingestion checkpoint.
 *
 * @details
 * `IngestionCursor` encapsulates the minimal state necessary to resume an interrupted training
 * or data pipeline without reprocessing datasets from line 0.
 *
 * ### Checkpoint Fields:
 * - `epoch`: The current training epoch number.
 * - `sample_offset`: The exact index within the epoch's permutation schedule.
 * - `checksum`: Dataset integrity checksum (prevents resuming against altered datasets).
 *
 * It serializes into a compact colon-delimited string token: `"epoch:sample_offset:checksum"`.
 */
struct IngestionCursor {
    uint32_t epoch = 0;          ///< Current epoch counter (0-indexed).
    size_t sample_offset = 0;    ///< Number of samples consumed within the current epoch.
    uint64_t checksum = 0;       ///< Expected 64-bit dataset checksum.

    /**
     * @brief Serializes the cursor state into a string token.
     * @return std::string Serialized token string in format `"epoch:sample_offset:checksum"`.
     */
    std::string to_string() const {
        std::ostringstream oss;
        oss << epoch << ":" << sample_offset << ":" << checksum;
        return oss.str();
    }

    /**
     * @brief Deserializes a cursor string token back into an IngestionCursor instance.
     *
     * @param str Serialized token string.
     * @return IngestionCursor Restored cursor.
     * @throws std::invalid_argument If the string does not match the `"epoch:offset:checksum"` format.
     */
    static IngestionCursor from_string(const std::string& str) {
        IngestionCursor cursor;
        char delim1, delim2;
        std::istringstream iss(str);
        if (iss >> cursor.epoch >> delim1 >> cursor.sample_offset >> delim2 >> cursor.checksum) {
            std::string extra;
            if (delim1 == ':' && delim2 == ':' && !(iss >> extra)) {
                return cursor;
            }
        }
        throw std::invalid_argument("Invalid serialized IngestionCursor format: " + str);
    }

    bool operator==(const IngestionCursor& other) const noexcept {
        return epoch == other.epoch && sample_offset == other.sample_offset && checksum == other.checksum;
    }

    bool operator!=(const IngestionCursor& other) const noexcept {
        return !(*this == other);
    }
};

} // namespace data
} // namespace morphosml
