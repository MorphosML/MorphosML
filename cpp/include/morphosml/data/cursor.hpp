#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <sstream>

namespace morphosml {
namespace data {

struct IngestionCursor {
    uint32_t epoch = 0;
    size_t sample_offset = 0;
    uint64_t checksum = 0;

    std::string to_string() const {
        std::ostringstream oss;
        oss << epoch << ":" << sample_offset << ":" << checksum;
        return oss.str();
    }

    static IngestionCursor from_string(const std::string& str) {
        IngestionCursor cursor;
        char delim1, delim2;
        std::istringstream iss(str);
        if (iss >> cursor.epoch >> delim1 >> cursor.sample_offset >> delim2 >> cursor.checksum) {
            if (delim1 == ':' && delim2 == ':') {
                return cursor;
            }
        }
        throw std::invalid_argument("Invalid serialized IngestionCursor format: " + str);
    }
};

} // namespace data
} // namespace morphosml

