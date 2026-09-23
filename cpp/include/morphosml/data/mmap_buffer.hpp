#pragma once

#include <string>
#include <cstdint>
#include <cstddef>
#include "morphosml/tensor_view.hpp"
#include "morphosml/matrix.hpp"

namespace morphosml {
namespace data {

#pragma pack(push, 1)
struct FileHeader {
    char magic[4];       // "MML\0"
    uint32_t version;    // 1
    uint64_t rows;
    uint64_t cols;
    uint64_t checksum;   // Simple 64-bit hash for validation & idempotency
    char reserved[36];   // Alignment padding for 64-byte header
};
#pragma pack(pop)

class MMapBuffer {
private:
    std::string filepath_;
    int fd_ = -1;
    size_t file_size_ = 0;
    void* mapped_data_ = nullptr;
    const FileHeader* header_ = nullptr;
    const double* raw_payload_ = nullptr;

public:
    explicit MMapBuffer(const std::string& filepath);
    ~MMapBuffer();

    // Disable copy, allow move
    MMapBuffer(const MMapBuffer&) = delete;
    MMapBuffer& operator=(const MMapBuffer&) = delete;
    MMapBuffer(MMapBuffer&& other) noexcept;
    MMapBuffer& operator=(MMapBuffer&& other) noexcept;

    // Zero-Copy Slice Access
    TensorView get_slice(size_t start_row, size_t num_rows) const;
    TensorView full_view() const;

    size_t rows() const noexcept { return header_ ? header_->rows : 0; }
    size_t cols() const noexcept { return header_ ? header_->cols : 0; }
    size_t size() const noexcept { return rows() * cols(); }
    uint64_t checksum() const noexcept { return header_ ? header_->checksum : 0; }
    const std::string& filepath() const noexcept { return filepath_; }

    // Static helper to create a high-speed binary .mldat dataset
    static void write_mldat(const std::string& filepath, const Matrix& mat);
};

} // namespace data
} // namespace morphosml

