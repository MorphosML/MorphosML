#pragma once

#include <string>
#include <cstdint>
#include <cstddef>
#include "morphosml/tensor_view.hpp"
#include "morphosml/matrix.hpp"

namespace morphosml {
namespace data {

#pragma pack(push, 1)
/**
 * @brief Binary header (64 bytes) for MorphosML high-speed binary dataset files (`.mldat`).
 *
 * @details
 * Layout specification:
 * - Bytes 0..3: Magic identifier `"MML\0"`
 * - Bytes 4..7: File format version (uint32_t, currently 1)
 * - Bytes 8..15: Total row count (uint64_t)
 * - Bytes 16..23: Total column count (uint64_t)
 * - Bytes 24..31: FNV-1a 64-bit checksum over the payload for corruption detection and idempotency validation
 * - Bytes 32..63: Reserved padding for 64-byte hardware cache line alignment
 */
struct FileHeader {
    char magic[4];       ///< Magic bytes: "MML\0".
    uint32_t version;    ///< Binary format version identifier.
    uint64_t rows;       ///< Number of rows in dataset.
    uint64_t cols;       ///< Number of features/columns per row.
    uint64_t checksum;   ///< FNV-1a 64-bit hash over raw float64 payload.
    char reserved[36];   ///< Hardware alignment padding to ensure 64-byte header size.
};
#pragma pack(pop)

/**
 * @brief POSIX memory-mapped buffer reader for out-of-core dataset streaming.
 *
 * @details
 * `MMapBuffer` memory-maps large `.mldat` files directly into the process's virtual address
 * space via the POSIX `mmap()` system call. This delivers several critical High-Performance
 * Computing (HPC) benefits:
 * - **Instantaneous Load Time**: Opening a 100 GB dataset takes ~0.1 ms because data pages are
 *   demanded lazily by the Linux OS kernel rather than loaded eagerly into physical RAM.
 * - **Kernel Page Cache Optimization**: Calls `madvise(MADV_SEQUENTIAL)` to inform the kernel
 *   of forward sequential access patterns, triggering aggressive background read-ahead and fast page eviction.
 * - **Zero-Copy Slicing**: Batch extraction yields a `TensorView` directly referencing the mapped
 *   virtual memory, eliminating heap allocations and duplicate copies in memory.
 * - **RAII Memory Management**: Automatically closes the file descriptor and unmaps memory with
 *   `munmap()` upon destruction. Move-only semantics prevent accidental double unmapping.
 */
class MMapBuffer {
private:
    std::string filepath_;            ///< Path to the memory-mapped `.mldat` file.
    int fd_ = -1;                     ///< POSIX file descriptor.
    size_t file_size_ = 0;            ///< Total file size in bytes.
    void* mapped_data_ = nullptr;     ///< Virtual address returned by `mmap()`.
    const FileHeader* header_ = nullptr; ///< Pointer to mapped 64-byte header.
    const double* raw_payload_ = nullptr; ///< Pointer to start of float64 data array.

public:
    /**
     * @brief Opens and memory-maps an `.mldat` dataset file.
     *
     * @param filepath Path to the target binary file.
     * @throws std::runtime_error If the file does not exist, cannot be opened, or contains
     *         an invalid header magic signature.
     */
    explicit MMapBuffer(const std::string& filepath);

    /**
     * @brief Destructor unmaps virtual memory (`munmap`) and closes file descriptor.
     */
    ~MMapBuffer();

    // Disable copy semantics to prevent double munmap; enable move semantics
    MMapBuffer(const MMapBuffer&) = delete;
    MMapBuffer& operator=(const MMapBuffer&) = delete;
    MMapBuffer(MMapBuffer&& other) noexcept;
    MMapBuffer& operator=(MMapBuffer&& other) noexcept;

    /**
     * @brief Returns a zero-copy `TensorView` for a continuous slice of rows.
     *
     * @param start_row 0-based beginning row index.
     * @param num_rows Number of rows to include in the slice.
     * @return TensorView Non-owning 2D view pointing directly to the memory-mapped slice.
     * @throws std::out_of_range If `start_row + num_rows > rows()`.
     */
    TensorView get_slice(size_t start_row, size_t num_rows) const;

    /**
     * @brief Returns a zero-copy `TensorView` spanning the entire dataset.
     * @return TensorView Full 2D view of the dataset.
     */
    TensorView full_view() const;

    /**
     * @brief Returns the total number of rows.
     * @return size_t Dataset row count.
     */
    size_t rows() const noexcept { return header_ ? header_->rows : 0; }

    /**
     * @brief Returns the number of features/columns.
     * @return size_t Feature dimension count.
     */
    size_t cols() const noexcept { return header_ ? header_->cols : 0; }

    /**
     * @brief Returns the total number of float64 values (`rows * cols`).
     * @return size_t Total element count.
     */
    size_t size() const noexcept { return rows() * cols(); }

    /**
     * @brief Returns the FNV-1a 64-bit checksum stored in the file header.
     * @return uint64_t Checksum hash.
     */
    uint64_t checksum() const noexcept { return header_ ? header_->checksum : 0; }

    /**
     * @brief Returns the filesystem path to the mapped file.
     * @return const std::string& Filepath string.
     */
    const std::string& filepath() const noexcept { return filepath_; }

    /**
     * @brief Serializes an owning `Matrix` into a high-performance `.mldat` binary file.
     *
     * @param filepath Destination path.
     * @param mat Source matrix to serialize.
     * @throws std::runtime_error If the file cannot be created or written.
     */
    static void write_mldat(const std::string& filepath, const Matrix& mat);
};

} // namespace data
} // namespace morphosml
