#include "morphosml/data/mmap_buffer.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <limits>
#include <cerrno>

namespace morphosml {
namespace data {

// Fast 64-bit FNV-1a hash for payload checksum
static uint64_t compute_checksum(const double* data, size_t num_elements) {
    uint64_t hash = 14695981039346656037ULL;
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(data);
    const size_t total_bytes = num_elements * sizeof(double);
    for (size_t i = 0; i < total_bytes; ++i) {
        hash ^= byte_ptr[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

MMapBuffer::MMapBuffer(const std::string& filepath) : filepath_(filepath) {
    fd_ = open(filepath.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("MMapBuffer failed to open file: " + filepath);
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("MMapBuffer failed to stat file: " + filepath);
    }
    if (!S_ISREG(sb.st_mode)) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("MMapBuffer requires a regular file: " + filepath);
    }
    if (sb.st_size < 0 || static_cast<uint64_t>(sb.st_size) < sizeof(FileHeader) ||
        static_cast<uint64_t>(sb.st_size) > std::numeric_limits<size_t>::max()) {
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("File is smaller than valid .mldat header or invalid size: " + filepath);
    }
    file_size_ = static_cast<size_t>(sb.st_size);

    mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        mapped_data_ = nullptr;
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("MMapBuffer mmap failed on file: " + filepath);
    }

    // Inform Linux kernel to optimize page prefetching sequentially
    madvise(mapped_data_, file_size_, MADV_SEQUENTIAL);

    header_ = reinterpret_cast<const FileHeader*>(mapped_data_);

    // Validate magic bytes
    if (header_->magic[0] != 'M' || header_->magic[1] != 'M' || header_->magic[2] != 'L' || header_->magic[3] != '\0') {
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("Invalid magic bytes in .mldat dataset: " + filepath);
    }

    if (header_->version != 1) {
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("Unsupported .mldat version in: " + filepath);
    }

    const uint64_t header_rows = header_->rows;
    const uint64_t header_cols = header_->cols;
    if (header_rows > std::numeric_limits<size_t>::max() ||
        header_cols > std::numeric_limits<size_t>::max() ||
        (header_cols > 0 && header_rows > (std::numeric_limits<size_t>::max() - sizeof(FileHeader)) / sizeof(double) / header_cols)) {
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("Corrupted .mldat dimensions cause size overflow: " + filepath);
    }

    const size_t expected_payload_bytes = static_cast<size_t>(header_rows * header_cols) * sizeof(double);
    if (file_size_ < sizeof(FileHeader) + expected_payload_bytes) {
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
        close(fd_);
        fd_ = -1;
        throw std::runtime_error("File size truncated or payload corrupted in: " + filepath);
    }

    raw_payload_ = (expected_payload_bytes == 0)
        ? nullptr
        : reinterpret_cast<const double*>(static_cast<const char*>(mapped_data_) + sizeof(FileHeader));
}

MMapBuffer::~MMapBuffer() {
    if (mapped_data_ && mapped_data_ != MAP_FAILED) {
        munmap(mapped_data_, file_size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
}

MMapBuffer::MMapBuffer(MMapBuffer&& other) noexcept
    : filepath_(std::move(other.filepath_)),
      fd_(other.fd_),
      file_size_(other.file_size_),
      mapped_data_(other.mapped_data_),
      header_(other.header_),
      raw_payload_(other.raw_payload_) {
    other.fd_ = -1;
    other.file_size_ = 0;
    other.mapped_data_ = nullptr;
    other.header_ = nullptr;
    other.raw_payload_ = nullptr;
}

MMapBuffer& MMapBuffer::operator=(MMapBuffer&& other) noexcept {
    if (this != &other) {
        if (mapped_data_ && mapped_data_ != MAP_FAILED) {
            munmap(mapped_data_, file_size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }

        filepath_ = std::move(other.filepath_);
        fd_ = other.fd_;
        file_size_ = other.file_size_;
        mapped_data_ = other.mapped_data_;
        header_ = other.header_;
        raw_payload_ = other.raw_payload_;

        other.fd_ = -1;
        other.file_size_ = 0;
        other.mapped_data_ = nullptr;
        other.header_ = nullptr;
        other.raw_payload_ = nullptr;
    }
    return *this;
}

TensorView MMapBuffer::get_slice(size_t start_row, size_t num_rows) const {
    if (!header_) {
        throw std::runtime_error("MMapBuffer is not initialized");
    }
    if (start_row > header_->rows || num_rows > header_->rows - start_row) {
        throw std::out_of_range("Requested slice exceeds total rows in MMapBuffer");
    }
    if (num_rows == 0 || header_->cols == 0) {
        return TensorView(nullptr, num_rows, header_->cols, header_->cols);
    }
    const double* slice_ptr = raw_payload_ + (start_row * header_->cols);
    return TensorView(slice_ptr, num_rows, header_->cols, header_->cols);
}

TensorView MMapBuffer::full_view() const {
    if (!header_) {
        return TensorView();
    }
    if (header_->rows == 0 || header_->cols == 0) {
        return TensorView(nullptr, header_->rows, header_->cols, header_->cols);
    }
    return TensorView(raw_payload_, header_->rows, header_->cols, header_->cols);
}

static void write_all(int fd, const void* buf, size_t count, const std::string& filepath, const char* stage) {
    const char* ptr = static_cast<const char*>(buf);
    size_t remaining = count;
    while (remaining > 0) {
        ssize_t written = write(fd, ptr, remaining);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            close(fd);
            throw std::runtime_error(std::string("Failed writing ") + stage + " to: " + filepath);
        }
        if (written == 0) {
            close(fd);
            throw std::runtime_error(std::string("Unexpected zero-byte write for ") + stage + " in: " + filepath);
        }
        ptr += written;
        remaining -= static_cast<size_t>(written);
    }
}

void MMapBuffer::write_mldat(const std::string& filepath, const Matrix& mat) {
    int out_fd = open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out_fd == -1) {
        throw std::runtime_error("Failed to create .mldat file: " + filepath);
    }

    FileHeader header;
    header.magic[0] = 'M';
    header.magic[1] = 'M';
    header.magic[2] = 'L';
    header.magic[3] = '\0';
    header.version = 1;
    header.rows = mat.rows();
    header.cols = mat.cols();
    header.checksum = compute_checksum(mat.data(), mat.size());
    std::memset(header.reserved, 0, sizeof(header.reserved));

    write_all(out_fd, &header, sizeof(header), filepath, "header");

    const size_t payload_bytes = mat.size() * sizeof(double);
    if (payload_bytes > 0) {
        write_all(out_fd, mat.data(), payload_bytes, filepath, "payload");
    }

    close(out_fd);
}

} // namespace data
} // namespace morphosml

