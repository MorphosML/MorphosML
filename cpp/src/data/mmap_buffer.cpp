#include "morphosml/data/mmap_buffer.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <fstream>

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
        throw std::runtime_error("MMapBuffer failed to stat file: " + filepath);
    }
    file_size_ = sb.st_size;

    if (file_size_ < sizeof(FileHeader)) {
        close(fd_);
        throw std::runtime_error("File is smaller than valid .mldat header: " + filepath);
    }

    mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("MMapBuffer mmap failed on file: " + filepath);
    }

    // Inform Linux kernel to optimize page prefetching sequentially
    madvise(mapped_data_, file_size_, MADV_SEQUENTIAL);

    header_ = reinterpret_cast<const FileHeader*>(mapped_data_);

    // Validate magic bytes
    if (header_->magic[0] != 'M' || header_->magic[1] != 'M' || header_->magic[2] != 'L' || header_->magic[3] != '\0') {
        munmap(mapped_data_, file_size_);
        close(fd_);
        throw std::runtime_error("Invalid magic bytes in .mldat dataset: " + filepath);
    }

    const size_t expected_payload_bytes = header_->rows * header_->cols * sizeof(double);
    if (file_size_ < sizeof(FileHeader) + expected_payload_bytes) {
        munmap(mapped_data_, file_size_);
        close(fd_);
        throw std::runtime_error("File size truncated or payload corrupted in: " + filepath);
    }

    raw_payload_ = reinterpret_cast<const double*>(static_cast<const char*>(mapped_data_) + sizeof(FileHeader));
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
    if (start_row + num_rows > header_->rows) {
        throw std::out_of_range("Requested slice exceeds total rows in MMapBuffer");
    }
    const double* slice_ptr = raw_payload_ + (start_row * header_->cols);
    return TensorView(slice_ptr, num_rows, header_->cols, header_->cols);
}

TensorView MMapBuffer::full_view() const {
    if (!header_) {
        return TensorView();
    }
    return TensorView(raw_payload_, header_->rows, header_->cols, header_->cols);
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

    if (write(out_fd, &header, sizeof(header)) != sizeof(header)) {
        close(out_fd);
        throw std::runtime_error("Failed writing header to: " + filepath);
    }

    const size_t payload_bytes = mat.size() * sizeof(double);
    if (write(out_fd, mat.data(), payload_bytes) != static_cast<ssize_t>(payload_bytes)) {
        close(out_fd);
        throw std::runtime_error("Failed writing payload to: " + filepath);
    }

    close(out_fd);
}

} // namespace data
} // namespace morphosml

