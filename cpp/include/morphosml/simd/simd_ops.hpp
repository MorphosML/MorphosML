#pragma once

#include <cstddef>
#include <cmath>
#include <cstring>
#include <string>

// Detect x86_64 AVX2 / FMA support
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__AVX2__)
        #include <immintrin.h>
        #define MORPHOSML_HAS_AVX2 1
    #endif
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #include <arm_neon.h>
    #define MORPHOSML_HAS_NEON 1
#endif

// Detect OpenMP support
#ifdef _OPENMP
    #include <omp.h>
    #define MORPHOSML_HAS_OPENMP 1
#endif

namespace morphosml {
namespace simd {

/**
 * @brief Returns a string describing hardware acceleration features active at compile-time.
 */
inline std::string get_simd_capabilities() {
    std::string caps;
#if defined(MORPHOSML_HAS_AVX2)
    caps += "AVX2 ";
#endif
#if defined(__FMA__)
    caps += "FMA ";
#endif
#if defined(MORPHOSML_HAS_NEON)
    caps += "ARM_NEON ";
#endif
#if defined(MORPHOSML_HAS_OPENMP)
    caps += "OpenMP ";
#endif
    if (caps.empty()) {
        caps = "Scalar Fallback";
    }
    return caps;
}

/**
 * @brief Returns the number of active OpenMP worker threads.
 */
inline int get_num_threads() {
#if defined(MORPHOSML_HAS_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/**
 * @brief Sets the number of OpenMP worker threads.
 */
inline void set_num_threads(int num_threads) {
#if defined(MORPHOSML_HAS_OPENMP)
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#else
    (void)num_threads;
#endif
}

/**
 * @brief High-performance vector dot product with AVX2/FMA unrolling.
 *
 * @param a Pointer to first contiguous double buffer.
 * @param b Pointer to second contiguous double buffer.
 * @param n Number of elements.
 * @return double Computed inner dot product.
 */
inline double dot(const double* a, const double* b, size_t n) {
#if defined(MORPHOSML_HAS_AVX2)
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    __m256d sum2 = _mm256_setzero_pd();
    __m256d sum3 = _mm256_setzero_pd();

    size_t i = 0;
    // Unroll 16 doubles (4 accumulators x 4 lanes) per iteration
    for (; i + 16 <= n; i += 16) {
        __m256d a0 = _mm256_loadu_pd(a + i);
        __m256d b0 = _mm256_loadu_pd(b + i);
        __m256d a1 = _mm256_loadu_pd(a + i + 4);
        __m256d b1 = _mm256_loadu_pd(b + i + 4);
        __m256d a2 = _mm256_loadu_pd(a + i + 8);
        __m256d b2 = _mm256_loadu_pd(b + i + 8);
        __m256d a3 = _mm256_loadu_pd(a + i + 12);
        __m256d b3 = _mm256_loadu_pd(b + i + 12);

#if defined(__FMA__)
        sum0 = _mm256_fmadd_pd(a0, b0, sum0);
        sum1 = _mm256_fmadd_pd(a1, b1, sum1);
        sum2 = _mm256_fmadd_pd(a2, b2, sum2);
        sum3 = _mm256_fmadd_pd(a3, b3, sum3);
#else
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(a0, b0));
        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(a1, b1));
        sum2 = _mm256_add_pd(sum2, _mm256_mul_pd(a2, b2));
        sum3 = _mm256_add_pd(sum3, _mm256_mul_pd(a3, b3));
#endif
    }

    // 4-element loop
    for (; i + 4 <= n; i += 4) {
        __m256d a0 = _mm256_loadu_pd(a + i);
        __m256d b0 = _mm256_loadu_pd(b + i);
#if defined(__FMA__)
        sum0 = _mm256_fmadd_pd(a0, b0, sum0);
#else
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(a0, b0));
#endif
    }

    // Reduce 4 accumulators into one 256-bit register
    __m256d sum = _mm256_add_pd(_mm256_add_pd(sum0, sum1), _mm256_add_pd(sum2, sum3));

    // Horizontal sum of 4 double lanes in __m256d
    __m128d hi = _mm256_extractf128_pd(sum, 1);
    __m128d lo = _mm256_castpd256_pd128(sum);
    __m128d s128 = _mm_add_pd(lo, hi);
    __m128d shuf = _mm_unpackhi_pd(s128, s128);
    double total = _mm_cvtsd_f64(_mm_add_pd(s128, shuf));

    // Scalar cleanup for remaining elements
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;

#elif defined(MORPHOSML_HAS_NEON)
    float64x2_t sum0 = vdupq_n_f64(0.0);
    float64x2_t sum1 = vdupq_n_f64(0.0);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float64x2_t a0 = vld1q_f64(a + i);
        float64x2_t b0 = vld1q_f64(b + i);
        float64x2_t a1 = vld1q_f64(a + i + 2);
        float64x2_t b1 = vld1q_f64(b + i + 2);
        sum0 = vmlaq_f64(sum0, a0, b0);
        sum1 = vmlaq_f64(sum1, a1, b1);
    }
    float64x2_t sum = vaddq_f64(sum0, sum1);
    double total = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;

#else
    // Scalar fallback
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
#endif
}

/**
 * @brief Computes squared Euclidean distance sum((a_i - b_i)^2) using SIMD.
 */
inline double squared_distance(const double* a, const double* b, size_t n) {
#if defined(MORPHOSML_HAS_AVX2)
    __m256d sum0 = _mm256_setzero_pd();
    __m256d sum1 = _mm256_setzero_pd();
    size_t i = 0;

    for (; i + 8 <= n; i += 8) {
        __m256d va0 = _mm256_loadu_pd(a + i);
        __m256d vb0 = _mm256_loadu_pd(b + i);
        __m256d diff0 = _mm256_sub_pd(va0, vb0);

        __m256d va1 = _mm256_loadu_pd(a + i + 4);
        __m256d vb1 = _mm256_loadu_pd(b + i + 4);
        __m256d diff1 = _mm256_sub_pd(va1, vb1);

#if defined(__FMA__)
        sum0 = _mm256_fmadd_pd(diff0, diff0, sum0);
        sum1 = _mm256_fmadd_pd(diff1, diff1, sum1);
#else
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(diff0, diff0));
        sum1 = _mm256_add_pd(sum1, _mm256_mul_pd(diff1, diff1));
#endif
    }

    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        __m256d diff = _mm256_sub_pd(va, vb);
#if defined(__FMA__)
        sum0 = _mm256_fmadd_pd(diff, diff, sum0);
#else
        sum0 = _mm256_add_pd(sum0, _mm256_mul_pd(diff, diff));
#endif
    }

    __m256d sum = _mm256_add_pd(sum0, sum1);
    __m128d hi = _mm256_extractf128_pd(sum, 1);
    __m128d lo = _mm256_castpd256_pd128(sum);
    __m128d s128 = _mm_add_pd(lo, hi);
    __m128d shuf = _mm_unpackhi_pd(s128, s128);
    double total = _mm_cvtsd_f64(_mm_add_pd(s128, shuf));

    for (; i < n; ++i) {
        double d = a[i] - b[i];
        total += d * d;
    }
    return total;
#else
    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        total += d * d;
    }
    return total;
#endif
}

/**
 * @brief Vectorized element-wise addition out = a + b.
 */
inline void vec_add(const double* a, const double* b, double* out, size_t n) {
#if defined(MORPHOSML_HAS_AVX2)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_add_pd(va, vb));
    }
    for (; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
#endif
}

/**
 * @brief Vectorized element-wise subtraction out = a - b.
 */
inline void vec_sub(const double* a, const double* b, double* out, size_t n) {
#if defined(MORPHOSML_HAS_AVX2)
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vb = _mm256_loadu_pd(b + i);
        _mm256_storeu_pd(out + i, _mm256_sub_pd(va, vb));
    }
    for (; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
#endif
}

/**
 * @brief Vectorized scalar scaling out = a * scalar.
 */
inline void vec_scale(const double* a, double scalar, double* out, size_t n) {
#if defined(MORPHOSML_HAS_AVX2)
    __m256d s = _mm256_set1_pd(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        _mm256_storeu_pd(out + i, _mm256_mul_pd(va, s));
    }
    for (; i < n; ++i) {
        out[i] = a[i] * scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * scalar;
    }
#endif
}

/**
 * @brief Vectorized fused multiply-add: out[i] += a[i] * scalar.
 */
inline void vec_fmadd(const double* a, double scalar, double* out, size_t n) {
#if defined(MORPHOSML_HAS_AVX2)
    __m256d s = _mm256_set1_pd(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(a + i);
        __m256d vo = _mm256_loadu_pd(out + i);
#if defined(__FMA__)
        vo = _mm256_fmadd_pd(va, s, vo);
#else
        vo = _mm256_add_pd(vo, _mm256_mul_pd(va, s));
#endif
        _mm256_storeu_pd(out + i, vo);
    }
    for (; i < n; ++i) {
        out[i] += a[i] * scalar;
    }
#else
    for (size_t i = 0; i < n; ++i) {
        out[i] += a[i] * scalar;
    }
#endif
}

/**
 * @brief Multi-threaded and AVX2-accelerated Matrix Multiplication (GEMM): C = A * B.
 *
 * @param A Pointer to row-major M x K matrix.
 * @param B Pointer to row-major K x N matrix.
 * @param C Pointer to row-major M x N output matrix (overwritten).
 * @param M Number of rows in A and C.
 * @param K Inner dimension.
 * @param N Number of columns in B and C.
 */
inline void parallel_gemm(
    const double* A, const double* B, double* C,
    size_t M, size_t K, size_t N
) {
    // Zero out output buffer
    std::memset(C, 0, M * N * sizeof(double));

#if defined(MORPHOSML_HAS_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const double a_ik = A[i * K + k];

#if defined(MORPHOSML_HAS_AVX2)
            __m256d a_val = _mm256_set1_pd(a_ik);
            size_t j = 0;

            for (; j + 4 <= N; j += 4) {
                __m256d c_val = _mm256_loadu_pd(C + i * N + j);
                __m256d b_val = _mm256_loadu_pd(B + k * N + j);
#if defined(__FMA__)
                c_val = _mm256_fmadd_pd(a_val, b_val, c_val);
#else
                c_val = _mm256_add_pd(c_val, _mm256_mul_pd(a_val, b_val));
#endif
                _mm256_storeu_pd(C + i * N + j, c_val);
            }

            for (; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
#else
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
#endif
        }
    }
}

} // namespace simd
} // namespace morphosml
