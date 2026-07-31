#pragma once

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>


namespace rtdl_num_embeddings_cuda {

constexpr int kBlockThreads = 256;
constexpr int kSamplesPerBlock = 64;
constexpr int kMaxLaunchBlocks = 65535;
constexpr int kWarpSize = 32;


template <typename T>
__host__ __device__ __forceinline__ T ceil_div(
    T numerator,
    T denominator
) {
    return (numerator + denominator - 1) / denominator;
}


__device__ __forceinline__ int find_logical_bin_right(
    float value,
    const float* edges,
    int n_real_bins
) {
    // Equivalent to:
    //
    // searchsorted(edges[:n_real_bins + 1], value, side="right") - 1
    //
    // followed by clamp to [0, n_real_bins - 1]. NaN follows CUDA/PyTorch
    // searchsorted behavior here because every comparison with NaN is false,
    // which sends it to the last bin.
    int low = 0;
    int high = n_real_bins + 1;

    while (low < high) {
        const int middle = low + (high - low) / 2;

        if (value < edges[middle]) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }

    int logical_bin = low - 1;

    if (logical_bin < 0) {
        logical_bin = 0;
    }

    if (logical_bin >= n_real_bins) {
        logical_bin = n_real_bins - 1;
    }

    return logical_bin;
}


__host__ __device__ __forceinline__ int physical_bin_index(
    int logical_bin,
    int n_real_bins,
    int max_n_bins
) {
    // rtdl_num_embeddings keeps the last real bin at max_n_bins - 1 while
    // earlier real bins stay at their logical indices.
    return (
        logical_bin == n_real_bins - 1
            ? max_n_bins - 1
            : logical_bin
    );
}


template <typename scalar_t>
__device__ __forceinline__ float quantize_to_float(
    float value
) {
    return static_cast<float>(
        static_cast<scalar_t>(value)
    );
}


template <>
__device__ __forceinline__ float quantize_to_float<float>(
    float value
) {
    return value;
}


template <typename scalar_t>
__device__ __forceinline__ float load_as_float(
    const scalar_t* pointer
) {
    return static_cast<float>(*pointer);
}


template <>
__device__ __forceinline__ float load_as_float<float>(
    const float* pointer
) {
    return *pointer;
}

}  // namespace rtdl_num_embeddings_cuda
