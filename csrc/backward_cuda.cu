#include "common.cuh"

#include <ATen/cuda/cub.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>


namespace rtdl_num_embeddings_cuda {

constexpr int kTargetPersistentBlocks = 256;
constexpr int kPrivateBlockThreads = 64;
constexpr int kMaxBatchPartitions = 4;
constexpr int kMinSamplesPerPartition = 256;

constexpr int kSortedReductionWarps = 4;
constexpr int kSortedReductionThreads = (
    kSortedReductionWarps * kWarpSize
);
constexpr int kSortedDPerBlock = 64;
constexpr int64_t kSortedMinEmbeddingDim = 20;
constexpr int64_t kSortedMinPairs = 262144;

template <int kTileD>
__device__ __forceinline__ float group_reduce_sum(
    float value
) {
    for (
        int offset = kTileD / 2;
        offset > 0;
        offset /= 2
    ) {
        value += __shfl_down_sync(
            0xffffffffu,
            value,
            offset,
            kTileD
        );
    }
    return value;
}


template <int kTileD>
__device__ __forceinline__ unsigned int subgroup_mask() {
    if constexpr (kTileD == 32) {
        return 0xffffffffu;
    } else {
        return (
            (threadIdx.x & 16) == 0
                ? 0x0000ffffu
                : 0xffff0000u
        );
    }
}


template <typename value_t, int kTileD>
__device__ __forceinline__ value_t subgroup_broadcast(
    value_t value
) {
    return __shfl_sync(
        subgroup_mask<kTileD>(),
        value,
        0,
        kTileD
    );
}


// Each logical group owns a private FP32 histogram. Therefore, no atomic
// operation is needed in the sample loop. Different batch partitions only
// meet after the complete reverse-prefix result has been formed, reducing
// global atomics to at most one operation per output element and partition.
template <typename grad_t, typename bin_index_t, int kTileD>
__global__ void ple_backward_private_histogram_kernel(
    const grad_t* __restrict__ grad_output,
    const float* __restrict__ x,
    const float* __restrict__ bin_edges,
    const float* __restrict__ inv_bin_widths,
    const int32_t* __restrict__ n_bins,
    const bin_index_t* __restrict__ bin_indices,
    float* __restrict__ grad_weight,
    int64_t batch_size,
    int64_t n_features,
    int64_t max_n_bins,
    int64_t d_embedding,
    int64_t n_d_tiles,
    int64_t n_partitions,
    int64_t total_persistent_tiles
) {
    static_assert(
        kPrivateBlockThreads % kTileD == 0,
        "The private histogram block must contain complete groups"
    );

    constexpr int kGroupsPerBlock = (
        kPrivateBlockThreads / kTileD
    );
    constexpr int kPrivateWarps = (
        kPrivateBlockThreads / kWarpSize
    );

    extern __shared__ float shared_memory[];

    // Bank-aware layout: [private warp][bin][physical warp lane].
    // For kTileD=16, the two logical groups in a warp occupy disjoint
    // half-warps of the same 32-float row. Thus, even if the two groups
    // update different bins, every physical lane retains its own bank.
    const int histogram_elements = (
        kPrivateWarps
        * static_cast<int>(max_n_bins)
        * kWarpSize
    );

    float* shared_bucket = shared_memory;
    float* shared_weighted = (
        shared_bucket + histogram_elements
    );

    const int group_idx = threadIdx.x / kTileD;
    const int lane_idx = threadIdx.x % kTileD;

    for (
        int64_t persistent_tile = blockIdx.x;
        persistent_tile < total_persistent_tiles;
        persistent_tile += gridDim.x
    ) {
        int64_t remainder = persistent_tile;
        const int64_t partition_idx = (
            remainder % n_partitions
        );
        remainder /= n_partitions;
        const int64_t d_tile = remainder % n_d_tiles;
        const int64_t feature_idx = remainder / n_d_tiles;

        const int n_real_bins = n_bins[feature_idx];
        const int64_t d_global = (
            d_tile * kTileD + lane_idx
        );

        for (
            int index = threadIdx.x;
            index < histogram_elements;
            index += blockDim.x
        ) {
            shared_bucket[index] = 0.0f;
            shared_weighted[index] = 0.0f;
        }
        __syncthreads();

        const int64_t partition_begin = (
            batch_size * partition_idx / n_partitions
        );
        const int64_t partition_end = (
            batch_size * (partition_idx + 1) / n_partitions
        );

        int cached_bin = -1;
        float cached_bucket = 0.0f;
        float cached_weighted = 0.0f;

        for (
            int64_t sample_idx = partition_begin + group_idx;
            sample_idx < partition_end;
            sample_idx += kGroupsPerBlock
        ) {
            int logical_bin = 0;
            float quantized_position = 0.0f;

            if (lane_idx == 0) {
                const int64_t pair_idx = (
                    sample_idx * n_features + feature_idx
                );
                logical_bin = static_cast<int>(
                    bin_indices[pair_idx]
                );
                const float value = x[pair_idx];
                const float left_edge = bin_edges[
                    feature_idx * (max_n_bins + 1)
                    + logical_bin
                ];
                const float current_inv_width = inv_bin_widths[
                    feature_idx * max_n_bins
                    + logical_bin
                ];
                const float position = (
                    value - left_edge
                ) * current_inv_width;
                quantized_position = (
                    quantize_to_float<grad_t>(position)
                );
            }

            logical_bin = subgroup_broadcast<int, kTileD>(
                logical_bin
            );
            quantized_position = subgroup_broadcast<float, kTileD>(
                quantized_position
            );

            if (d_global < d_embedding) {
                const int64_t output_index = (
                    (
                        sample_idx * n_features
                        + feature_idx
                    )
                    * d_embedding
                    + d_global
                );
                const float grad_value = load_as_float<grad_t>(
                    grad_output + output_index
                );

                // Run-length aggregation is cheap when consecutive samples
                // land in the same bin and does not introduce any atomics.
                if (logical_bin != cached_bin) {
                    if (cached_bin >= 0) {
                        const int cached_index = (
                            (
                                (threadIdx.x / kWarpSize)
                                * static_cast<int>(max_n_bins)
                                + cached_bin
                            )
                            * kWarpSize
                            + (threadIdx.x % kWarpSize)
                        );
                        shared_bucket[cached_index] += cached_bucket;
                        shared_weighted[cached_index] += cached_weighted;
                    }
                    cached_bin = logical_bin;
                    cached_bucket = grad_value;
                    cached_weighted = (
                        quantized_position * grad_value
                    );
                } else {
                    cached_bucket += grad_value;
                    cached_weighted += (
                        quantized_position * grad_value
                    );
                }
            }
        }

        if (cached_bin >= 0) {
            const int cached_index = (
                (
                    (threadIdx.x / kWarpSize)
                    * static_cast<int>(max_n_bins)
                    + cached_bin
                )
                * kWarpSize
                + (threadIdx.x % kWarpSize)
            );
            shared_bucket[cached_index] += cached_bucket;
            shared_weighted[cached_index] += cached_weighted;
        }

        __syncthreads();

        if (
            threadIdx.x < kTileD
            && d_global < d_embedding
        ) {
            float future_bucket_sum = 0.0f;

            for (
                int logical_bin = n_real_bins - 1;
                logical_bin >= 0;
                --logical_bin
            ) {
                float bucket_value = 0.0f;
                float weighted_value = 0.0f;

                #pragma unroll
                for (
                    int private_group = 0;
                    private_group < kGroupsPerBlock;
                    ++private_group
                ) {
                    const int private_warp = (
                        private_group * kTileD / kWarpSize
                    );
                    const int private_lane = (
                        private_group * kTileD % kWarpSize
                        + lane_idx
                    );
                    const int histogram_index = (
                        (
                            private_warp
                            * static_cast<int>(max_n_bins)
                            + logical_bin
                        )
                        * kWarpSize
                        + private_lane
                    );
                    bucket_value += shared_bucket[histogram_index];
                    weighted_value += shared_weighted[histogram_index];
                }

                const float partial_grad = (
                    weighted_value + future_bucket_sum
                );
                future_bucket_sum += bucket_value;

                const int physical_bin = physical_bin_index(
                    logical_bin,
                    n_real_bins,
                    static_cast<int>(max_n_bins)
                );
                const int64_t output_index = (
                    (
                        feature_idx * max_n_bins
                        + physical_bin
                    )
                    * d_embedding
                    + d_global
                );

                if (n_partitions == 1) {
                    grad_weight[output_index] = partial_grad;
                } else {
                    atomicAdd(
                        grad_weight + output_index,
                        partial_grad
                    );
                }
            }
        }

        __syncthreads();
    }
}


template <typename grad_t, typename bin_index_t>
__global__ void ple_backward_grad_x_kernel(
    const grad_t* __restrict__ grad_output,
    const float* __restrict__ x,
    const float* __restrict__ bin_edges,
    const float* __restrict__ inv_bin_widths,
    const int32_t* __restrict__ n_bins,
    const float* __restrict__ weight,
    const bin_index_t* __restrict__ bin_indices,
    float* __restrict__ grad_x,
    int64_t n_features,
    int64_t max_n_bins,
    int64_t d_embedding,
    int64_t total_pairs
) {
    constexpr int kWarpsPerBlock = (
        kBlockThreads / kWarpSize
    );

    const int warp_idx = threadIdx.x / kWarpSize;
    const int lane_idx = threadIdx.x % kWarpSize;

    for (
        int64_t pair_idx = (
            static_cast<int64_t>(blockIdx.x)
            * kWarpsPerBlock
            + warp_idx
        );
        pair_idx < total_pairs;
        pair_idx += (
            static_cast<int64_t>(gridDim.x)
            * kWarpsPerBlock
        )
    ) {
        const int64_t feature_idx = pair_idx % n_features;
        const int n_real_bins = n_bins[feature_idx];

        int logical_bin = 0;
        int is_internal_boundary = 0;
        float current_inv_width = 1.0f;
        float previous_inv_width = 1.0f;

        if (lane_idx == 0) {
            const float value = x[pair_idx];
            logical_bin = static_cast<int>(
                bin_indices[pair_idx]
            );
            const float left_edge = bin_edges[
                feature_idx * (max_n_bins + 1)
                + logical_bin
            ];
            current_inv_width = inv_bin_widths[
                feature_idx * max_n_bins
                + logical_bin
            ];
            is_internal_boundary = (
                logical_bin > 0
                && value == left_edge
            );
            if (is_internal_boundary) {
                previous_inv_width = inv_bin_widths[
                    feature_idx * max_n_bins
                    + logical_bin - 1
                ];
            }
        }

        logical_bin = __shfl_sync(
            0xffffffffu,
            logical_bin,
            0
        );
        is_internal_boundary = __shfl_sync(
            0xffffffffu,
            is_internal_boundary,
            0
        );
        current_inv_width = __shfl_sync(
            0xffffffffu,
            current_inv_width,
            0
        );
        previous_inv_width = __shfl_sync(
            0xffffffffu,
            previous_inv_width,
            0
        );

        const int current_physical_bin = physical_bin_index(
            logical_bin,
            n_real_bins,
            static_cast<int>(max_n_bins)
        );
        float grad_x_partial = 0.0f;

        for (
            int64_t d = lane_idx;
            d < d_embedding;
            d += kWarpSize
        ) {
            const float grad_value = load_as_float<grad_t>(
                grad_output + pair_idx * d_embedding + d
            );
            const float current_weight = (
                quantize_to_float<grad_t>(
                    weight[
                        (
                            feature_idx * max_n_bins
                            + current_physical_bin
                        )
                        * d_embedding
                        + d
                    ]
                )
            );
            grad_x_partial += (
                grad_value
                * current_weight
                * current_inv_width
            );

            if (is_internal_boundary) {
                const float previous_weight = (
                    quantize_to_float<grad_t>(
                        weight[
                            (
                                feature_idx * max_n_bins
                                + logical_bin - 1
                            )
                            * d_embedding
                            + d
                        ]
                    )
                );
                grad_x_partial += (
                    grad_value
                    * previous_weight
                    * previous_inv_width
                );
            }
        }

        grad_x_partial = group_reduce_sum<kWarpSize>(
            grad_x_partial
        );
        if (lane_idx == 0) {
            grad_x[pair_idx] = grad_x_partial;
        }
    }
}


// Correct fallback for shapes whose private FP32 histograms do not fit in
// dynamic shared memory. This is not expected for the target B <= 96 cases.
template <typename grad_t, typename bin_index_t>
__global__ void ple_backward_generic_weight_kernel(
    const grad_t* __restrict__ grad_output,
    const float* __restrict__ x,
    const float* __restrict__ bin_edges,
    const float* __restrict__ inv_bin_widths,
    const bin_index_t* __restrict__ bin_indices,
    float* __restrict__ bucket_sum,
    float* __restrict__ weighted_sum,
    int64_t n_features,
    int64_t max_n_bins,
    int64_t d_embedding,
    int64_t total_pairs
) {
    constexpr int kWarpsPerBlock = (
        kBlockThreads / kWarpSize
    );

    const int warp_idx = threadIdx.x / kWarpSize;
    const int lane_idx = threadIdx.x % kWarpSize;

    for (
        int64_t pair_idx = (
            static_cast<int64_t>(blockIdx.x)
            * kWarpsPerBlock
            + warp_idx
        );
        pair_idx < total_pairs;
        pair_idx += (
            static_cast<int64_t>(gridDim.x)
            * kWarpsPerBlock
        )
    ) {
        const int64_t feature_idx = pair_idx % n_features;

        int logical_bin = 0;
        float quantized_position = 0.0f;

        if (lane_idx == 0) {
            logical_bin = static_cast<int>(
                bin_indices[pair_idx]
            );
            const float value = x[pair_idx];
            const float left_edge = bin_edges[
                feature_idx * (max_n_bins + 1)
                + logical_bin
            ];
            const float current_inv_width = inv_bin_widths[
                feature_idx * max_n_bins
                + logical_bin
            ];
            const float position = (
                value - left_edge
            ) * current_inv_width;
            quantized_position = (
                quantize_to_float<grad_t>(position)
            );
        }

        logical_bin = __shfl_sync(
            0xffffffffu,
            logical_bin,
            0
        );
        quantized_position = __shfl_sync(
            0xffffffffu,
            quantized_position,
            0
        );

        for (
            int64_t d = lane_idx;
            d < d_embedding;
            d += kWarpSize
        ) {
            const float grad_value = load_as_float<grad_t>(
                grad_output + pair_idx * d_embedding + d
            );
            const int64_t output_index = (
                (
                    feature_idx * max_n_bins
                    + logical_bin
                )
                * d_embedding
                + d
            );
            atomicAdd(
                bucket_sum + output_index,
                grad_value
            );
            atomicAdd(
                weighted_sum + output_index,
                quantized_position * grad_value
            );
        }
    }
}



template <typename bin_index_t>
__global__ void build_sorted_keys_kernel(
    const bin_index_t* __restrict__ bin_indices,
    int32_t* __restrict__ keys,
    int32_t* __restrict__ pair_ids,
    int64_t n_features,
    int64_t max_n_bins,
    int64_t total_pairs
) {
    for (
        int64_t pair_idx = (
            static_cast<int64_t>(blockIdx.x)
            * blockDim.x
            + threadIdx.x
        );
        pair_idx < total_pairs;
        pair_idx += (
            static_cast<int64_t>(gridDim.x)
            * blockDim.x
        )
    ) {
        const int64_t feature_idx = pair_idx % n_features;
        const int32_t logical_bin = static_cast<int32_t>(
            bin_indices[pair_idx]
        );

        keys[pair_idx] = static_cast<int32_t>(
            feature_idx * max_n_bins + logical_bin
        );
        pair_ids[pair_idx] = static_cast<int32_t>(pair_idx);
    }
}


// Build offsets for the complete dense key range [0, F * B].
// This avoids dynamic unique-key output and host synchronization.
__global__ void build_dense_segment_offsets_kernel(
    const int32_t* __restrict__ sorted_keys,
    int32_t* __restrict__ segment_offsets,
    int32_t total_pairs,
    int32_t n_keys
) {
    for (
        int32_t key = (
            static_cast<int32_t>(blockIdx.x)
            * blockDim.x
            + threadIdx.x
        );
        key <= n_keys;
        key += static_cast<int32_t>(
            gridDim.x * blockDim.x
        )
    ) {
        int32_t low = 0;
        int32_t high = total_pairs;

        while (low < high) {
            const int32_t middle = low + (high - low) / 2;

            if (sorted_keys[middle] < key) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }

        segment_offsets[key] = low;
    }
}


// One block owns one (feature-bin key, 64-dimensional tile).
// Four warps split the sorted segment. Each lane accumulates one or two
// embedding dimensions in FP32. The warp partials are merged in shared
// memory, so this path has no floating-point atomics.
template <typename grad_t>
__global__ void sorted_segment_reduce_kernel(
    const grad_t* __restrict__ grad_output,
    const float* __restrict__ x,
    const float* __restrict__ bin_edges,
    const float* __restrict__ inv_bin_widths,
    const int32_t* __restrict__ sorted_pair_ids,
    const int32_t* __restrict__ segment_offsets,
    float* __restrict__ bucket_sum,
    float* __restrict__ weighted_sum,
    int64_t max_n_bins,
    int64_t d_embedding,
    int64_t n_d_tiles,
    int64_t total_work_items
) {
    __shared__ float partial_bucket[
        kSortedReductionWarps
        * kSortedDPerBlock
    ];
    __shared__ float partial_weighted[
        kSortedReductionWarps
        * kSortedDPerBlock
    ];

    const int warp_idx = threadIdx.x / kWarpSize;
    const int lane_idx = threadIdx.x % kWarpSize;

    for (
        int64_t work_item = blockIdx.x;
        work_item < total_work_items;
        work_item += gridDim.x
    ) {
        const int64_t d_tile = work_item % n_d_tiles;
        const int64_t key = work_item / n_d_tiles;

        const int64_t feature_idx = key / max_n_bins;
        const int logical_bin = static_cast<int>(
            key - feature_idx * max_n_bins
        );

        const int32_t segment_begin = segment_offsets[key];
        const int32_t segment_end = segment_offsets[key + 1];

        const int64_t d0 = (
            d_tile * kSortedDPerBlock + lane_idx
        );
        const int64_t d1 = d0 + kWarpSize;

        float local_bucket0 = 0.0f;
        float local_weighted0 = 0.0f;
        float local_bucket1 = 0.0f;
        float local_weighted1 = 0.0f;

        for (
            int32_t sorted_idx = segment_begin + warp_idx;
            sorted_idx < segment_end;
            sorted_idx += kSortedReductionWarps
        ) {
            const int32_t pair_idx = sorted_pair_ids[sorted_idx];

            float quantized_position = 0.0f;
            if (lane_idx == 0) {
                const float value = x[pair_idx];
                const float left_edge = bin_edges[
                    feature_idx * (max_n_bins + 1)
                    + logical_bin
                ];
                const float current_inv_width = inv_bin_widths[
                    feature_idx * max_n_bins
                    + logical_bin
                ];
                const float position = (
                    value - left_edge
                ) * current_inv_width;
                quantized_position = (
                    quantize_to_float<grad_t>(position)
                );
            }

            quantized_position = __shfl_sync(
                0xffffffffu,
                quantized_position,
                0
            );

            const int64_t output_base = (
                static_cast<int64_t>(pair_idx)
                * d_embedding
            );

            if (d0 < d_embedding) {
                const float grad_value0 = load_as_float<grad_t>(
                    grad_output + output_base + d0
                );
                local_bucket0 += grad_value0;
                local_weighted0 += (
                    quantized_position * grad_value0
                );
            }

            if (d1 < d_embedding) {
                const float grad_value1 = load_as_float<grad_t>(
                    grad_output + output_base + d1
                );
                local_bucket1 += grad_value1;
                local_weighted1 += (
                    quantized_position * grad_value1
                );
            }
        }

        const int warp_partial_base = (
            warp_idx * kSortedDPerBlock
        );
        partial_bucket[
            warp_partial_base + lane_idx
        ] = local_bucket0;
        partial_weighted[
            warp_partial_base + lane_idx
        ] = local_weighted0;
        partial_bucket[
            warp_partial_base + kWarpSize + lane_idx
        ] = local_bucket1;
        partial_weighted[
            warp_partial_base + kWarpSize + lane_idx
        ] = local_weighted1;

        __syncthreads();

        if (warp_idx == 0) {
            if (d0 < d_embedding) {
                float bucket0 = 0.0f;
                float weighted0 = 0.0f;

                #pragma unroll
                for (
                    int partial_warp = 0;
                    partial_warp < kSortedReductionWarps;
                    ++partial_warp
                ) {
                    const int partial_index = (
                        partial_warp * kSortedDPerBlock
                        + lane_idx
                    );
                    bucket0 += partial_bucket[partial_index];
                    weighted0 += partial_weighted[partial_index];
                }

                const int64_t output_index0 = (
                    key * d_embedding + d0
                );
                bucket_sum[output_index0] = bucket0;
                weighted_sum[output_index0] = weighted0;
            }

            if (d1 < d_embedding) {
                float bucket1 = 0.0f;
                float weighted1 = 0.0f;

                #pragma unroll
                for (
                    int partial_warp = 0;
                    partial_warp < kSortedReductionWarps;
                    ++partial_warp
                ) {
                    const int partial_index = (
                        partial_warp * kSortedDPerBlock
                        + kWarpSize
                        + lane_idx
                    );
                    bucket1 += partial_bucket[partial_index];
                    weighted1 += partial_weighted[partial_index];
                }

                const int64_t output_index1 = (
                    key * d_embedding + d1
                );
                bucket_sum[output_index1] = bucket1;
                weighted_sum[output_index1] = weighted1;
            }
        }

        __syncthreads();
    }
}


__global__ void finalize_grad_weight_kernel(
    const int32_t* __restrict__ n_bins,
    const float* __restrict__ bucket_sum,
    const float* __restrict__ weighted_sum,
    float* __restrict__ grad_weight,
    int64_t n_features,
    int64_t max_n_bins,
    int64_t d_embedding,
    int64_t total_feature_dimensions
) {
    for (
        int64_t feature_dimension = (
            static_cast<int64_t>(blockIdx.x)
            * blockDim.x
            + threadIdx.x
        );
        feature_dimension < total_feature_dimensions;
        feature_dimension += (
            static_cast<int64_t>(gridDim.x)
            * blockDim.x
        )
    ) {
        const int64_t feature_idx = (
            feature_dimension / d_embedding
        );
        const int64_t d = (
            feature_dimension % d_embedding
        );
        const int n_real_bins = n_bins[feature_idx];

        float future_bucket_sum = 0.0f;

        for (
            int logical_bin = n_real_bins - 1;
            logical_bin >= 0;
            --logical_bin
        ) {
            const int64_t logical_index = (
                (
                    feature_idx * max_n_bins
                    + logical_bin
                )
                * d_embedding
                + d
            );
            const int physical_bin = physical_bin_index(
                logical_bin,
                n_real_bins,
                static_cast<int>(max_n_bins)
            );
            const int64_t physical_index = (
                (
                    feature_idx * max_n_bins
                    + physical_bin
                )
                * d_embedding
                + d
            );

            grad_weight[physical_index] = (
                weighted_sum[logical_index]
                + future_bucket_sum
            );
            future_bucket_sum += bucket_sum[logical_index];
        }
    }
}



template <typename grad_t, typename bin_index_t>
void launch_sorted_segment_weight_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& bin_edges,
    const at::Tensor& inv_bin_widths,
    const at::Tensor& n_bins,
    const at::Tensor& weight,
    const at::Tensor& bin_indices,
    at::Tensor& grad_weight,
    int64_t total_pairs,
    cudaStream_t stream
) {
    const int64_t n_features = x.size(1);
    const int64_t max_n_bins = weight.size(1);
    const int64_t d_embedding = weight.size(2);

    TORCH_CHECK(
        n_features <= (
            std::numeric_limits<int64_t>::max()
            / max_n_bins
        ),
        "n_features * max_n_bins overflows int64"
    );
    const int64_t n_keys = n_features * max_n_bins;

    TORCH_CHECK(
        total_pairs <= std::numeric_limits<int32_t>::max(),
        "The sorted backward supports at most INT_MAX feature pairs"
    );
    TORCH_CHECK(
        n_keys <= std::numeric_limits<int32_t>::max(),
        "The sorted backward supports at most INT_MAX feature-bin keys"
    );

    const auto int_options = x.options().dtype(at::kInt);

    at::Tensor keys = at::empty(
        {total_pairs},
        int_options
    );
    at::Tensor pair_ids = at::empty(
        {total_pairs},
        int_options
    );
    at::Tensor sorted_keys = at::empty(
        {total_pairs},
        int_options
    );
    at::Tensor sorted_pair_ids = at::empty(
        {total_pairs},
        int_options
    );

    const int build_blocks = static_cast<int>(
        std::min<int64_t>(
            ceil_div<int64_t>(
                total_pairs,
                kBlockThreads
            ),
            kMaxLaunchBlocks
        )
    );

    build_sorted_keys_kernel<
        bin_index_t
    ><<<
        build_blocks,
        kBlockThreads,
        0,
        stream
    >>>(
        bin_indices.data_ptr<bin_index_t>(),
        keys.data_ptr<int32_t>(),
        pair_ids.data_ptr<int32_t>(),
        n_features,
        max_n_bins,
        total_pairs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const int sort_bits = at::cuda::cub::get_num_bits(
        static_cast<uint64_t>(
            std::max<int64_t>(1, n_keys - 1)
        )
    );

    at::cuda::cub::radix_sort_pairs<int32_t, int32_t>(
        keys.data_ptr<int32_t>(),
        sorted_keys.data_ptr<int32_t>(),
        pair_ids.data_ptr<int32_t>(),
        sorted_pair_ids.data_ptr<int32_t>(),
        total_pairs,
        false,
        0,
        sort_bits
    );

    at::Tensor segment_offsets = at::empty(
        {n_keys + 1},
        int_options
    );

    const int offset_blocks = static_cast<int>(
        std::min<int64_t>(
            ceil_div<int64_t>(
                n_keys + 1,
                kBlockThreads
            ),
            kMaxLaunchBlocks
        )
    );

    build_dense_segment_offsets_kernel<<<
        offset_blocks,
        kBlockThreads,
        0,
        stream
    >>>(
        sorted_keys.data_ptr<int32_t>(),
        segment_offsets.data_ptr<int32_t>(),
        static_cast<int32_t>(total_pairs),
        static_cast<int32_t>(n_keys)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    at::Tensor bucket_sum = at::empty_like(weight);
    at::Tensor weighted_sum = at::empty_like(weight);

    const int64_t n_d_tiles = ceil_div<int64_t>(
        d_embedding,
        kSortedDPerBlock
    );
    TORCH_CHECK(
        n_keys <= (
            std::numeric_limits<int64_t>::max()
            / n_d_tiles
        ),
        "The sorted backward work-item count overflows int64"
    );
    const int64_t total_work_items = n_keys * n_d_tiles;

    const int reduction_blocks = static_cast<int>(
        std::min<int64_t>(
            total_work_items,
            kMaxLaunchBlocks
        )
    );

    sorted_segment_reduce_kernel<
        grad_t
    ><<<
        reduction_blocks,
        kSortedReductionThreads,
        0,
        stream
    >>>(
        grad_output.data_ptr<grad_t>(),
        x.data_ptr<float>(),
        bin_edges.data_ptr<float>(),
        inv_bin_widths.data_ptr<float>(),
        sorted_pair_ids.data_ptr<int32_t>(),
        segment_offsets.data_ptr<int32_t>(),
        bucket_sum.data_ptr<float>(),
        weighted_sum.data_ptr<float>(),
        max_n_bins,
        d_embedding,
        n_d_tiles,
        total_work_items
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    TORCH_CHECK(
        n_features <= (
            std::numeric_limits<int64_t>::max()
            / d_embedding
        ),
        "n_features * d_embedding overflows int64"
    );
    const int64_t total_feature_dimensions = (
        n_features * d_embedding
    );
    const int finalize_blocks = static_cast<int>(
        std::min<int64_t>(
            ceil_div<int64_t>(
                total_feature_dimensions,
                kBlockThreads
            ),
            kMaxLaunchBlocks
        )
    );

    finalize_grad_weight_kernel<<<
        finalize_blocks,
        kBlockThreads,
        0,
        stream
    >>>(
        n_bins.data_ptr<int32_t>(),
        bucket_sum.data_ptr<float>(),
        weighted_sum.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        n_features,
        max_n_bins,
        d_embedding,
        total_feature_dimensions
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}



template <typename grad_t, typename bin_index_t>
void launch_ple_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& bin_edges,
    const at::Tensor& inv_bin_widths,
    const at::Tensor& n_bins,
    const at::Tensor& weight,
    const at::Tensor& bin_indices,
    at::Tensor& grad_x,
    at::Tensor& grad_weight,
    bool compute_grad_x,
    bool compute_grad_weight
) {
    const int64_t batch_size = x.size(0);
    const int64_t n_features = x.size(1);
    const int64_t max_n_bins = weight.size(1);
    const int64_t d_embedding = weight.size(2);

    if (!compute_grad_x && !compute_grad_weight) {
        return;
    }

    TORCH_CHECK(
        n_features == 0
        || batch_size <= (
            std::numeric_limits<int64_t>::max()
            / n_features
        ),
        "batch_size * n_features overflows int64"
    );
    const int64_t total_pairs = batch_size * n_features;

    const int device_index = x.get_device();
    const cudaStream_t stream = (
        c10::cuda::getCurrentCUDAStream(device_index).stream()
    );

    if (compute_grad_x && total_pairs > 0) {
        constexpr int kWarpsPerBlock = (
            kBlockThreads / kWarpSize
        );
        const int grad_x_blocks = static_cast<int>(
            std::min<int64_t>(
                ceil_div<int64_t>(
                    total_pairs,
                    kWarpsPerBlock
                ),
                kMaxLaunchBlocks
            )
        );

        ple_backward_grad_x_kernel<
            grad_t,
            bin_index_t
        ><<<
            grad_x_blocks,
            kBlockThreads,
            0,
            stream
        >>>(
            grad_output.data_ptr<grad_t>(),
            x.data_ptr<float>(),
            bin_edges.data_ptr<float>(),
            inv_bin_widths.data_ptr<float>(),
            n_bins.data_ptr<int32_t>(),
            weight.data_ptr<float>(),
            bin_indices.data_ptr<bin_index_t>(),
            grad_x.data_ptr<float>(),
            n_features,
            max_n_bins,
            d_embedding,
            total_pairs
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (
        !compute_grad_weight
        || total_pairs == 0
        || n_features == 0
    ) {
        return;
    }


    // Private histogram is faster for small embedding dimensions and for
    // small numbers of (sample, feature) pairs, where radix-sort overhead
    // does not have enough work to amortize.
    const bool use_small_shape_path = (
        d_embedding < kSortedMinEmbeddingDim
        || total_pairs < kSortedMinPairs
    );

    if (use_small_shape_path) {
        const int tile_d = 16;
        const int64_t n_d_tiles = ceil_div<int64_t>(
            d_embedding,
            tile_d
        );

        TORCH_CHECK(
            n_features <= (
                std::numeric_limits<int64_t>::max()
                / n_d_tiles
            ),
            "n_features * n_d_tiles overflows int64"
        );
        const int64_t feature_d_tiles = (
            n_features * n_d_tiles
        );

        // Two FP32 histograms:
        // [private warp][logical bin][physical warp lane].
        const size_t shared_bytes = (
            static_cast<size_t>(2)
            * static_cast<size_t>(
                kPrivateBlockThreads / kWarpSize
            )
            * static_cast<size_t>(max_n_bins)
            * static_cast<size_t>(kWarpSize)
            * sizeof(float)
        );

        int max_optin_shared_bytes = 0;
        C10_CUDA_CHECK(
            cudaDeviceGetAttribute(
                &max_optin_shared_bytes,
                cudaDevAttrMaxSharedMemoryPerBlockOptin,
                device_index
            )
        );

        const bool use_private_histogram = (
            shared_bytes
            <= static_cast<size_t>(max_optin_shared_bytes)
            && shared_bytes
            <= static_cast<size_t>(
                std::numeric_limits<int>::max()
            )
        );

        if (use_private_histogram) {
            const int64_t desired_partitions = std::max<int64_t>(
                1,
                ceil_div<int64_t>(
                    kTargetPersistentBlocks,
                    feature_d_tiles
                )
            );
            const int64_t partitions_by_batch = std::max<int64_t>(
                1,
                batch_size / kMinSamplesPerPartition
            );
            const int64_t n_partitions = std::min<int64_t>(
                kMaxBatchPartitions,
                std::min<int64_t>(
                    desired_partitions,
                    partitions_by_batch
                )
            );

            TORCH_CHECK(
                feature_d_tiles <= (
                    std::numeric_limits<int64_t>::max()
                    / n_partitions
                ),
                "The private backward tile count overflows int64"
            );
            const int64_t total_persistent_tiles = (
                feature_d_tiles * n_partitions
            );
            const int launch_blocks = static_cast<int>(
                std::min<int64_t>(
                    total_persistent_tiles,
                    kMaxLaunchBlocks
                )
            );

            C10_CUDA_CHECK(
                cudaFuncSetAttribute(
                    ple_backward_private_histogram_kernel<
                        grad_t,
                        bin_index_t,
                        16
                    >,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes)
                )
            );
            ple_backward_private_histogram_kernel<
                grad_t,
                bin_index_t,
                16
            ><<<
                launch_blocks,
                kPrivateBlockThreads,
                shared_bytes,
                stream
            >>>(
                grad_output.data_ptr<grad_t>(),
                x.data_ptr<float>(),
                bin_edges.data_ptr<float>(),
                inv_bin_widths.data_ptr<float>(),
                n_bins.data_ptr<int32_t>(),
                bin_indices.data_ptr<bin_index_t>(),
                grad_weight.data_ptr<float>(),
                batch_size,
                n_features,
                max_n_bins,
                d_embedding,
                n_d_tiles,
                n_partitions,
                total_persistent_tiles
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }

    if (
        total_pairs <= std::numeric_limits<int32_t>::max()
        && n_features <= (
            std::numeric_limits<int32_t>::max()
            / max_n_bins
        )
    ) {
        launch_sorted_segment_weight_backward<
            grad_t,
            bin_index_t
        >(
            grad_output,
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            bin_indices,
            grad_weight,
            total_pairs,
            stream
        );
        return;
    }

    // Very-large-shape fallback. The target benchmark shapes use the sorted
    // path above.
    at::Tensor bucket_sum = at::zeros_like(weight);
    at::Tensor weighted_sum = at::zeros_like(weight);

    constexpr int kWarpsPerBlock = (
        kBlockThreads / kWarpSize
    );
    const int generic_blocks = static_cast<int>(
        std::min<int64_t>(
            ceil_div<int64_t>(
                total_pairs,
                kWarpsPerBlock
            ),
            kMaxLaunchBlocks
        )
    );

    ple_backward_generic_weight_kernel<
        grad_t,
        bin_index_t
    ><<<
        generic_blocks,
        kBlockThreads,
        0,
        stream
    >>>(
        grad_output.data_ptr<grad_t>(),
        x.data_ptr<float>(),
        bin_edges.data_ptr<float>(),
        inv_bin_widths.data_ptr<float>(),
        bin_indices.data_ptr<bin_index_t>(),
        bucket_sum.data_ptr<float>(),
        weighted_sum.data_ptr<float>(),
        n_features,
        max_n_bins,
        d_embedding,
        total_pairs
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    TORCH_CHECK(
        n_features <= (
            std::numeric_limits<int64_t>::max()
            / d_embedding
        ),
        "n_features * d_embedding overflows int64"
    );
    const int64_t total_feature_dimensions = (
        n_features * d_embedding
    );
    const int finalize_blocks = static_cast<int>(
        std::min<int64_t>(
            ceil_div<int64_t>(
                total_feature_dimensions,
                kBlockThreads
            ),
            kMaxLaunchBlocks
        )
    );

    finalize_grad_weight_kernel<<<
        finalize_blocks,
        kBlockThreads,
        0,
        stream
    >>>(
        n_bins.data_ptr<int32_t>(),
        bucket_sum.data_ptr<float>(),
        weighted_sum.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        n_features,
        max_n_bins,
        d_embedding,
        total_feature_dimensions
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <typename grad_t>
void dispatch_bin_index_type_backward(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& bin_edges,
    const at::Tensor& inv_bin_widths,
    const at::Tensor& n_bins,
    const at::Tensor& weight,
    const at::Tensor& bin_indices,
    at::Tensor& grad_x,
    at::Tensor& grad_weight,
    bool compute_grad_x,
    bool compute_grad_weight
) {
    if (bin_indices.scalar_type() == at::kByte) {
        launch_ple_backward<grad_t, uint8_t>(
            grad_output,
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            bin_indices,
            grad_x,
            grad_weight,
            compute_grad_x,
            compute_grad_weight
        );
    } else if (bin_indices.scalar_type() == at::kShort) {
        launch_ple_backward<grad_t, int16_t>(
            grad_output,
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            bin_indices,
            grad_x,
            grad_weight,
            compute_grad_x,
            compute_grad_weight
        );
    } else {
        launch_ple_backward<grad_t, int32_t>(
            grad_output,
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            bin_indices,
            grad_x,
            grad_weight,
            compute_grad_x,
            compute_grad_weight
        );
    }
}

}  // namespace rtdl_num_embeddings_cuda


std::tuple<at::Tensor, at::Tensor> ple_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& x,
    const at::Tensor& bin_edges,
    const at::Tensor& inv_bin_widths,
    const at::Tensor& n_bins,
    const at::Tensor& weight,
    const at::Tensor& bin_indices,
    bool compute_grad_x,
    bool compute_grad_weight
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bin_edges.is_cuda(), "bin_edges must be CUDA");
    TORCH_CHECK(
        inv_bin_widths.is_cuda(),
        "inv_bin_widths must be CUDA"
    );
    TORCH_CHECK(n_bins.is_cuda(), "n_bins must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bin_indices.is_cuda(), "bin_indices must be CUDA");
    TORCH_CHECK(
        grad_output.device() == x.device()
        && x.device() == bin_edges.device()
        && x.device() == inv_bin_widths.device()
        && x.device() == n_bins.device()
        && x.device() == weight.device()
        && x.device() == bin_indices.device(),
        "All backward tensors must be on the same CUDA device"
    );

    TORCH_CHECK(
        grad_output.is_contiguous(),
        "grad_output must be contiguous"
    );
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bin_edges.is_contiguous(), "bin_edges must be contiguous");
    TORCH_CHECK(
        inv_bin_widths.is_contiguous(),
        "inv_bin_widths must be contiguous"
    );
    TORCH_CHECK(n_bins.is_contiguous(), "n_bins must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(
        bin_indices.is_contiguous(),
        "bin_indices must be contiguous"
    );

    TORCH_CHECK(x.dim() == 2, "x must have shape [N, F]");
    TORCH_CHECK(
        grad_output.dim() == 3,
        "grad_output must have shape [N, F, D]"
    );
    TORCH_CHECK(
        bin_edges.dim() == 2,
        "bin_edges must have shape [F, B+1]"
    );
    TORCH_CHECK(
        inv_bin_widths.dim() == 2,
        "inv_bin_widths must have shape [F, B]"
    );
    TORCH_CHECK(n_bins.dim() == 1, "n_bins must have shape [F]");
    TORCH_CHECK(weight.dim() == 3, "weight must have shape [F, B, D]");
    TORCH_CHECK(
        bin_indices.dim() == 2,
        "bin_indices must have shape [N, F]"
    );

    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
    TORCH_CHECK(
        bin_edges.scalar_type() == at::kFloat,
        "bin_edges must be float32"
    );
    TORCH_CHECK(
        inv_bin_widths.scalar_type() == at::kFloat,
        "inv_bin_widths must be float32"
    );
    TORCH_CHECK(n_bins.scalar_type() == at::kInt, "n_bins must be int32");
    TORCH_CHECK(weight.scalar_type() == at::kFloat, "weight must be float32");
    TORCH_CHECK(
        grad_output.scalar_type() == at::kFloat
        || grad_output.scalar_type() == at::kHalf
        || grad_output.scalar_type() == at::kBFloat16,
        "grad_output must be float32, float16, or bfloat16"
    );

    const int64_t batch_size = x.size(0);
    const int64_t n_features = x.size(1);
    const int64_t max_n_bins = weight.size(1);
    const int64_t d_embedding = weight.size(2);
    const at::ScalarType expected_bin_index_dtype = (
        max_n_bins <= 256
            ? at::kByte
            : (
                max_n_bins <= 32768
                    ? at::kShort
                    : at::kInt
            )
    );

    TORCH_CHECK(
        bin_indices.scalar_type() == expected_bin_index_dtype,
        "bin_indices has an invalid dtype"
    );
    TORCH_CHECK(
        grad_output.size(0) == batch_size
        && grad_output.size(1) == n_features
        && grad_output.size(2) == d_embedding,
        "grad_output shape must be [N, F, D]"
    );
    TORCH_CHECK(
        bin_indices.size(0) == batch_size
        && bin_indices.size(1) == n_features,
        "bin_indices shape must be [N, F]"
    );
    TORCH_CHECK(weight.size(0) == n_features, "weight F mismatch");
    TORCH_CHECK(bin_edges.size(0) == n_features, "bin_edges F mismatch");
    TORCH_CHECK(
        bin_edges.size(1) == max_n_bins + 1,
        "bin_edges B mismatch"
    );
    TORCH_CHECK(
        inv_bin_widths.size(0) == n_features,
        "inv_bin_widths F mismatch"
    );
    TORCH_CHECK(
        inv_bin_widths.size(1) == max_n_bins,
        "inv_bin_widths B mismatch"
    );
    TORCH_CHECK(n_bins.size(0) == n_features, "n_bins F mismatch");

    c10::cuda::CUDAGuard device_guard(x.device());

    at::Tensor grad_x = (
        compute_grad_x
            ? at::zeros_like(x)
            : at::empty({0}, x.options())
    );
    at::Tensor grad_weight = (
        compute_grad_weight
            ? at::zeros_like(weight)
            : at::empty({0}, weight.options())
    );

    if (!compute_grad_x && !compute_grad_weight) {
        return {grad_x, grad_weight};
    }

    if (grad_output.scalar_type() == at::kFloat) {
        rtdl_num_embeddings_cuda::dispatch_bin_index_type_backward<float>(
            grad_output,
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            bin_indices,
            grad_x,
            grad_weight,
            compute_grad_x,
            compute_grad_weight
        );
    } else if (grad_output.scalar_type() == at::kHalf) {
        rtdl_num_embeddings_cuda::dispatch_bin_index_type_backward<
            c10::Half
        >(
            grad_output,
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            bin_indices,
            grad_x,
            grad_weight,
            compute_grad_x,
            compute_grad_weight
        );
    } else {
        rtdl_num_embeddings_cuda::dispatch_bin_index_type_backward<
            c10::BFloat16
        >(
            grad_output,
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            bin_indices,
            grad_x,
            grad_weight,
            compute_grad_x,
            compute_grad_weight
        );
    }

    return {grad_x, grad_weight};
}
