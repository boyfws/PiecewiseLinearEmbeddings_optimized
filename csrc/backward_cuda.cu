#include "common.cuh"

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


template <typename grad_t, typename bin_index_t, int kTileD>
__global__ void ple_backward_shared_kernel(
    const grad_t* __restrict__ grad_output,
    const float* __restrict__ x,
    const float* __restrict__ bin_edges,
    const float* __restrict__ inv_bin_widths,
    const int32_t* __restrict__ n_bins,
    const float* __restrict__ weight,
    const bin_index_t* __restrict__ bin_indices,
    float* __restrict__ grad_x,
    float* __restrict__ bucket_sum,
    float* __restrict__ weighted_sum,
    int64_t batch_size,
    int64_t n_features,
    int64_t max_n_bins,
    int64_t d_embedding,
    int64_t n_batch_tiles,
    int64_t n_d_tiles,
    int64_t n_partitions,
    int64_t total_persistent_tiles,
    bool compute_grad_x,
    bool compute_grad_weight
) {
    extern __shared__ float shared_memory[];

    float* shared_edges = shared_memory;
    float* shared_inv_bin_widths = (
        shared_edges + max_n_bins + 1
    );
    float* shared_bucket = (
        shared_inv_bin_widths + max_n_bins
    );
    float* shared_weighted = (
        shared_bucket
        + (
            compute_grad_weight
                ? max_n_bins * kTileD
                : 0
        )
    );

    constexpr int kGroupsPerBlock = (
        kBlockThreads / kTileD
    );

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
        const int64_t edge_row_offset = (
            feature_idx * (max_n_bins + 1)
        );
        const int64_t width_row_offset = (
            feature_idx * max_n_bins
        );

        for (
            int edge_idx = threadIdx.x;
            edge_idx < n_real_bins + 1;
            edge_idx += blockDim.x
        ) {
            shared_edges[edge_idx] = (
                bin_edges[edge_row_offset + edge_idx]
            );
        }
        for (
            int logical_bin = threadIdx.x;
            logical_bin < n_real_bins;
            logical_bin += blockDim.x
        ) {
            shared_inv_bin_widths[logical_bin] = (
                inv_bin_widths[
                    width_row_offset + logical_bin
                ]
            );
        }

        if (compute_grad_weight) {
            const int shared_elements = (
                n_real_bins * kTileD
            );
            for (
                int index = threadIdx.x;
                index < shared_elements;
                index += blockDim.x
            ) {
                shared_bucket[index] = 0.0f;
                shared_weighted[index] = 0.0f;
            }
        }

        __syncthreads();

        const int group_idx = threadIdx.x / kTileD;
        const int lane_idx = threadIdx.x % kTileD;
        const int64_t d_global = (
            d_tile * kTileD + lane_idx
        );

        for (
            int64_t batch_tile = partition_idx;
            batch_tile < n_batch_tiles;
            batch_tile += n_partitions
        ) {
            const int64_t batch_base = (
                batch_tile * kSamplesPerBlock
            );

            for (
                int sample_offset = group_idx;
                sample_offset < kSamplesPerBlock;
                sample_offset += kGroupsPerBlock
            ) {
                const int64_t sample_idx = (
                    batch_base + sample_offset
                );

                int logical_bin = 0;
                int is_internal_boundary = 0;
                float position = 0.0f;
                float current_inv_width = 1.0f;
                float previous_inv_width = 1.0f;

                if (
                    lane_idx == 0
                    && sample_idx < batch_size
                ) {
                    const int64_t pair_idx = (
                        sample_idx * n_features
                        + feature_idx
                    );
                    const float value = x[pair_idx];
                    logical_bin = static_cast<int>(
                        bin_indices[pair_idx]
                    );
                    const float left_edge = (
                        shared_edges[logical_bin]
                    );
                    current_inv_width = (
                        shared_inv_bin_widths[logical_bin]
                    );
                    position = (
                        value - left_edge
                    ) * current_inv_width;

                    // At an exact internal edge, the original clamp-based PLE
                    // has a derivative contribution from both adjacent
                    // channels.
                    is_internal_boundary = (
                        logical_bin > 0
                        && value == left_edge
                    );
                    if (is_internal_boundary) {
                        previous_inv_width = (
                            shared_inv_bin_widths[
                                logical_bin - 1
                            ]
                        );
                    }
                }

                logical_bin = __shfl_sync(
                    0xffffffffu,
                    logical_bin,
                    0,
                    kTileD
                );
                is_internal_boundary = __shfl_sync(
                    0xffffffffu,
                    is_internal_boundary,
                    0,
                    kTileD
                );
                position = __shfl_sync(
                    0xffffffffu,
                    position,
                    0,
                    kTileD
                );
                current_inv_width = __shfl_sync(
                    0xffffffffu,
                    current_inv_width,
                    0,
                    kTileD
                );
                previous_inv_width = __shfl_sync(
                    0xffffffffu,
                    previous_inv_width,
                    0,
                    kTileD
                );

                float grad_x_partial = 0.0f;

                if (
                    sample_idx < batch_size
                    && d_global < d_embedding
                ) {
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
                    const float quantized_position = (
                        quantize_to_float<grad_t>(position)
                    );

                    if (compute_grad_weight) {
                        atomicAdd(
                            shared_bucket
                            + logical_bin * kTileD
                            + lane_idx,
                            grad_value
                        );
                        atomicAdd(
                            shared_weighted
                            + logical_bin * kTileD
                            + lane_idx,
                            quantized_position * grad_value
                        );
                    }

                    if (compute_grad_x) {
                        const int current_physical_bin = (
                            physical_bin_index(
                                logical_bin,
                                n_real_bins,
                                static_cast<int>(max_n_bins)
                            )
                        );
                        const int64_t current_weight_index = (
                            (
                                feature_idx * max_n_bins
                                + current_physical_bin
                            )
                            * d_embedding
                            + d_global
                        );
                        const float current_weight = (
                            quantize_to_float<grad_t>(
                                weight[current_weight_index]
                            )
                        );

                        grad_x_partial = (
                            grad_value
                            * current_weight
                            * current_inv_width
                        );

                        if (is_internal_boundary) {
                            const int previous_physical_bin = (
                                logical_bin - 1
                            );
                            const int64_t previous_weight_index = (
                                (
                                    feature_idx * max_n_bins
                                    + previous_physical_bin
                                )
                                * d_embedding
                                + d_global
                            );
                            const float previous_weight = (
                                quantize_to_float<grad_t>(
                                    weight[previous_weight_index]
                                )
                            );
                            grad_x_partial += (
                                grad_value
                                * previous_weight
                                * previous_inv_width
                            );
                        }
                    }
                }

                if (compute_grad_x) {
                    grad_x_partial = group_reduce_sum<kTileD>(
                        grad_x_partial
                    );

                    if (
                        lane_idx == 0
                        && sample_idx < batch_size
                    ) {
                        atomicAdd(
                            grad_x
                            + sample_idx * n_features
                            + feature_idx,
                            grad_x_partial
                        );
                    }
                }
            }
        }

        __syncthreads();

        // One global flush per persistent batch partition.
        if (compute_grad_weight) {
            const int shared_elements = (
                n_real_bins * kTileD
            );
            for (
                int index = threadIdx.x;
                index < shared_elements;
                index += blockDim.x
            ) {
                const int logical_bin = index / kTileD;
                const int d_local = index % kTileD;
                const int64_t d_global_for_flush = (
                    d_tile * kTileD + d_local
                );

                if (d_global_for_flush < d_embedding) {
                    const int64_t bucket_index = (
                        (
                            feature_idx * max_n_bins
                            + logical_bin
                        )
                        * d_embedding
                        + d_global_for_flush
                    );

                    atomicAdd(
                        bucket_sum + bucket_index,
                        shared_bucket[index]
                    );
                    atomicAdd(
                        weighted_sum + bucket_index,
                        shared_weighted[index]
                    );
                }
            }
        }

        __syncthreads();
    }
}


template <typename grad_t, typename bin_index_t>
__global__ void ple_backward_generic_kernel(
    const grad_t* __restrict__ grad_output,
    const float* __restrict__ x,
    const float* __restrict__ bin_edges,
    const float* __restrict__ inv_bin_widths,
    const int32_t* __restrict__ n_bins,
    const float* __restrict__ weight,
    const bin_index_t* __restrict__ bin_indices,
    float* __restrict__ grad_x,
    float* __restrict__ bucket_sum,
    float* __restrict__ weighted_sum,
    int64_t n_features,
    int64_t max_n_bins,
    int64_t d_embedding,
    int64_t total_pairs,
    bool compute_grad_x,
    bool compute_grad_weight
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
        const float* feature_edges = (
            bin_edges
            + feature_idx * (max_n_bins + 1)
        );
        const float* feature_inv_bin_widths = (
            inv_bin_widths
            + feature_idx * max_n_bins
        );

        int logical_bin = 0;
        int is_internal_boundary = 0;
        float position = 0.0f;
        float current_inv_width = 1.0f;
        float previous_inv_width = 1.0f;

        if (lane_idx == 0) {
            const float value = x[pair_idx];
            logical_bin = static_cast<int>(
                bin_indices[pair_idx]
            );
            const float left_edge = feature_edges[logical_bin];
            current_inv_width = (
                feature_inv_bin_widths[logical_bin]
            );
            position = (
                value - left_edge
            ) * current_inv_width;
            is_internal_boundary = (
                logical_bin > 0
                && value == left_edge
            );
            if (is_internal_boundary) {
                previous_inv_width = (
                    feature_inv_bin_widths[logical_bin - 1]
                );
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
        position = __shfl_sync(
            0xffffffffu,
            position,
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
        const float quantized_position = (
            quantize_to_float<grad_t>(position)
        );
        float grad_x_partial = 0.0f;

        for (
            int64_t d = lane_idx;
            d < d_embedding;
            d += kWarpSize
        ) {
            const int64_t output_index = (
                pair_idx * d_embedding + d
            );
            const float grad_value = load_as_float<grad_t>(
                grad_output + output_index
            );

            if (compute_grad_weight) {
                const int64_t bucket_index = (
                    (
                        feature_idx * max_n_bins
                        + logical_bin
                    )
                    * d_embedding
                    + d
                );
                atomicAdd(
                    bucket_sum + bucket_index,
                    grad_value
                );
                atomicAdd(
                    weighted_sum + bucket_index,
                    quantized_position * grad_value
                );
            }

            if (compute_grad_x) {
                const int64_t current_weight_index = (
                    (
                        feature_idx * max_n_bins
                        + current_physical_bin
                    )
                    * d_embedding
                    + d
                );
                const float current_weight = (
                    quantize_to_float<grad_t>(
                        weight[current_weight_index]
                    )
                );
                grad_x_partial += (
                    grad_value
                    * current_weight
                    * current_inv_width
                );

                if (is_internal_boundary) {
                    const int64_t previous_weight_index = (
                        (
                            feature_idx * max_n_bins
                            + logical_bin - 1
                        )
                        * d_embedding
                        + d
                    );
                    const float previous_weight = (
                        quantize_to_float<grad_t>(
                            weight[previous_weight_index]
                        )
                    );
                    grad_x_partial += (
                        grad_value
                        * previous_weight
                        * previous_inv_width
                    );
                }
            }
        }

        if (compute_grad_x) {
            grad_x_partial = group_reduce_sum<kWarpSize>(
                grad_x_partial
            );
            if (lane_idx == 0) {
                grad_x[pair_idx] = grad_x_partial;
            }
        }
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

    at::Tensor bucket_sum;
    at::Tensor weighted_sum;

    if (compute_grad_weight) {
        bucket_sum = at::zeros_like(weight);
        weighted_sum = at::zeros_like(weight);
    }

    if (batch_size > 0) {
        const int tile_d = d_embedding <= 16 ? 16 : 32;
        const int64_t n_batch_tiles = ceil_div<int64_t>(
            batch_size,
            kSamplesPerBlock
        );
        const int64_t n_d_tiles = ceil_div<int64_t>(
            d_embedding,
            tile_d
        );

        TORCH_CHECK(
            n_features <= (
                std::numeric_limits<int64_t>::max()
                / n_d_tiles
            ),
            "The backward persistent tile count overflows int64"
        );
        const int64_t feature_d_tiles = (
            n_features * n_d_tiles
        );
        const int64_t n_partitions = std::min<int64_t>(
            n_batch_tiles,
            std::max<int64_t>(
                1,
                ceil_div<int64_t>(
                    kTargetPersistentBlocks,
                    feature_d_tiles
                )
            )
        );
        TORCH_CHECK(
            feature_d_tiles <= (
                std::numeric_limits<int64_t>::max()
                / n_partitions
            ),
            "The backward persistent tile count overflows int64"
        );
        const int64_t total_persistent_tiles = (
            feature_d_tiles * n_partitions
        );

        const size_t accumulator_bytes = (
            compute_grad_weight
                ? static_cast<size_t>(2)
                    * static_cast<size_t>(max_n_bins)
                    * static_cast<size_t>(tile_d)
                    * sizeof(float)
                : 0
        );
        const size_t shared_bytes = (
            static_cast<size_t>(2 * max_n_bins + 1)
            * sizeof(float)
            + accumulator_bytes
        );

        const int device_index = x.get_device();
        int max_optin_shared_bytes = 0;
        C10_CUDA_CHECK(
            cudaDeviceGetAttribute(
                &max_optin_shared_bytes,
                cudaDevAttrMaxSharedMemoryPerBlockOptin,
                device_index
            )
        );
        const cudaStream_t stream = (
            c10::cuda::getCurrentCUDAStream(device_index).stream()
        );
        const int launch_blocks = static_cast<int>(
            std::min<int64_t>(
                total_persistent_tiles,
                kMaxLaunchBlocks
            )
        );
        const bool use_shared_path = (
            shared_bytes
            <= static_cast<size_t>(max_optin_shared_bytes)
            && shared_bytes
            <= static_cast<size_t>(
                std::numeric_limits<int>::max()
            )
        );

        float* grad_x_pointer = (
            compute_grad_x
                ? grad_x.data_ptr<float>()
                : nullptr
        );
        float* bucket_pointer = (
            compute_grad_weight
                ? bucket_sum.data_ptr<float>()
                : nullptr
        );
        float* weighted_pointer = (
            compute_grad_weight
                ? weighted_sum.data_ptr<float>()
                : nullptr
        );

        if (use_shared_path) {
            if (tile_d == 16) {
                C10_CUDA_CHECK(
                    cudaFuncSetAttribute(
                        ple_backward_shared_kernel<
                            grad_t,
                            bin_index_t,
                            16
                        >,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(shared_bytes)
                    )
                );
                ple_backward_shared_kernel<
                    grad_t,
                    bin_index_t,
                    16
                ><<<
                    launch_blocks,
                    kBlockThreads,
                    shared_bytes,
                    stream
                >>>(
                    grad_output.data_ptr<grad_t>(),
                    x.data_ptr<float>(),
                    bin_edges.data_ptr<float>(),
                    inv_bin_widths.data_ptr<float>(),
                    n_bins.data_ptr<int32_t>(),
                    weight.data_ptr<float>(),
                    bin_indices.data_ptr<bin_index_t>(),
                    grad_x_pointer,
                    bucket_pointer,
                    weighted_pointer,
                    batch_size,
                    n_features,
                    max_n_bins,
                    d_embedding,
                    n_batch_tiles,
                    n_d_tiles,
                    n_partitions,
                    total_persistent_tiles,
                    compute_grad_x,
                    compute_grad_weight
                );
            } else {
                C10_CUDA_CHECK(
                    cudaFuncSetAttribute(
                        ple_backward_shared_kernel<
                            grad_t,
                            bin_index_t,
                            32
                        >,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        static_cast<int>(shared_bytes)
                    )
                );
                ple_backward_shared_kernel<
                    grad_t,
                    bin_index_t,
                    32
                ><<<
                    launch_blocks,
                    kBlockThreads,
                    shared_bytes,
                    stream
                >>>(
                    grad_output.data_ptr<grad_t>(),
                    x.data_ptr<float>(),
                    bin_edges.data_ptr<float>(),
                    inv_bin_widths.data_ptr<float>(),
                    n_bins.data_ptr<int32_t>(),
                    weight.data_ptr<float>(),
                    bin_indices.data_ptr<bin_index_t>(),
                    grad_x_pointer,
                    bucket_pointer,
                    weighted_pointer,
                    batch_size,
                    n_features,
                    max_n_bins,
                    d_embedding,
                    n_batch_tiles,
                    n_d_tiles,
                    n_partitions,
                    total_persistent_tiles,
                    compute_grad_x,
                    compute_grad_weight
                );
            }
        } else {
            TORCH_CHECK(
                batch_size <= (
                    std::numeric_limits<int64_t>::max()
                    / n_features
                ),
                "batch_size * n_features overflows int64"
            );
            const int64_t total_pairs = batch_size * n_features;
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

            ple_backward_generic_kernel<
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
                n_bins.data_ptr<int32_t>(),
                weight.data_ptr<float>(),
                bin_indices.data_ptr<bin_index_t>(),
                grad_x_pointer,
                bucket_pointer,
                weighted_pointer,
                n_features,
                max_n_bins,
                d_embedding,
                total_pairs,
                compute_grad_x,
                compute_grad_weight
            );
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    if (compute_grad_weight) {
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
        const int device_index = x.get_device();
        const cudaStream_t stream = (
            c10::cuda::getCurrentCUDAStream(device_index).stream()
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
