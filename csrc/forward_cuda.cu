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


template <typename output_t, typename bin_index_t, int kTileD>
__global__ void ple_forward_shared_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bin_edges,
    const float* __restrict__ inv_bin_widths,
    const int32_t* __restrict__ n_bins,
    const float* __restrict__ weight,
    output_t* __restrict__ output,
    bin_index_t* __restrict__ bin_indices,
    int64_t batch_size,
    int64_t n_features,
    int64_t max_n_bins,
    int64_t d_embedding,
    int64_t n_batch_tiles,
    int64_t n_d_tiles,
    int64_t n_partitions,
    int64_t total_persistent_tiles
) {
    extern __shared__ float shared_memory[];

    float* shared_edges = shared_memory;
    float* shared_inv_bin_widths = (
        shared_edges
        + max_n_bins
        + 1
    );
    float* shared_prefix = (
        shared_inv_bin_widths
        + max_n_bins
    );

    constexpr int kGroupsPerBlock = (
        kBlockThreads / kTileD
    );

    for (
        int64_t persistent_tile = blockIdx.x;
        persistent_tile < total_persistent_tiles;
        persistent_tile += gridDim.x
    ) {
        const int64_t partition_idx = (
            persistent_tile % n_partitions
        );
        const int64_t feature_idx = (
            persistent_tile / n_partitions
        );
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

        __syncthreads();

        // Compute the bin once for every (sample, feature) pair. The saved
        // compact index is reused by all D tiles and by backward.
        for (
            int64_t batch_tile = partition_idx;
            batch_tile < n_batch_tiles;
            batch_tile += n_partitions
        ) {
            const int64_t batch_base = (
                batch_tile * kSamplesPerBlock
            );

            for (
                int sample_offset = threadIdx.x;
                sample_offset < kSamplesPerBlock;
                sample_offset += blockDim.x
            ) {
                const int64_t sample_idx = (
                    batch_base + sample_offset
                );

                if (sample_idx < batch_size) {
                    const float value = x[
                        sample_idx * n_features + feature_idx
                    ];
                    const int logical_bin = find_logical_bin_right(
                        value,
                        shared_edges,
                        n_real_bins
                    );
                    bin_indices[
                        sample_idx * n_features + feature_idx
                    ] = static_cast<bin_index_t>(logical_bin);
                }
            }
        }

        __syncthreads();

        for (
            int64_t d_tile = 0;
            d_tile < n_d_tiles;
            ++d_tile
        ) {
            // One thread computes the complete prefix for one D component.
            if (threadIdx.x < kTileD) {
                const int d_local = threadIdx.x;
                const int64_t d_global = (
                    d_tile * kTileD + d_local
                );

                if (d_global < d_embedding) {
                    float accumulator = 0.0f;

                    for (
                        int logical_bin = 0;
                        logical_bin < n_real_bins;
                        ++logical_bin
                    ) {
                        shared_prefix[
                            logical_bin * kTileD + d_local
                        ] = accumulator;

                        const int physical_bin = physical_bin_index(
                            logical_bin,
                            n_real_bins,
                            static_cast<int>(max_n_bins)
                        );
                        const int64_t weight_index = (
                            (
                                feature_idx * max_n_bins
                                + physical_bin
                            )
                            * d_embedding
                            + d_global
                        );

                        accumulator += quantize_to_float<output_t>(
                            weight[weight_index]
                        );
                    }
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

                // All groups execute the same iteration count, which keeps
                // the shuffle masks valid for kTileD=16 and kTileD=32.
                for (
                    int sample_offset = group_idx;
                    sample_offset < kSamplesPerBlock;
                    sample_offset += kGroupsPerBlock
                ) {
                    const int64_t sample_idx = (
                        batch_base + sample_offset
                    );

                    int logical_bin = 0;
                    float position = 0.0f;

                    if (
                        lane_idx == 0
                        && sample_idx < batch_size
                    ) {
                        const int64_t pair_idx = (
                            sample_idx * n_features
                            + feature_idx
                        );
                        logical_bin = static_cast<int>(
                            bin_indices[pair_idx]
                        );
                        const float value = x[pair_idx];
                        position = (
                            value - shared_edges[logical_bin]
                        ) * shared_inv_bin_widths[logical_bin];
                    }

                    logical_bin = __shfl_sync(
                        0xffffffffu,
                        logical_bin,
                        0,
                        kTileD
                    );
                    position = __shfl_sync(
                        0xffffffffu,
                        position,
                        0,
                        kTileD
                    );

                    if (
                        sample_idx < batch_size
                        && d_global < d_embedding
                    ) {
                        const int physical_bin = physical_bin_index(
                            logical_bin,
                            n_real_bins,
                            static_cast<int>(max_n_bins)
                        );
                        const int64_t weight_index = (
                            (
                                feature_idx * max_n_bins
                                + physical_bin
                            )
                            * d_embedding
                            + d_global
                        );
                        const float quantized_position = (
                            quantize_to_float<output_t>(position)
                        );
                        const float quantized_weight = (
                            quantize_to_float<output_t>(
                                weight[weight_index]
                            )
                        );
                        const float prefix = shared_prefix[
                            logical_bin * kTileD + lane_idx
                        ];
                        const float result = fmaf(
                            quantized_position,
                            quantized_weight,
                            prefix
                        );

                        output[
                            (
                                sample_idx * n_features
                                + feature_idx
                            )
                            * d_embedding
                            + d_global
                        ] = static_cast<output_t>(result);
                    }
                }
            }

            __syncthreads();
        }
    }
}


template <typename output_t, typename bin_index_t>
__global__ void ple_forward_generic_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bin_edges,
    const float* __restrict__ inv_bin_widths,
    const int32_t* __restrict__ n_bins,
    const float* __restrict__ weight,
    output_t* __restrict__ output,
    bin_index_t* __restrict__ bin_indices,
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
        const float* feature_edges = (
            bin_edges
            + feature_idx * (max_n_bins + 1)
        );
        const float* feature_inv_bin_widths = (
            inv_bin_widths
            + feature_idx * max_n_bins
        );

        int logical_bin = 0;
        float position = 0.0f;

        if (lane_idx == 0) {
            const float value = x[pair_idx];

            logical_bin = find_logical_bin_right(
                value,
                feature_edges,
                n_real_bins
            );
            bin_indices[pair_idx] = (
                static_cast<bin_index_t>(logical_bin)
            );
            position = (
                value - feature_edges[logical_bin]
            ) * feature_inv_bin_widths[logical_bin];
        }

        logical_bin = __shfl_sync(
            0xffffffffu,
            logical_bin,
            0
        );
        position = __shfl_sync(
            0xffffffffu,
            position,
            0
        );

        const int physical_bin = physical_bin_index(
            logical_bin,
            n_real_bins,
            static_cast<int>(max_n_bins)
        );
        const float quantized_position = (
            quantize_to_float<output_t>(position)
        );

        for (
            int64_t d = lane_idx;
            d < d_embedding;
            d += kWarpSize
        ) {
            float prefix = 0.0f;

            for (
                int previous_bin = 0;
                previous_bin < logical_bin;
                ++previous_bin
            ) {
                const int64_t previous_index = (
                    (
                        feature_idx * max_n_bins
                        + previous_bin
                    )
                    * d_embedding
                    + d
                );

                prefix += quantize_to_float<output_t>(
                    weight[previous_index]
                );
            }

            const int64_t current_index = (
                (
                    feature_idx * max_n_bins
                    + physical_bin
                )
                * d_embedding
                + d
            );
            const float current_weight = (
                quantize_to_float<output_t>(weight[current_index])
            );

            output[pair_idx * d_embedding + d] = (
                static_cast<output_t>(
                    fmaf(
                        quantized_position,
                        current_weight,
                        prefix
                    )
                )
            );
        }
    }
}


template <typename output_t, typename bin_index_t>
void launch_ple_forward(
    const at::Tensor& x,
    const at::Tensor& bin_edges,
    const at::Tensor& inv_bin_widths,
    const at::Tensor& n_bins,
    const at::Tensor& weight,
    at::Tensor& output,
    at::Tensor& bin_indices
) {
    const int64_t batch_size = x.size(0);
    const int64_t n_features = x.size(1);
    const int64_t max_n_bins = weight.size(1);
    const int64_t d_embedding = weight.size(2);

    if (batch_size == 0) {
        return;
    }

    const int tile_d = d_embedding <= 16 ? 16 : 32;
    const int64_t n_batch_tiles = ceil_div<int64_t>(
        batch_size,
        kSamplesPerBlock
    );
    const int64_t n_d_tiles = ceil_div<int64_t>(
        d_embedding,
        tile_d
    );
    const int64_t n_partitions = std::min<int64_t>(
        n_batch_tiles,
        std::max<int64_t>(
            1,
            ceil_div<int64_t>(
                kTargetPersistentBlocks,
                n_features
            )
        )
    );

    TORCH_CHECK(
        n_features <= (
            std::numeric_limits<int64_t>::max()
            / n_partitions
        ),
        "The forward persistent tile count overflows int64"
    );
    const int64_t total_persistent_tiles = (
        n_features * n_partitions
    );

    const size_t shared_bytes = (
        static_cast<size_t>(2 * max_n_bins + 1)
        * sizeof(float)
        + static_cast<size_t>(max_n_bins)
        * static_cast<size_t>(tile_d)
        * sizeof(float)
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

    if (use_shared_path) {
        if (tile_d == 16) {
            C10_CUDA_CHECK(
                cudaFuncSetAttribute(
                    ple_forward_shared_kernel<
                        output_t,
                        bin_index_t,
                        16
                    >,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes)
                )
            );
            ple_forward_shared_kernel<
                output_t,
                bin_index_t,
                16
            ><<<
                launch_blocks,
                kBlockThreads,
                shared_bytes,
                stream
            >>>(
                x.data_ptr<float>(),
                bin_edges.data_ptr<float>(),
                inv_bin_widths.data_ptr<float>(),
                n_bins.data_ptr<int32_t>(),
                weight.data_ptr<float>(),
                output.data_ptr<output_t>(),
                bin_indices.data_ptr<bin_index_t>(),
                batch_size,
                n_features,
                max_n_bins,
                d_embedding,
                n_batch_tiles,
                n_d_tiles,
                n_partitions,
                total_persistent_tiles
            );
        } else {
            C10_CUDA_CHECK(
                cudaFuncSetAttribute(
                    ple_forward_shared_kernel<
                        output_t,
                        bin_index_t,
                        32
                    >,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes)
                )
            );
            ple_forward_shared_kernel<
                output_t,
                bin_index_t,
                32
            ><<<
                launch_blocks,
                kBlockThreads,
                shared_bytes,
                stream
            >>>(
                x.data_ptr<float>(),
                bin_edges.data_ptr<float>(),
                inv_bin_widths.data_ptr<float>(),
                n_bins.data_ptr<int32_t>(),
                weight.data_ptr<float>(),
                output.data_ptr<output_t>(),
                bin_indices.data_ptr<bin_index_t>(),
                batch_size,
                n_features,
                max_n_bins,
                d_embedding,
                n_batch_tiles,
                n_d_tiles,
                n_partitions,
                total_persistent_tiles
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

        ple_forward_generic_kernel<
            output_t,
            bin_index_t
        ><<<
            generic_blocks,
            kBlockThreads,
            0,
            stream
        >>>(
            x.data_ptr<float>(),
            bin_edges.data_ptr<float>(),
            inv_bin_widths.data_ptr<float>(),
            n_bins.data_ptr<int32_t>(),
            weight.data_ptr<float>(),
            output.data_ptr<output_t>(),
            bin_indices.data_ptr<bin_index_t>(),
            n_features,
            max_n_bins,
            d_embedding,
            total_pairs
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <typename output_t>
void dispatch_bin_index_type_forward(
    const at::Tensor& x,
    const at::Tensor& bin_edges,
    const at::Tensor& inv_bin_widths,
    const at::Tensor& n_bins,
    const at::Tensor& weight,
    at::Tensor& output,
    at::Tensor& bin_indices
) {
    if (bin_indices.scalar_type() == at::kByte) {
        launch_ple_forward<output_t, uint8_t>(
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            output,
            bin_indices
        );
    } else if (bin_indices.scalar_type() == at::kShort) {
        launch_ple_forward<output_t, int16_t>(
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            output,
            bin_indices
        );
    } else {
        launch_ple_forward<output_t, int32_t>(
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            output,
            bin_indices
        );
    }
}

}  // namespace rtdl_num_embeddings_cuda


std::tuple<at::Tensor, at::Tensor> ple_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& bin_edges,
    const at::Tensor& inv_bin_widths,
    const at::Tensor& n_bins,
    const at::Tensor& weight,
    at::ScalarType output_dtype
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bin_edges.is_cuda(), "bin_edges must be a CUDA tensor");
    TORCH_CHECK(
        inv_bin_widths.is_cuda(),
        "inv_bin_widths must be a CUDA tensor"
    );
    TORCH_CHECK(n_bins.is_cuda(), "n_bins must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(
        x.device() == bin_edges.device()
        && x.device() == inv_bin_widths.device()
        && x.device() == n_bins.device()
        && x.device() == weight.device(),
        "All forward tensors must be on the same CUDA device"
    );

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bin_edges.is_contiguous(), "bin_edges must be contiguous");
    TORCH_CHECK(
        inv_bin_widths.is_contiguous(),
        "inv_bin_widths must be contiguous"
    );
    TORCH_CHECK(n_bins.is_contiguous(), "n_bins must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    TORCH_CHECK(x.dim() == 2, "x must have shape [N, F]");
    TORCH_CHECK(
        bin_edges.dim() == 2,
        "bin_edges must have shape [F, B + 1]"
    );
    TORCH_CHECK(
        inv_bin_widths.dim() == 2,
        "inv_bin_widths must have shape [F, B]"
    );
    TORCH_CHECK(n_bins.dim() == 1, "n_bins must have shape [F]");
    TORCH_CHECK(
        weight.dim() == 3,
        "weight must have shape [F, B, D]"
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
    TORCH_CHECK(
        n_bins.scalar_type() == at::kInt,
        "n_bins must be int32"
    );
    TORCH_CHECK(
        weight.scalar_type() == at::kFloat,
        "weight parameters must be float32"
    );
    TORCH_CHECK(
        output_dtype == at::kFloat
        || output_dtype == at::kHalf
        || output_dtype == at::kBFloat16,
        "output_dtype must be float32, float16, or bfloat16"
    );

    const int64_t batch_size = x.size(0);
    const int64_t n_features = x.size(1);
    const int64_t max_n_bins = weight.size(1);
    const int64_t d_embedding = weight.size(2);

    TORCH_CHECK(n_features > 0, "n_features must be positive");
    TORCH_CHECK(max_n_bins > 0, "max_n_bins must be positive");
    TORCH_CHECK(d_embedding > 0, "d_embedding must be positive");
    TORCH_CHECK(
        weight.size(0) == n_features,
        "weight.size(0) must equal x.size(1)"
    );
    TORCH_CHECK(
        bin_edges.size(0) == n_features,
        "bin_edges.size(0) must equal x.size(1)"
    );
    TORCH_CHECK(
        bin_edges.size(1) == max_n_bins + 1,
        "bin_edges.size(1) must equal max_n_bins + 1"
    );
    TORCH_CHECK(
        inv_bin_widths.size(0) == n_features,
        "inv_bin_widths.size(0) must equal x.size(1)"
    );
    TORCH_CHECK(
        inv_bin_widths.size(1) == max_n_bins,
        "inv_bin_widths.size(1) must equal max_n_bins"
    );
    TORCH_CHECK(
        n_bins.size(0) == n_features,
        "n_bins.size(0) must equal x.size(1)"
    );

    c10::cuda::CUDAGuard device_guard(x.device());

    at::Tensor output = at::empty(
        {
            batch_size,
            n_features,
            d_embedding,
        },
        weight.options().dtype(output_dtype)
    );
    const at::ScalarType bin_index_dtype = (
        max_n_bins <= 256
            ? at::kByte
            : (
                max_n_bins <= 32768
                    ? at::kShort
                    : at::kInt
            )
    );
    at::Tensor bin_indices = at::empty(
        {
            batch_size,
            n_features,
        },
        weight.options().dtype(bin_index_dtype)
    );

    if (output.numel() == 0) {
        return {output, bin_indices};
    }

    if (output_dtype == at::kFloat) {
        rtdl_num_embeddings_cuda::dispatch_bin_index_type_forward<float>(
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            output,
            bin_indices
        );
    } else if (output_dtype == at::kHalf) {
        rtdl_num_embeddings_cuda::dispatch_bin_index_type_forward<
            c10::Half
        >(
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            output,
            bin_indices
        );
    } else {
        rtdl_num_embeddings_cuda::dispatch_bin_index_type_forward<
            c10::BFloat16
        >(
            x,
            bin_edges,
            inv_bin_widths,
            n_bins,
            weight,
            output,
            bin_indices
        );
    }

    return {output, bin_indices};
}
