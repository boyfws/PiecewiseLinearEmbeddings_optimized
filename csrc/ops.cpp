#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <tuple>


std::tuple<at::Tensor, at::Tensor> ple_forward_cuda(
    const at::Tensor& x,
    const at::Tensor& bin_edges,
    const at::Tensor& inv_bin_widths,
    const at::Tensor& n_bins,
    const at::Tensor& weight,
    at::ScalarType output_dtype
);


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
);


TORCH_LIBRARY(
    rtdl_num_embeddings_cuda,
    m
) {
    m.def(
        "forward("
        "Tensor x, "
        "Tensor bin_edges, "
        "Tensor inv_bin_widths, "
        "Tensor n_bins, "
        "Tensor weight, "
        "ScalarType output_dtype"
        ") -> (Tensor output, Tensor bin_indices)"
    );

    m.def(
        "backward("
        "Tensor grad_output, "
        "Tensor x, "
        "Tensor bin_edges, "
        "Tensor inv_bin_widths, "
        "Tensor n_bins, "
        "Tensor weight, "
        "Tensor bin_indices, "
        "bool compute_grad_x, "
        "bool compute_grad_weight"
        ") -> (Tensor grad_x, Tensor grad_weight)"
    );
}


TORCH_LIBRARY_IMPL(
    rtdl_num_embeddings_cuda,
    CUDA,
    m
) {
    m.impl(
        "forward",
        &ple_forward_cuda
    );
    m.impl(
        "backward",
        &ple_backward_cuda
    );
}


// Importing the extension executes the TORCH_LIBRARY registrations.
PYBIND11_MODULE(
    TORCH_EXTENSION_NAME,
    m
) {}
