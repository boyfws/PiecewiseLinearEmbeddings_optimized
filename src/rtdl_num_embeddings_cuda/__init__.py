"""Fused CUDA operators for rtdl_num_embeddings."""

# Load C++ dispatcher registrations first, then Python FakeTensor/autograd
# registrations, and only then expose the module class.
from . import _C as _C  # noqa: F401
from . import _meta as _meta  # noqa: F401
from . import _autograd as _autograd  # noqa: F401
from .module import CudaPiecewiseLinearEmbeddings

__all__ = [
    "CudaPiecewiseLinearEmbeddings",
]

__version__ = "0.2.0.dev0"
