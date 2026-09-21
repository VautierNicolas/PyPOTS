from __future__ import annotations

"""
Vendored from the TS-ICL model architecture: https://github.com/EDF-Lab/ts-icl
"""

from .utils import (
    DiagonalGaussianDistribution,
    cache_fn,
    count_parameters,
    default,
    dropout_seq,
    exists,
    linear_scheduler,
)

__all__ = [
    "count_parameters",
    "exists",
    "default",
    "cache_fn",
    "dropout_seq",
    "DiagonalGaussianDistribution",
    "linear_scheduler"
]