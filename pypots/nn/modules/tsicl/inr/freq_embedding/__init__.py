from __future__ import annotations

"""
Vendored from the TS-ICL model architecture: https://github.com/EDF-Lab/ts-icl
"""

from .fourier_features import FourierPositionalEmbedding
from .gaussian import GaussianEncoding
from .nerf import MultiScaleNeRFEncoding, NeRFEncoding

__all__ = [
    'GaussianEncoding',
    'NeRFEncoding',
    'MultiScaleNeRFEncoding',
    'FourierPositionalEmbedding'
]