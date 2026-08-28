# Vendored from the TS-ICL model architecture: https://github.com/EDF-Lab/ts-icl
# Copyright (c) 2026 EDF SA. Licensed under the TS-ICL Non-Commercial License v1.0,
# NOT under PyPOTS' BSD-3-Clause license. See the NOTICE file in pypots/nn/modules/tsicl/
# for the full license text and restrictions (non-commercial research/evaluation use only).

from __future__ import annotations

from .fourier_features import FourierPositionalEmbedding
from .gaussian import GaussianEncoding
from .nerf import MultiScaleNeRFEncoding, NeRFEncoding

__all__ = [
    'GaussianEncoding',
    'NeRFEncoding',
    'MultiScaleNeRFEncoding',
    'FourierPositionalEmbedding'
]