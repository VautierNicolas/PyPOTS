# Vendored from the TS-ICL model architecture: https://github.com/EDF-Lab/ts-icl
# Copyright (c) 2026 EDF SA. Licensed under the TS-ICL Non-Commercial License v1.0,
# NOT under PyPOTS' BSD-3-Clause license. See the NOTICE file in pypots/nn/modules/tsicl/
# for the full license text and restrictions (non-commercial research/evaluation use only).

from __future__ import annotations

"""
The TS-ICL model architecture (Perceiver encoder + in-context-learning transformer head),
vendored from https://github.com/EDF-Lab/ts-icl for use by ``pypots.imputation.tsicl``.

"""

from .encoder import PerceiverEncoder, UnivariatePerceiverEncoder
from .icl_learning import ICLearning, ICLearningCrossAttn
from .inr import LocalityAwareINRDecoder
from .network import PerceiverINR, TSICLNetwork

__all__ = [
    "TSICLNetwork",
    "PerceiverINR",
    "PerceiverEncoder",
    "UnivariatePerceiverEncoder",
    "ICLearning",
    "ICLearningCrossAttn",
    "LocalityAwareINRDecoder",
]
