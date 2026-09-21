"""
The TS-ICL model architecture (Perceiver encoder + in-context-learning transformer head),
for use by ``pypots.imputation.tsicl``.
"""

from .encoder import PerceiverEncoder
from .icl_learning import ICLearning, ICLearningCrossAttn
from .inr import LocalityAwareINRDecoder
from .network import TSICLNetwork
from .inference_utils import (
    HF_REPO_ID,
    CustomStandardScaler,
    build_tsicl_network,
    complete_nans,
    fetch_X,
    flatten_channel_independent,
    get_quantile_indices,
    load_tsicl_checkpoint,
    make_grid,
    nan_row_fallback,
    prepare_context_tensors,
    resolve_quantile_selection,
    run_batched_point_estimate,
    unflatten_channel_independent,
)

__all__ = [
    "TSICLNetwork",
    "PerceiverEncoder",
    "ICLearning",
    "ICLearningCrossAttn",
    "LocalityAwareINRDecoder",
    "HF_REPO_ID",
    "CustomStandardScaler",
    "build_tsicl_network",
    "complete_nans",
    "fetch_X",
    "flatten_channel_independent",
    "get_quantile_indices",
    "load_tsicl_checkpoint",
    "make_grid",
    "nan_row_fallback",
    "prepare_context_tensors",
    "resolve_quantile_selection",
    "run_batched_point_estimate",
    "unflatten_channel_independent",
]
