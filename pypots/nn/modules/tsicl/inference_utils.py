"""
Utilities for TS-ICL model inference, adapted from the original TS-ICL codebase.
"""

"""
Task-agnostic TS-ICL inference plumbing (checkpoint loading, the TSICLNetwork builder,
z-normalization, and context/target grid construction) shared by
``pypots.imputation.tsicl`` and ``pypots.forecasting.tsicl``.

"""

# Created by Nicolas Vautier <nicolas.vautier@edf.fr>
# License: BSD-3-Clause for this file; the vendored TS-ICL architecture and weights
# it loads are under the TS-ICL Non-Commercial License v1.0, see this package's NOTICE.

import importlib
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn

# Where the checkpoint's Hydra-style ``_target_`` dotted paths (e.g. "tsicl.model.encoder.PerceiverEncoder")
# originally pointed, remapped to this vendored copy of the architecture.
_TARGET_REMAP = {
    "tsicl.TSICLNetwork": "pypots.nn.modules.tsicl.TSICLNetwork",
    "tsicl.model.network.TSICLNetwork": "pypots.nn.modules.tsicl.TSICLNetwork",
    "tsicl.model.encoder.PerceiverEncoder": "pypots.nn.modules.tsicl.PerceiverEncoder",
    "tsicl.model.icl_learning.ICLearning": "pypots.nn.modules.tsicl.ICLearning",
    "tsicl.model.icl_learning.ICLearningCrossAttn": "pypots.nn.modules.tsicl.ICLearningCrossAttn",
    "tsicl.model.inr.LocalityAwareINRDecoder": "pypots.nn.modules.tsicl.LocalityAwareINRDecoder",
}

HF_REPO_ID = "taharnbl/TS-ICL"


def _instantiate(config):
    """Minimal re-implementation of ``hydra.utils.instantiate`` for TS-ICL checkpoint configs.

    TS-ICL checkpoints store a plain nested dict describing how to rebuild the network
    (a ``_target_`` dotted class path plus its constructor kwargs, recursively for nested
    submodules). The original code resolves it with Hydra; here the dotted paths are remapped
    to this vendored copy of the architecture (see ``_TARGET_REMAP``) and resolved with plain
    ``importlib``, avoiding a new hard dependency on ``hydra-core`` for this single use case.
    """
    if isinstance(config, dict) and "_target_" in config:
        kwargs = {k: _instantiate(v) for k, v in config.items() if k != "_target_"}
        target = _TARGET_REMAP.get(config["_target_"], config["_target_"])
        module_path, cls_name = target.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), cls_name)
        return cls(**kwargs)
    if isinstance(config, list):
        return [_instantiate(v) for v in config]
    if isinstance(config, dict):
        return {k: _instantiate(v) for k, v in config.items()}
    return config


def load_tsicl_checkpoint(
    model_path: Optional[str],
    checkpoint_version: str,
    allow_auto_download: bool = True,
) -> dict:
    """Load a TS-ICL checkpoint from a local path, or the Hugging Face Hub if not found.

    Ported from ``TSICL._load_model`` (credits therein: adapted from the TabICL repo,
    https://github.com/soda-inria/tabicl/blob/main/src/tabicl/_sklearn/regressor.py).
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    filename = checkpoint_version

    if model_path is None:
        try:
            resolved_path = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=filename, local_files_only=True))
        except LocalEntryNotFoundError:
            if not allow_auto_download:
                raise ValueError(
                    f"Checkpoint '{filename}' not cached and automatic download is disabled.\n"
                    f"Set allow_auto_download=True to download the checkpoint from Hugging Face Hub ({HF_REPO_ID})."
                )
            resolved_path = Path(hf_hub_download(repo_id=HF_REPO_ID, filename=filename))
    else:
        resolved_path = Path(model_path)
        if not resolved_path.exists():
            if not allow_auto_download:
                raise ValueError(
                    f"Checkpoint not found at '{resolved_path}' and automatic download is disabled.\n"
                    f"Either provide a valid checkpoint path, or set allow_auto_download=True to download "
                    f"'{filename}' from Hugging Face Hub ({HF_REPO_ID})."
                )
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename, local_dir=resolved_path.parent)
            Path(cache_path).rename(resolved_path)

    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
    assert "config" in checkpoint, "The checkpoint doesn't contain the model configuration."
    return checkpoint


def build_tsicl_network(checkpoint: dict, state_dict_key: str) -> nn.Module:
    """Build the ``TSICLNetwork`` described by a checkpoint's config and load its weights.

    TS-ICL checkpoints store two separate sets of weights for the same architecture, one
    fine-tuned for imputation (``state_dict_key="imputer"``) and one for forecasting
    (``state_dict_key="forecaster"``), matching the original ``TSICL.imputer`` /
    ``TSICL.forecaster`` pipeline attributes.
    """
    model_config = checkpoint["config"]
    network = _instantiate(model_config)
    if state_dict_key in checkpoint:
        network.load_state_dict(checkpoint[state_dict_key])
    network.eval()
    return network


class CustomStandardScaler:
    """Per-sample z-normalization, ignoring NaNs. Ported from ``tsicl.utils.scaler``."""

    def __init__(self, dim: int = 1, epsilon: float = 1e-5):
        self.dim = dim
        self.epsilon = epsilon
        self.mean = torch.empty((1,))
        self.std = torch.empty((1,))

    def fit(self, X: torch.Tensor) -> None:
        self.mean = torch.nan_to_num(torch.nanmean(X, dim=self.dim, keepdim=True), nan=0.0)
        scale = torch.nan_to_num((X - self.mean).square().nanmean(dim=self.dim, keepdim=True).sqrt(), nan=1.0)
        self.std = torch.where(scale == 0, self.epsilon, scale)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.mean) / self.std

    def inv_transform(self, X: torch.Tensor) -> torch.Tensor:
        return X * self.std + self.mean


def make_grid(grid_length: int, num_samples: int) -> torch.Tensor:
    """Build a `(num_samples, grid_length, 1)` grid of time coordinates in [0, 1]. Ported from ``tsicl.utils.utils``."""
    grid = torch.linspace(0.0, 1.0, grid_length).float().unsqueeze(0).repeat_interleave(repeats=num_samples, dim=0)
    return grid.unsqueeze(-1)


def complete_nans(X: torch.Tensor, grid: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Replace NaNs in `X` by other observed values of the same series (deterministic).

    Ported from ``tsicl.utils.utils``.
    """
    if not torch.isnan(X).any():
        return {"values": X.clone(), "coords": grid.clone()}

    torch.manual_seed(42)
    mask_observed = ~torch.isnan(X[..., 0])  # (N, T)
    X_filled = X[..., 0].clone()
    grid_filled = grid[..., 0].clone()

    for i in range(X_filled.shape[0]):
        valid_indices = torch.where(mask_observed[i])[0]
        invalid_indices = torch.where(~mask_observed[i])[0]
        if valid_indices.numel() > 0 and invalid_indices.numel() > 0:
            replacements = valid_indices[torch.randint(0, valid_indices.numel(), (invalid_indices.numel(),))]
            X_filled[i, invalid_indices] = X[i, replacements, 0]
            grid_filled[i, invalid_indices] = grid[i, replacements, 0]

    return {"values": X_filled.unsqueeze(-1), "coords": grid_filled.unsqueeze(-1)}


def prepare_context_tensors(grid: torch.Tensor, series_c: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Build fixed-length context (coords, values) tensors from a batch with a variable number of
    missing points per sample, right-padding the shorter ones with duplicated observed values.
    Ported (covariate-free) from ``tsicl.utils.task_utils.prepare_context_tensors``.
    """
    nb_missing_points = torch.isnan(series_c).sum(1).squeeze()
    if nb_missing_points.ndim == 0:
        nb_missing_points = nb_missing_points.unsqueeze(0)
    max_context_len = int(series_c.shape[1] - min(nb_missing_points))

    list_coords_c, list_series_c = [], []
    for idx in range(len(series_c)):
        sample = series_c[idx].squeeze(-1)
        is_missing_mask = torch.isnan(sample)

        if is_missing_mask.sum() > min(nb_missing_points):
            cur_len = len(sample[~is_missing_mask])
            mis_len = max_context_len - cur_len
            series_i = torch.cat(
                [sample[~is_missing_mask], torch.nan * torch.ones(mis_len).to(sample.device)]
            ).unsqueeze(-1)
            grid_i = torch.cat(
                [grid[idx][~is_missing_mask].squeeze(-1), torch.ones(mis_len).to(sample.device)]
            ).unsqueeze(-1)
            out = complete_nans(series_i, grid_i)
            series_i, grid_i = out["values"], out["coords"]
        else:
            series_i, grid_i = series_c[idx][~is_missing_mask], grid[idx][~is_missing_mask]

        list_coords_c.append(grid_i.reshape(-1, 1))
        list_series_c.append(series_i.reshape(-1, 1))

    return {
        "series_c": torch.stack(list_series_c),
        "coords_c": torch.stack(list_coords_c),
    }


def get_quantile_indices(network: nn.Module, quantile_levels: List[float]) -> List[int]:
    """Map requested quantile levels to their index in a TS-ICL network's output head.
    Ported from ``TSICL._get_quantile_indices``."""
    start_quantile = network.tf_icl.start_quantile
    end_quantile = network.tf_icl.end_quantile
    nb_quantiles = network.tf_icl.nb_quantiles
    training_quantile_levels = np.linspace(start_quantile * 100, end_quantile * 100, nb_quantiles).tolist()

    requested = [100 * x for x in quantile_levels]
    if not set(requested).issubset(training_quantile_levels):
        raise ValueError(f"quantile_levels={quantile_levels} must be a subset of TS-ICL's trained quantiles.")
    return [training_quantile_levels.index(q) for q in requested]


def fetch_X(data: Union[dict, str], file_type: str = "hdf5") -> np.ndarray:
    """Load the ``X`` array from a PyPOTS dict/H5-file dataset, as both ``TSICL`` model
    classes (imputation and forecasting) expect it: shape `(n_samples, n_steps, n_features)`,
    NaNs allowed.
    """
    if isinstance(data, str):
        with h5py.File(data, "r") as f:
            X = f["X"][:]
    else:
        X = data["X"]
    if isinstance(X, list):
        X = np.asarray(X)
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    assert len(X.shape) == 3, (
        f"Input X should have 3 dimensions [n_samples, n_steps, n_features], "
        f"but the actual shape of X: {X.shape}"
    )
    return X


def resolve_quantile_selection(
    network: nn.Module,
    point_estimator: str,
    quantile_levels: Optional[List[float]],
) -> Tuple[List[int], Optional[int], str]:
    """Resolve the requested quantile levels to indices in the network's output head, and
    fall back from ``'median'`` to ``'mean'`` if 0.5 wasn't requested (there'd be no median
    to report).
    """
    quantile_levels = quantile_levels or [0.1, 0.3, 0.5, 0.7, 0.9]
    quantile_indices = get_quantile_indices(network, quantile_levels)
    median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else None
    if point_estimator == "median" and median_idx is None:
        point_estimator = "mean"
    return quantile_indices, median_idx, point_estimator


def flatten_channel_independent(X: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Flatten a `(n_samples, n_steps, n_features)` array into `(n_samples * n_features,
    n_steps)`, one row per univariate series (TS-ICL models one series at a time; see the
    channel-independence note in ``TSICL``'s docstring).
    """
    n_samples, n_steps, n_features = X.shape
    flat = X.astype(np.float32).transpose(0, 2, 1).reshape(n_samples * n_features, n_steps)
    return flat, n_samples, n_features


def unflatten_channel_independent(out: np.ndarray, n_samples: int, n_features: int) -> np.ndarray:
    """Undo :func:`flatten_channel_independent`, folding the per-series rows of `out`
    `(n_samples * n_features, output_len)` back into `(n_samples, output_len, n_features)`.
    """
    output_len = out.shape[1]
    return out.reshape(n_samples, n_features, output_len).transpose(0, 2, 1)


def nan_row_fallback(out: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Safety net: fall back to the per-row mean of the observed part of `flat` (or 0 if
    the whole row is missing) wherever `out` still holds NaN, e.g. for rows with too few
    observed points to condition the model on.
    """
    if np.isfinite(out).all():
        return out
    with warnings.catch_warnings():
        # nanmean on an all-NaN row (fully-missing series) warns; nan_to_num handles it
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        row_fallback = np.nan_to_num(
            np.nanmean(np.where(np.isfinite(flat), flat, np.nan), axis=1, keepdims=True), nan=0.0
        )
    return np.where(np.isfinite(out), out, row_fallback)


def run_batched_point_estimate(
    flat: np.ndarray,
    output_len: int,
    batch_size: int,
    device: torch.device,
    quantile_indices: List[int],
    point_estimator: str,
    median_idx: Optional[int],
    predict_batch_fn: Callable[[torch.Tensor], torch.Tensor],
) -> np.ndarray:
    """Run `predict_batch_fn` batch-wise over `flat` (`n_series, n_steps`, one univariate
    series per row) and extract the requested point estimate from its predicted quantiles.

    This is the batching/point-estimate harness shared by ``impute_with_tsicl`` and
    ``forecast_with_tsicl``; the per-batch TS-ICL forward pass itself (context/target grid
    construction, single-pass vs. autoregressive rollout) is task-specific and supplied by
    the caller as `predict_batch_fn`, which takes a `(bs, n_steps, 1)` context tensor and
    returns its predicted quantiles of shape `(bs, output_len, num_quantiles)`.
    """
    out = np.empty((flat.shape[0], output_len), dtype=np.float32)
    for start in range(0, len(flat), batch_size):
        batch = flat[start : start + batch_size]
        series_c = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(-1)

        with torch.no_grad():
            quantiles = predict_batch_fn(series_c)[..., quantile_indices]

        if point_estimator == "median":
            point = quantiles[..., median_idx : median_idx + 1]
        else:
            point = quantiles.mean(-1, keepdim=True)
        out[start : start + batch_size] = point.squeeze(-1).cpu().numpy()

    return out
