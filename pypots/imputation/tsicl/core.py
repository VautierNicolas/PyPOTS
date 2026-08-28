# Vendored (imputation-only) inference logic ported from the TS-ICL pipeline:
# https://github.com/EDF-Lab/ts-icl (src/tsicl/pipeline.py, src/tsicl/utils/*).
# Copyright (c) 2026 EDF SA. Licensed under the TS-ICL Non-Commercial License v1.0,
# NOT under PyPOTS' BSD-3-Clause license. See the NOTICE file in this directory
# for the full license text and restrictions (non-commercial research/evaluation use only).

"""
The core imputation logic of TS-ICL, ported from the (zero-shot, gradient-free)
``TSICL.impute()`` pipeline of the original implementation. Covariates and the
forecasting task are out of scope here since PyPOTS' imputation contract has no
notion of either.

"""

import importlib
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import repeat

# Where the checkpoint's Hydra-style ``_target_`` dotted paths (e.g. "tsicl.model.encoder.PerceiverEncoder")
# originally pointed, remapped to this vendored copy of the architecture.
_TARGET_REMAP = {
    "tsicl.TSICLNetwork": "pypots.nn.modules.tsicl.TSICLNetwork",
    "tsicl.model.network.TSICLNetwork": "pypots.nn.modules.tsicl.TSICLNetwork",
    "tsicl.model.network.PerceiverINR": "pypots.nn.modules.tsicl.PerceiverINR",
    "tsicl.model.encoder.PerceiverEncoder": "pypots.nn.modules.tsicl.PerceiverEncoder",
    "tsicl.model.encoder.UnivariatePerceiverEncoder": "pypots.nn.modules.tsicl.UnivariatePerceiverEncoder",
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


def build_tsicl_imputer(checkpoint: dict) -> nn.Module:
    """Build the ``TSICLNetwork`` imputer described by a checkpoint's config and load its weights."""
    model_config = checkpoint["config"]
    imputer = _instantiate(model_config)
    if "imputer" in checkpoint:
        imputer.load_state_dict(checkpoint["imputer"])
    imputer.eval()
    return imputer


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


def get_quantile_indices(imputer: nn.Module, quantile_levels: List[float]) -> List[int]:
    """Map requested quantile levels to their index in TS-ICL's output head. Ported from
    ``TSICL._get_quantile_indices``."""
    start_quantile = imputer.tf_icl.start_quantile
    end_quantile = imputer.tf_icl.end_quantile
    nb_quantiles = imputer.tf_icl.nb_quantiles
    training_quantile_levels = np.linspace(start_quantile * 100, end_quantile * 100, nb_quantiles).tolist()

    requested = [100 * x for x in quantile_levels]
    if not set(requested).issubset(training_quantile_levels):
        raise ValueError(f"quantile_levels={quantile_levels} must be a subset of TS-ICL's trained quantiles.")
    return [training_quantile_levels.index(q) for q in requested]


def _predict_batch(imputer: nn.Module, grid: torch.Tensor, series_c: torch.Tensor) -> torch.Tensor:
    """Run one forward pass of the TS-ICL imputer on a batch, in the model's normalized
    space; the whole grid is queried, and imputed values at originally-observed positions
    are overwritten with the (normalized) ground truth. Ported (covariate-free) from
    ``TSICL._get_context_target_coords_i`` + ``TSICL._predict_batch``.
    """
    context = prepare_context_tensors(grid=grid, series_c=series_c)
    series_c_ctx, coords_c = context["series_c"], context["coords_c"]
    coords_t = grid

    scaler = CustomStandardScaler(dim=1, epsilon=1e-5)
    scaler.fit(series_c_ctx)
    series_c_norm = scaler.transform(series_c_ctx)

    query_coords = torch.cat([coords_c, coords_t], dim=1)

    quantiles, _ = imputer(
        series=series_c_norm,
        coords=coords_c,
        target_coords=query_coords,
        undo_asinh_transform=True,
    )

    # positions in coords_t matching an (originally-observed) context coordinate are
    # forced back to the ground truth, so only genuinely missing positions are model-predicted
    match = coords_t == coords_c.transpose(1, 2)
    b_idx, t1_idx, t2_idx = match.nonzero(as_tuple=True)
    quantiles[b_idx, t1_idx] = repeat(series_c_norm[b_idx, t2_idx], "n 1 -> n q", q=quantiles.shape[-1])

    return scaler.inv_transform(quantiles)


def impute_with_tsicl(
    imputer: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    point_estimator: str = "median",
    quantile_levels: Optional[List[float]] = None,
) -> np.ndarray:
    """Impute a `(n_samples, n_steps, n_features)` array with NaNs, feature by feature
    (TS-ICL models one series at a time; see the channel-independence note in ``TSICL``'s
    docstring).
    """
    quantile_levels = quantile_levels or [0.1, 0.3, 0.5, 0.7, 0.9]
    quantile_indices = get_quantile_indices(imputer, quantile_levels)
    median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else None
    if point_estimator == "median" and median_idx is None:
        point_estimator = "mean"

    n_samples, n_steps, n_features = X.shape
    flat = X.astype(np.float32).transpose(0, 2, 1).reshape(n_samples * n_features, n_steps)
    imputer = imputer.to(device)

    out = np.empty_like(flat)
    for start in range(0, len(flat), batch_size):
        batch = flat[start : start + batch_size]
        series_c = torch.tensor(batch, dtype=torch.float32, device=device).unsqueeze(-1)
        grid = make_grid(n_steps, num_samples=len(batch)).to(device)

        with torch.no_grad():
            quantiles = _predict_batch(imputer, grid, series_c)[..., quantile_indices]

        if point_estimator == "median":
            point = quantiles[..., median_idx : median_idx + 1]
        else:
            point = quantiles.mean(-1, keepdim=True)
        out[start : start + batch_size] = point.squeeze(-1).cpu().numpy()

    # safety net: fall back to the per-row mean (or 0 if the whole row is missing) wherever
    # the model still produced NaN, e.g. for rows with too few observed points to condition on
    if not np.isfinite(out).all():
        with warnings.catch_warnings():
            # nanmean on an all-NaN row (fully-missing series) warns; the nan_to_num fallback handles it
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            row_fallback = np.nan_to_num(
                np.nanmean(np.where(np.isfinite(flat), flat, np.nan), axis=1, keepdims=True), nan=0.0
            )
        out = np.where(np.isfinite(out), out, row_fallback)

    # belt and suspenders: observed values are carried through bit-exact, regardless of how
    # faithfully the model's own replace_by_gt round-trip (z-normalize / denormalize) preserved them
    observed = np.isfinite(flat)
    out = np.where(observed, flat, out)

    return out.reshape(n_samples, n_features, n_steps).transpose(0, 2, 1)
