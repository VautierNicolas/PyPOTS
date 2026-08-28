# Vendored (imputation-only) inference logic ported from the TS-ICL pipeline:
# https://github.com/EDF-Lab/ts-icl (src/tsicl/pipeline.py, src/tsicl/utils/*).
# Copyright (c) 2026 EDF SA. Licensed under the TS-ICL Non-Commercial License v1.0,
# NOT under PyPOTS' BSD-3-Clause license. See the NOTICE file in this directory
# for the full license text and restrictions (non-commercial research/evaluation use only).

"""
The core imputation logic of TS-ICL, ported from the (zero-shot, gradient-free)
``TSICL.impute()`` pipeline of the original implementation. Covariates are out of
scope here since PyPOTS' imputation contract has no notion of them. The checkpoint
loading and context/target grid plumbing shared with ``pypots.forecasting.tsicl``
live in ``pypots.nn.modules.tsicl``.

"""

import warnings
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import repeat

from ...nn.modules.tsicl import (
    CustomStandardScaler,
    get_quantile_indices,
    make_grid,
    prepare_context_tensors,
)


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
