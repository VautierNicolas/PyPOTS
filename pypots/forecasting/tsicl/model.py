# Vendored (forecasting-only) inference logic ported from the TS-ICL pipeline:
# https://github.com/EDF-Lab/ts-icl. Copyright (c) 2026 EDF SA. Licensed under the
# TS-ICL Non-Commercial License v1.0, NOT under PyPOTS' BSD-3-Clause license.
# See the NOTICE file in this directory for the full license text and restrictions
# (non-commercial research/evaluation use only).

"""
The implementation of TS-ICL for the partially-observed time-series forecasting task.

"""

# Created by Nicolas Vautier <nicolas.vautier@edf.fr>
# License: BSD-3-Clause for this file; the vendored TS-ICL architecture and weights
# it loads are under the TS-ICL Non-Commercial License v1.0, see this package's NOTICE.

import warnings

import torch

from ...nn.modules.tsicl import build_tsicl_network, fetch_X, load_tsicl_checkpoint
from ..base import BaseForecaster
from .core import forecast_with_tsicl


class TSICL(BaseForecaster):
    """The PyPOTS wrapper of the TS-ICL time-series foundation model :cite:`lenaour2026tsicl`.

    TS-ICL is a pretrained time-indexed foundation model that forecasts **zero-shot**,
    without any gradient step: its architecture and pretrained checkpoint are vendored
    into PyPOTS (see ``pypots.nn.modules.tsicl``), and ``fit()`` is a no-op, the model
    having no fine-tuning procedure in the original implementation this is ported from.

    Warnings
    --------
    Unlike the rest of PyPOTS, which is BSD-3-Clause, **the TS-ICL architecture and
    pretrained checkpoints are licensed by EDF SA under the TS-ICL Non-Commercial
    License v1.0** (for testing, evaluation, research, and benchmarking only — not
    commercial or production use, and not as part of a hosted/SaaS service). See
    ``pypots/nn/modules/tsicl/NOTICE`` for the full license text, or
    https://github.com/EDF-Lab/ts-icl for a commercial-license inquiry.

    Parameters
    ----------
    n_pred_steps :
        The number of steps ahead to forecast. Unlike PyPOTS' trainable forecasters,
        this isn't used to size any layer (TS-ICL is zero-shot), only to know how far
        ahead ``predict()`` should forecast, since PyPOTS' forecasting contract doesn't
        pass a horizon at prediction time.

    model_path :
        Path to a local TS-ICL checkpoint. If None, the checkpoint is downloaded
        from the Hugging Face Hub (repo ``taharnbl/TS-ICL``) on first use and cached
        locally, the same way PyPOTS' other pretrained backbones (e.g. GPT4TS, MOMENT)
        fetch their weights.

    checkpoint_version :
        Checkpoint release to fetch from the Hub when ``model_path`` is None.

    allow_auto_download :
        Whether to automatically download the checkpoint from the Hub when it isn't
        already cached locally / at ``model_path``.

    context_length :
        Maximum number of most-recent lookback steps to condition the forecast on.
        Defaults to the model's maximum context length (4096 for ``tsicl-v1``). If the
        input series is longer, only its most recent ``context_length`` steps are used;
        unlike imputation, this is a silent truncation rather than an error, since
        forecasting a long history through a short lookback window is a normal use case.

    batch_size :
        Number of univariate series forecast per forward pass. Note that a
        multivariate sample contributes ``n_features`` series (see the note on
        channel independence below).

    point_estimator :
        How the point estimate is derived from the predicted quantiles, either
        ``'median'`` or ``'mean'``. The median is optimal under absolute error,
        so it is the better default when evaluating with MAE.

    quantile_levels :
        Quantile levels requested from the model. Only used to derive the point
        estimate here; PyPOTS' forecasting contract returns point values only.

    device :
        The device for the model to run on.

    saving_path :
        The path for automatically saving model checkpoints and tensorboard files.
        Will not save if not given.

    verbose :
        Whether to print out the logs during the process.

    Notes
    -----
    **Channel independence.** TS-ICL models one series at a time. A multivariate
    sample of shape ``(n_steps, n_features)`` is therefore forecast feature by
    feature, and cross-feature correlation is not exploited. On datasets where
    features are strongly correlated this is a genuine handicap against
    multivariate models such as DLinear or TimeMixer, and results should be
    read with that in mind.

    **Long-horizon rollout.** TS-ICL predicts at most ``max_target_len`` steps
    (672 for ``tsicl-v1``) per forward pass; forecasting further ahead rolls the
    context forward autoregressively, feeding each chunk's prediction back in as
    context for the next, the same way the original implementation does.

    """

    def __init__(
        self,
        n_pred_steps: int,
        model_path: str | None = None,
        checkpoint_version: str = "tsicl-v1.ckpt",
        allow_auto_download: bool = True,
        context_length: int | None = None,
        batch_size: int = 32,
        point_estimator: str = "median",
        quantile_levels: list[float] | None = None,
        device: str | torch.device | list | None = None,
        saving_path: str = None,
        verbose: bool = True,
    ):
        super().__init__(device=device, saving_path=saving_path, verbose=verbose)

        assert point_estimator in ["median", "mean"]
        assert n_pred_steps > 0, f"n_pred_steps should be a positive int, but got {n_pred_steps}"

        self.n_pred_steps = n_pred_steps
        self.context_length = context_length
        self.batch_size = batch_size
        self.point_estimator = point_estimator
        self.quantile_levels = quantile_levels or [0.1, 0.3, 0.5, 0.7, 0.9]

        checkpoint = load_tsicl_checkpoint(model_path, checkpoint_version, allow_auto_download)
        self.forecaster = build_tsicl_network(checkpoint, "forecaster")
        self.max_context_length = checkpoint["config"]["max_context_len"]
        self.max_target_length = checkpoint["config"]["max_target_len"]

    def fit(
        self,
        train_set: dict | str,
        val_set: dict | str | None = None,
        file_type: str = "hdf5",
    ) -> None:
        """Do nothing: TS-ICL is used zero-shot and has no fine-tuning procedure.

        Warnings
        --------
        TS-ICL has no parameter to train in this integration. Please run func
        ``predict()`` directly.

        """
        warnings.warn(
            "TS-ICL is a pretrained foundation model used zero-shot here and has no "
            "fine-tuning procedure in this integration. Please run func `predict()` directly."
        )

    def predict(
        self,
        test_set: dict | str,
        file_type: str = "hdf5",
        **kwargs,
    ) -> dict:
        X = fetch_X(test_set, file_type)

        forecasting_data = forecast_with_tsicl(
            forecaster=self.forecaster,
            X=X,
            prediction_length=self.n_pred_steps,
            max_context_length=self.max_context_length,
            max_target_length=self.max_target_length,
            batch_size=self.batch_size,
            device=self.device,
            point_estimator=self.point_estimator,
            quantile_levels=self.quantile_levels,
            context_length=self.context_length,
        )
        return {"forecasting": forecasting_data}
