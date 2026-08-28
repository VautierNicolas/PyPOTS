# Vendored (forecasting-only) inference logic ported from the TS-ICL pipeline:
# https://github.com/EDF-Lab/ts-icl. Copyright (c) 2026 EDF SA. Licensed under the
# TS-ICL Non-Commercial License v1.0, NOT under PyPOTS' BSD-3-Clause license.
# See the NOTICE file in this directory for the full license text and restrictions
# (non-commercial research/evaluation use only).

"""
The package of the partially-observed time-series forecasting method TS-ICL.

Refer to the paper
`Etienne Le Naour, Tahar Nabil, and Adrien Petralia.
"TS-ICL: A Flexible Time-Indexed Foundation Model for Time Series via In-Context Learning".
arXiv preprint arXiv:2606.05878, 2026.
<https://arxiv.org/abs/2606.05878>`_

Notes
-----
This implementation is ported from the official one https://github.com/EDF-Lab/ts-icl,
which is licensed under the TS-ICL Non-Commercial License v1.0 (not PyPOTS' BSD-3-Clause
license) — see the NOTICE file in this directory.

"""

from .model import TSICL

__all__ = [
    "TSICL",
]
