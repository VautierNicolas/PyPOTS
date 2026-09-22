"""
Vendored from the TS-ICL model architecture: https://github.com/EDF-Lab/ts-icl
"""

# Created by Etienne Le Naour <etienne.le-naour@edf.fr>, Tahar Nabil <tahar.nabil@edf.fr>, and Adrien Petralia <adrien.petralia@gmail.com>
# License: BSD-3-Clause

from __future__ import annotations

from .attention import Attention, MultiScaleAttention, CrossAttention, FeedForward
from .utils import PreNorm, PreNormCross

__all__ = [
    "Attention",
    "MultiScaleAttention",
    "CrossAttention",
    "FeedForward",
    "PreNorm",
    "PreNormCross"

]