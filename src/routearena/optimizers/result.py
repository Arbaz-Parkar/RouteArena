"""Shared result object returned by all optimizers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptimizeResult:
    """Outcome and search history for one optimization run."""

    x: np.ndarray
    fun: float
    path: np.ndarray
    nit: int
    nfev: int
    njev: int
    success: bool
    message: str
