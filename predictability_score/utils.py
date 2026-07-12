"""
predictability_score.utils
==========================

Common utility functions used across
Predictability Score modules.

This module contains only helper functions.
"""

from __future__ import annotations


from typing import Any

import random

import numpy as np



def set_random_seed(seed):

    import tensorflow as tf

    tf.random.set_seed(seed)

    """
    Set reproducible random seeds.

    Parameters
    ----------
    seed:
        Random seed value.
    """

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



def validate_series(
    series: Any
) -> np.ndarray:
    """
    Convert input to clean one-dimensional numpy array.

    Parameters
    ----------
    series:
        Array-like time series.

    Returns
    -------
    np.ndarray
        Clean float64 series.

    Raises
    ------
    ValueError
        If series is invalid.
    """

    x = np.asarray(
        series,
        dtype=np.float64
    )


    if x.ndim != 1:

        raise ValueError(
            "Input series must be one-dimensional."
        )


    x = x[
        np.isfinite(x)
    ]


    if len(x) == 0:

        raise ValueError(
            "Series contains no valid values."
        )


    return x



def check_min_length(
    series: np.ndarray,
    minimum: int,
    name: str = "series"
) -> None:
    """
    Check minimum required length.

    Parameters
    ----------
    series:
        Input array.

    minimum:
        Minimum length.

    name:
        Variable name for error message.
    """

    if len(series) < minimum:

        raise ValueError(

            f"{name} length must be at least "
            f"{minimum}, got {len(series)}"

        )



def safe_mean(
    values
) -> float:
    """
    Calculate mean safely.

    Returns zero if input is empty.
    """

    values = np.asarray(
        values,
        dtype=float
    )


    if len(values) == 0:

        return 0.0


    return float(
        np.mean(values)
    )



def clip_score(
    value: float
) -> float:
    """
    Clip score to valid range [0,100].
    """

    return float(
        np.clip(
            value,
            0,
            100
        )
    )