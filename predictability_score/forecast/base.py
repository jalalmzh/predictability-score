"""
predictability_score.forecast.base
==================================

Abstract interface for forecastability backends.

A backend is responsible for measuring
how predictable a time series is using
a forecasting model.

This module has no dependency on any
machine learning framework.
"""

from __future__ import annotations


from abc import ABC
from abc import abstractmethod


from typing import Any
from typing import Dict



class ForecastBackend(ABC):
    """
    Abstract Forecastability Backend.

    Every forecasting backend must implement
    the evaluate method.

    Examples
    --------

    TensorFlow LSTM backend:

        ForecastBackend
                |
                |
        TensorFlowBackend

    Future possibilities:

        PyTorchBackend
        SklearnBackend
        TransformerBackend
    """

    def __init__(
        self,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:

        self.random_state = random_state

        self.verbose = verbose


    # -------------------------------------------------

    @abstractmethod
    def evaluate(
        self,
        series: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate forecastability.

        Parameters
        ----------
        series:
            One-dimensional time series.

        Returns
        -------
        dict

            Expected keys:

            score:
                Forecastability score [0,100]

            details:
                Diagnostic information.

        Notes
        -----
        Subclasses must implement this method.
        """

        raise NotImplementedError