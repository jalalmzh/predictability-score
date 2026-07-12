"""
predictability_score.config
===========================

Global configuration objects for the Predictability Score library.

This module contains only lightweight configuration classes and
does not import any third-party scientific libraries.

Author
------
Your Name

License
-------
MIT
"""

from __future__ import annotations

from typing import Optional


class PredictabilityConfig:
    """
    Configuration container for PredictabilityScore.

    Parameters
    ----------
    window : int, default=32
        Number of historical observations used by the forecasting backend.

    forecast_horizon : int, default=1
        Prediction horizon.

    forecast_backend : str | None, default="tensorflow"
        Forecast backend name.

        Supported values:
            - "tensorflow"
            - None

    random_state : int, default=42
        Random seed.

    verbose : bool, default=False
        Enable progress logging.

    shuffle_iterations : int, default=100
        Number of surrogate series used by the null significance test.

    stability_windows : int, default=8
        Number of rolling windows used by the stability evaluator.
    """

    def __init__(
        self,
        window: int = 32,
        forecast_horizon: int = 1,
        forecast_backend: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
        shuffle_iterations: int = 100,
        stability_windows: int = 8,
    ) -> None:

        if window < 4:
            raise ValueError("window must be >= 4")

        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")

        if shuffle_iterations < 10:
            raise ValueError("shuffle_iterations must be >= 10")

        if stability_windows < 2:
            raise ValueError("stability_windows must be >= 2")

        if forecast_backend not in ("tensorflow", None):
            raise ValueError(
                "forecast_backend must be 'tensorflow' or None."
            )

        self.window = int(window)
        self.forecast_horizon = int(forecast_horizon)

        self.forecast_backend = forecast_backend

        self.random_state = int(random_state)

        self.verbose = bool(verbose)

        self.shuffle_iterations = int(shuffle_iterations)

        self.stability_windows = int(stability_windows)

    # ---------------------------------------------------------

    def copy(self) -> "PredictabilityConfig":
        """
        Return a shallow copy of the configuration.
        """

        return PredictabilityConfig(
            window=self.window,
            forecast_horizon=self.forecast_horizon,
            forecast_backend=self.forecast_backend,
            random_state=self.random_state,
            verbose=self.verbose,
            shuffle_iterations=self.shuffle_iterations,
            stability_windows=self.stability_windows,
        )

    # ---------------------------------------------------------

    def as_dict(self) -> dict:
        """
        Convert configuration to dictionary.
        """

        return {
            "window": self.window,
            "forecast_horizon": self.forecast_horizon,
            "forecast_backend": self.forecast_backend,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "shuffle_iterations": self.shuffle_iterations,
            "stability_windows": self.stability_windows,
        }

    # ---------------------------------------------------------

    def __repr__(self) -> str:

        return (
            "PredictabilityConfig("
            f"window={self.window}, "
            f"forecast_horizon={self.forecast_horizon}, "
            f"forecast_backend={self.forecast_backend!r}, "
            f"random_state={self.random_state}, "
            f"verbose={self.verbose}, "
            f"shuffle_iterations={self.shuffle_iterations}, "
            f"stability_windows={self.stability_windows}"
            ")"
        )