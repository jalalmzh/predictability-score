"""
predictability_score.core
=========================

Main public interface for Predictability Score.

This module combines all evaluators
and produces the final composite score.
"""

from __future__ import annotations


from typing import Any, Dict, Optional


from .config import PredictabilityConfig

from .result import PredictabilityResult

from .stationarity import StationarityEvaluator

from .temporal import TemporalEvaluator

from .null_significance import NullSignificanceEvaluator

from .stability import StabilityEvaluator



class PredictabilityScore:
    """
    Main Predictability Score evaluator.

    Parameters
    ----------
    config :
        Global configuration.

    Notes
    -----
    Score components:

        Stationarity        20%
        Temporal            25%
        Forecastability     30%
        Null Significance   15%
        Stability           10%
    """

    # fixed weights

    WEIGHTS = {

        "stationarity": 0.20,

        "temporal": 0.25,

        "forecastability": 0.30,

        "null_significance": 0.15,

        "stability": 0.10,

    }


    def __init__(
        self,
        config: Optional[PredictabilityConfig] = None,
    ) -> None:


        self.config = (

            config

            if config is not None

            else PredictabilityConfig()

        )


        self.stationarity = (
            StationarityEvaluator()
        )


        self.temporal = (
            TemporalEvaluator()
        )


        self.null_significance = (

            NullSignificanceEvaluator(

                temporal_evaluator=
                    self.temporal,

                iterations=
                    self.config.shuffle_iterations,

                random_state=
                    self.config.random_state,

            )

        )


        self.stability = (

            StabilityEvaluator(

                temporal_evaluator=
                    self.temporal,

                windows=
                    self.config.stability_windows,

            )

        )


        self.forecast_backend = (

            self._create_forecast_backend()

        )


    # -------------------------------------------------

    def _create_forecast_backend(
        self,
    ):


        backend = self.config.forecast_backend


        if backend is None:

            return None



        if backend == "tensorflow":

            from .forecast.tensorflow_backend import (
                TensorFlowBackend
            )


            return TensorFlowBackend(

                window=
                    self.config.window,

                forecast_horizon=
                    self.config.forecast_horizon,

                random_state=
                    self.config.random_state,

                verbose=
                    self.config.verbose,

            )


        raise ValueError(

            f"Unknown backend: {backend}"

        )


    # -------------------------------------------------

    def evaluate(
        self,
        series: Any,
    ) -> PredictabilityResult:
        """
        Evaluate a time series.

        Parameters
        ----------
        series :
            One dimensional time series.

        Returns
        -------
        PredictabilityResult
        """


        details = {}



        # ----------------------------
        # Stationarity
        # ----------------------------

        stat = (

            self.stationarity

            .evaluate(series)

        )


        details["stationarity"] = stat["details"]



        # ----------------------------
        # Temporal
        # ----------------------------

        temporal = (

            self.temporal

            .evaluate(series)

        )


        details["temporal"] = temporal["details"]



        # ----------------------------
        # Null significance
        # ----------------------------

        null = (

            self.null_significance

            .evaluate(series)

        )


        details["null_significance"] = (
            null["details"]
        )



        # ----------------------------
        # Stability
        # ----------------------------

        stability = (

            self.stability

            .evaluate(series)

        )


        details["stability"] = (
            stability["details"]
        )



        # ----------------------------
        # Forecastability
        # ----------------------------

        if self.forecast_backend is not None:


            forecast = (

                self.forecast_backend

                .evaluate(series)

            )


            forecast_score = (
                forecast["score"]
            )


            details["forecastability"] = (
                forecast["details"]
            )


        else:

            forecast_score = 0.0


            details["forecastability"] = {

                "disabled": True

            }



        # ----------------------------
        # Final score
        # ----------------------------

        total = (

            self.WEIGHTS["stationarity"]
            *
            stat["score"]

            +

            self.WEIGHTS["temporal"]
            *
            temporal["score"]

            +

            self.WEIGHTS["forecastability"]
            *
            forecast_score

            +

            self.WEIGHTS["null_significance"]
            *
            null["score"]

            +

            self.WEIGHTS["stability"]
            *
            stability["score"]

        )



        return PredictabilityResult(

            total_score=total,

            stationarity=
                stat["score"],

            temporal=
                temporal["score"],

            forecastability=
                forecast_score,

            null_significance=
                null["score"],

            stability=
                stability["score"],

            details=details,

        )