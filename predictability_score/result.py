"""
predictability_score.result
===========================

Result containers for Predictability Score.

This module contains only data structures.
No statistical computation is performed here.
"""

from __future__ import annotations


from typing import Any, Dict, Optional


import json



class PredictabilityResult:
    """
    Container for final predictability evaluation result.

    Parameters
    ----------
    total_score : float
        Final composite score [0,100].

    stationarity : float
        Stationarity component score.

    temporal : float
        Temporal dependency score.

    forecastability : float
        Forecastability score.

    null_significance : float
        Null hypothesis significance score.

    stability : float
        Stability score.

    details : dict, optional
        Detailed diagnostics from evaluators.
    """

    def __init__(
        self,
        total_score: float,
        stationarity: float,
        temporal: float,
        forecastability: float,
        null_significance: float,
        stability: float,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:


        self.total_score = float(
            total_score
        )

        self.stationarity = float(
            stationarity
        )

        self.temporal = float(
            temporal
        )

        self.forecastability = float(
            forecastability
        )

        self.null_significance = float(
            null_significance
        )

        self.stability = float(
            stability
        )


        self.details = (
            details
            if details is not None
            else {}
        )


    # --------------------------------------------------

    def to_dict(
        self
    ) -> Dict[str, Any]:
        """
        Convert result to dictionary.
        """

        return {

            "total_score":
                self.total_score,

            "components": {

                "stationarity":
                    self.stationarity,

                "temporal":
                    self.temporal,

                "forecastability":
                    self.forecastability,

                "null_significance":
                    self.null_significance,

                "stability":
                    self.stability,
            },

            "details":
                self.details
        }


    # --------------------------------------------------

    def to_json(
        self,
        indent: int = 4
    ) -> str:
        """
        Convert result to JSON string.
        """

        return json.dumps(

            self.to_dict(),

            indent=indent,

            ensure_ascii=False

        )


    # --------------------------------------------------

    def __getitem__(
        self,
        key: str
    ) -> Any:
        """
        Dictionary-like access.

        Example
        -------
        result["total_score"]
        """

        return self.to_dict()[key]


    # --------------------------------------------------

    def __repr__(
        self
    ) -> str:

        return (

            "PredictabilityResult("
            f"total_score={self.total_score:.2f}, "
            f"stationarity={self.stationarity:.2f}, "
            f"temporal={self.temporal:.2f}, "
            f"forecastability={self.forecastability:.2f}, "
            f"null_significance={self.null_significance:.2f}, "
            f"stability={self.stability:.2f}"
            ")"

        )


    # --------------------------------------------------

    def __str__(
        self
    ) -> str:

        lines = [

            "",
            "Predictability Score",
            "====================",
            f"Total Score        : {self.total_score:6.2f}",
            "",
            "Components",
            "----------",
            f"Stationarity       : {self.stationarity:6.2f}",
            f"Temporal           : {self.temporal:6.2f}",
            f"Forecastability    : {self.forecastability:6.2f}",
            f"Null Significance  : {self.null_significance:6.2f}",
            f"Stability          : {self.stability:6.2f}",

        ]


        return "\n".join(lines)