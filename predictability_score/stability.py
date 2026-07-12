"""
predictability_score.stability
==============================

Temporal stability evaluator.

Measures whether temporal structure
remains consistent across different
time windows.

Uses TemporalEvaluator as the
underlying structural measure.
"""

from __future__ import annotations


from typing import Any, Dict, List


import numpy as np


from .temporal import TemporalEvaluator
from .utils import validate_series
from .utils import check_min_length
from .utils import clip_score



class StabilityEvaluator:
    """
    Evaluate temporal stability.

    Parameters
    ----------
    temporal_evaluator :
        Temporal structure evaluator.

    windows : int
        Number of rolling segments.

    min_length : int
        Minimum series length.
    """

    def __init__(
        self,
        temporal_evaluator: TemporalEvaluator = None,
        windows: int = 8,
        min_length: int = 1000,
    ) -> None:


        self.temporal_evaluator = (

            temporal_evaluator

            if temporal_evaluator is not None

            else TemporalEvaluator()

        )


        self.windows = windows

        self.min_length = min_length



    # --------------------------------------------------

    def _split_windows(
        self,
        x: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Split series into equal windows.
        """

        return list(

            np.array_split(

                x,

                self.windows

            )

        )



    # --------------------------------------------------

    def evaluate(
        self,
        series: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate temporal stability.

        Returns
        -------
        dict
            score and diagnostics.
        """

        x = validate_series(
            series
        )


        check_min_length(

            x,

            self.min_length

        )


        segments = self._split_windows(
            x
        )


        scores = []


        for segment in segments:


            if len(segment) < 50:

                continue


            result = (

                self.temporal_evaluator

                .evaluate(segment)

            )


            scores.append(

                result["score"]

            )



        scores = np.asarray(

            scores,

            dtype=float

        )


        if len(scores) < 2:

            return {

                "score": 0.0,

                "details":
                {

                    "reason":
                    "Not enough windows"

                }

            }



        mean_score = float(

            np.mean(scores)

        )


        std_score = float(

            np.std(scores)

        )


        cv = (

            std_score

            /

            (

                abs(mean_score)

                +

                1e-8

            )

        )


        stability_score = (

            1.0 - cv

        ) * 100.0



        stability_score = clip_score(

            stability_score

        )


        return {

            "score":
                float(stability_score),

            "details":
            {

                "window_scores":
                    scores.tolist(),

                "mean":
                    mean_score,

                "std":
                    std_score,

                "coefficient_of_variation":
                    float(cv),

                "windows":
                    len(scores)

            }

        }