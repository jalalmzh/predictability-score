"""
predictability_score.null_significance
======================================

Null significance evaluator.

Tests whether observed temporal
structure is significantly higher
than randomized surrogate data.

Method:

1. Calculate temporal score of original series.
2. Shuffle series N times.
3. Calculate temporal score of shuffled series.
4. Compute percentile rank.

A high score means:
    observed structure is unlikely
    to be produced by chance.
"""

from __future__ import annotations


from typing import Any, Dict, List


import numpy as np


from .temporal import TemporalEvaluator
from .utils import validate_series
from .utils import check_min_length
from .utils import clip_score



class NullSignificanceEvaluator:
    """
    Permutation based significance evaluator.

    Parameters
    ----------
    temporal_evaluator :
        Temporal dependence evaluator.

    iterations : int
        Number of shuffled surrogate series.

    random_state : int
        Random seed.

    min_length : int
        Minimum series length.
    """

    def __init__(
        self,
        temporal_evaluator: TemporalEvaluator = None,
        iterations: int = 100,
        random_state: int = 42,
        min_length: int = 300,
    ) -> None:


        self.temporal_evaluator = (

            temporal_evaluator

            if temporal_evaluator is not None

            else TemporalEvaluator()

        )


        self.iterations = iterations

        self.random_state = random_state

        self.min_length = min_length



    # --------------------------------------------------

    def _shuffle_series(
        self,
        x: np.ndarray,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """
        Create shuffled surrogate.
        """

        shuffled = np.array(
            x,
            copy=True
        )


        rng.shuffle(
            shuffled
        )


        return shuffled



    # --------------------------------------------------

    def evaluate(
        self,
        series: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate null significance.

        Returns
        -------
        dict

            score:
                percentile rank [0,100]

            details:
                diagnostics
        """

        x = validate_series(
            series
        )


        check_min_length(

            x,

            self.min_length

        )


        rng = np.random.RandomState(

            self.random_state

        )


        # ------------------------------
        # Real temporal score
        # ------------------------------

        real_result = (

            self.temporal_evaluator

            .evaluate(x)

        )


        real_score = (

            real_result["score"]

        )


        # ------------------------------
        # Null distribution
        # ------------------------------

        null_scores: List[float] = []


        for _ in range(
            self.iterations
        ):

            shuffled = self._shuffle_series(

                x,

                rng

            )


            result = (

                self.temporal_evaluator

                .evaluate(shuffled)

            )


            null_scores.append(

                result["score"]

            )



        null_scores = np.asarray(

            null_scores,

            dtype=float

        )


        # ------------------------------
        # percentile rank
        # ------------------------------

        percentile = (

            np.mean(

                null_scores < real_score

            )

            *

            100

        )


        percentile = clip_score(

            percentile

        )


        return {

            "score":
                float(percentile),

            "details":
            {

                "real_temporal_score":
                    float(real_score),

                "null_mean":
                    float(
                        np.mean(null_scores)
                    ),

                "null_std":
                    float(
                        np.std(null_scores)
                    ),

                "iterations":
                    self.iterations,

                "percentile":
                    float(percentile),

            }

        }