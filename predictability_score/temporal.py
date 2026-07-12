"""
predictability_score.temporal
=============================

Temporal dependence evaluator.

Measures whether a time series contains
temporal structure using:

- Ljung-Box test
- BDS test
- Durbin-Watson statistic

This module does not evaluate model
forecasting ability. That is handled
by Forecastability backends.
"""

from __future__ import annotations


from typing import Any, Dict


import numpy as np


from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import bds


from .utils import validate_series
from .utils import check_min_length
from .utils import clip_score



class TemporalEvaluator:
    """
    Evaluate temporal dependence of a series.

    Parameters
    ----------
    max_lag : int
        Maximum lag used for Ljung-Box.

    bds_max_dim : int
        Maximum embedding dimension for BDS.

    min_length : int
        Minimum required observations.
    """

    def __init__(
        self,
        max_lag: int = 20,
        bds_max_dim: int = 2,
        min_length: int = 300,
    ) -> None:

        self.max_lag = max_lag

        self.bds_max_dim = bds_max_dim

        self.min_length = min_length


    # --------------------------------------------------

    def _ljung_box(
        self,
        x: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Ljung-Box dependency score.

        Low p-value means significant
        autocorrelation exists.
        """

        result = acorr_ljungbox(

            x,

            lags=[self.max_lag],

            return_df=True

        )


        p_value = float(

            result["lb_pvalue"]
            .iloc[0]

        )


        score = np.clip(

            -np.log10(
                max(p_value, 1e-10)
            )
            /
            3.0,

            0,

            1

        ) * 100


        return {

            "p_value":
                p_value,

            "score":
                float(score)

        }


    # --------------------------------------------------

    def _bds(
        self,
        x: np.ndarray,
    ) -> Dict[str, Any]:
        """
        BDS nonlinear dependency score.

        Null hypothesis:
            series is IID.

        Strong rejection means
        temporal structure exists.
        """

        try:

            statistic, p_value = bds(

                x,

                max_dim=self.bds_max_dim

            )


            if np.ndim(p_value) > 0:

                p_value = float(
                    p_value[-1]
                )

            else:

                p_value = float(
                    p_value
                )


        except Exception:

            p_value = 1.0



        score = np.clip(

            -np.log10(
                max(p_value, 1e-10)
            )
            /
            3.0,

            0,

            1

        ) * 100



        return {

            "p_value":
                p_value,

            "score":
                float(score)

        }


    # --------------------------------------------------

    def _durbin_watson(
        self,
        x: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Durbin-Watson based dependency score.

        DW = 2 means no autocorrelation.

        Values far from 2 indicate
        temporal dependency.
        """

        dw = float(

            durbin_watson(
                x
            )

        )


        # distance from white noise point

        deviation = abs(
            dw - 2.0
        )


        score = np.clip(

            deviation / 2.0,

            0,

            1

        ) * 100



        return {

            "statistic":
                dw,

            "score":
                float(score)

        }


    # --------------------------------------------------

    def evaluate(
        self,
        series: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate temporal dependence.

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


        lb = self._ljung_box(
            x
        )


        bds_result = self._bds(
            x
        )


        dw = self._durbin_watson(
            x
        )


        score = (

            0.4 * lb["score"]

            +

            0.4 * bds_result["score"]

            +

            0.2 * dw["score"]

        )


        return {

            "score":
                clip_score(score),

            "details":
            {

                "ljung_box":
                    lb,

                "bds":
                    bds_result,

                "durbin_watson":
                    dw,

                "method":
                    "Ljung-Box + BDS + Durbin-Watson"

            }

        }