"""
Stationarity evaluation module.

Provides stationarity scoring based on KPSS and ADF tests.
"""

from __future__ import annotations

import warnings

import numpy as np

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

from .utils import validate_series


class StationarityEvaluator:
    """
    Evaluate stationarity of a time series.

    Output:
        score: float in range [0,100]

    Higher score means stronger evidence of stationarity.
    """

    def __init__(
        self,
        alpha: float = 0.05,
    ):
        self.alpha = alpha


    def evaluate(self, series) -> float:
        """
        Calculate stationarity score.

        Parameters
        ----------
        series:
            1D time series

        Returns
        -------
        float
            Stationarity score [0,100]
        """

        validate_series(series)

        x = np.asarray(series, dtype=float)

        # remove invalid values
        x = x[np.isfinite(x)]

        if len(x) < 50:
            return 0.0


        adf_score = self._adf_score(x)

        kpss_score = self._kpss_score(x)


        # combine both tests
        score = (
            0.5 * adf_score +
            0.5 * kpss_score
        )

        final_score = float(
            np.clip(score, 0, 100)
        )

        return {
            "score": final_score,
            "details": {
                "adf_score": adf_score,
                "kpss_score": kpss_score,
                "alpha": self.alpha
            }
        }


    def _adf_score(self, series):

        """
        ADF:
        Null hypothesis = unit root (non stationary)

        Lower p-value is better.
        """

        try:

            result = adfuller(
                series,
                autolag="AIC"
            )

            pvalue = result[1]


            if pvalue <= 0.01:
                return 100.0

            elif pvalue <= 0.05:
                return 80.0

            elif pvalue <= 0.1:
                return 50.0

            else:
                return 0.0


        except Exception:

            return 0.0



    def _kpss_score(self, series):

        """
        KPSS:
        Null hypothesis = stationary

        Higher p-value is better.
        """

        try:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore",
                    InterpolationWarning
                )


                statistic, pvalue, _, _ = kpss(
                    series,
                    regression="c",
                    nlags="auto"
                )


            if pvalue >= 0.1:
                return 100.0

            elif pvalue >= 0.05:
                return 70.0

            elif pvalue >= 0.01:
                return 30.0

            else:
                return 0.0


        except Exception:

            return 0.0