"""
predictability_score
====================

A framework for measuring time series predictability.

The library combines:

- Stationarity
- Temporal Dependence
- Forecastability
- Null Significance
- Stability

into a single score [0,100].

Example
-------

from predictability_score import PredictabilityScore


evaluator = PredictabilityScore()

result = evaluator.evaluate(series)

print(result)

"""

from .core import PredictabilityScore

from .config import PredictabilityConfig

from .result import PredictabilityResult


__version__ = "0.1.0"


__author__ = "Jalal Mazaheri"


__all__ = [

    "PredictabilityScore",

    "PredictabilityConfig",

    "PredictabilityResult",

]