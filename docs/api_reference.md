# API Reference

## Overview

This document describes the public API of `predictability-score`.

The public API is designed around:

- `PredictabilityScore`
- `PredictabilityConfig`
- `PredictabilityResult`
- Evaluators
- Forecast Backends

---

# Main Entry Point

## PredictabilityScore

Location:

```python
predictability_score.core

The main class responsible for running the complete evaluation pipeline.

Constructor
PredictabilityScore(
    config=None,
    forecast_backend=None
)
Parameters
Parameter	Type	Description
config	PredictabilityConfig	Global configuration
forecast_backend	ForecastBackend	Optional ML forecasting backend

Example:

from predictability_score import PredictabilityScore


evaluator = PredictabilityScore()
evaluate()
Signature
evaluate(series)
Parameters
series

Type:

numpy.ndarray

Requirements:

One dimensional
Ordered by time
No missing values
No future information leakage

Example:

result = evaluator.evaluate(
    series
)
Returns

PredictabilityResult

PredictabilityResult

Location:

predictability_score.result

Container object holding the evaluation output.

Attributes
total_score

Type:

float

Range:

0 <= total_score <= 100

Final predictability score.

Example:

print(
    result.total_score
)
stationarity

Contains stationarity evaluation.

Example:

result.stationarity.score

Contains:

ADF result
KPSS result
diagnostics
temporal

Contains temporal dependency evaluation.

Includes:

Ljung-Box
BDS
Durbin-Watson

Example:

result.temporal.score
forecastability

Contains forecasting model performance.

Available only when a ForecastBackend is provided.

Example:

result.forecastability.score
null_significance

Contains permutation significance analysis.

Example:

result.null_significance.score
stability

Contains temporal stability analysis.

Example:

result.stability.score
PredictabilityConfig

Location:

predictability_score.config

Configuration object controlling evaluation behavior.

Constructor:

PredictabilityConfig(
    ...
)

Example:

config = PredictabilityConfig(

    forecast_backend=None

)


evaluator = PredictabilityScore(

    config=config

)
StationarityEvaluator

Location:

predictability_score.stationarity
Purpose

Evaluates statistical stationarity.

Method
evaluate(series)

Returns:

{
    "score": float,
    "details": dict
}

Details include:

details = {

    "adf": {},

    "kpss": {}

}
TemporalEvaluator

Location:

predictability_score.temporal
Purpose

Measures temporal dependence.

Uses:

Ljung-Box
BDS
Durbin-Watson

Method:

evaluate(series)

Returns:

{
    "score": float,
    "details": dict
}
NullSignificanceEvaluator

Location:

predictability_score.null_significance
Purpose

Tests whether temporal structure is statistically significant.

Method:

evaluate(series)

Parameters:

iterations

Number of permutation experiments.

Example:

evaluator = NullSignificanceEvaluator(

    iterations=500

)
StabilityEvaluator

Location:

predictability_score.stability
Purpose

Measures consistency of predictive structure over time.

Method:

evaluate(series)

Parameters:

windows

Number of temporal partitions.

Example:

evaluator = StabilityEvaluator(

    windows=10

)
Forecast Backend Interface

Location:

predictability_score.forecast.base
ForecastBackend

Abstract interface for forecasting models.

Required method:

evaluate(series)

Return format:

{
    "score": float,

    "details": dict
}
TensorFlow Backend

Location:

predictability_score.forecast.tensorflow_backend
TensorFlowBackend

Optional LSTM-based forecasting backend.

Requires:

pip install tensorflow

Constructor:

TensorFlowBackend(

    window=64,

    forecast_horizon=1,

    epochs=20,

    batch_size=32

)

Parameters:

Parameter	Description
window	Historical sequence length
forecast_horizon	Prediction horizon
epochs	Training iterations
batch_size	Training batch size

Example:

backend = TensorFlowBackend(

    window=64,

    epochs=20

)


evaluator = PredictabilityScore(

    forecast_backend=backend

)
Complete Example
import numpy as np


from predictability_score import PredictabilityScore


series = np.random.randn(2000)



model = PredictabilityScore()



result = model.evaluate(

    series

)



print(

    result.total_score

)
Financial Example
import pandas as pd

import numpy as np


from predictability_score import PredictabilityScore



df = pd.read_csv(
    "gold.csv"
)



returns = (

    np.log(

        df["Close"]

        /

        df["Close"].shift(1)

    )

    .dropna()

    .values

)



result = PredictabilityScore().evaluate(

    returns

)



print(

    result.total_score

)
Error Handling

The library validates:

Input dimension
Missing values
Empty series
Invalid parameters

Example:

ValueError:
Series contains NaN values
Public API Stability

The following are considered stable public interfaces:

PredictabilityScore

PredictabilityConfig

PredictabilityResult

ForecastBackend