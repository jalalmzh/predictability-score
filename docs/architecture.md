# Architecture

## Overview

`predictability-score` is designed as a modular evaluation framework for measuring the predictive quality of time series.

The architecture separates:

1. Statistical properties of the signal.
2. Temporal structure detection.
3. Machine learning forecastability.
4. Significance validation.
5. Stability analysis.

The main design goal is:

> Measure whether a time series contains exploitable predictive structure without coupling the library to a specific forecasting model.



---

# High-Level Architecture


             Time Series
                  |
                  v
          PredictabilityScore
                  |
  -----------------------------------
  |          |          |            |
  v          v          v            v

Stationarity Temporal Stability Null Significance

                  |
                  v

          Forecastability Backend

                  |
                  v

         Final Predictability Score



---

# Core Components


## PredictabilityScore

Location:


predictability_score/core.py



The main orchestration layer.

Responsibilities:

- Receive input series.
- Execute evaluators.
- Combine component scores.
- Return unified result.



Example:


```python
result = evaluator.evaluate(series)

The core engine does not know:

which forecasting model is used.
how stationarity is calculated.
how temporal dependence is measured.

This follows the principle:

High-level logic
        |
        |
Abstract interfaces
        |
        |
Concrete implementations
Evaluators
StationarityEvaluator

File:

stationarity.py

Measures:

ADF test
KPSS test

Purpose:

Determine whether the statistical properties of the signal are stable over time.

TemporalEvaluator

File:

temporal.py

Measures temporal dependence using:

Ljung-Box autocorrelation test
BDS nonlinear dependency test
Durbin-Watson statistic

Purpose:

Answer:

Does the past contain information about the future?

NullSignificanceEvaluator

File:

null_significance.py

Uses permutation testing.

Process:

Original Series

      |
      v

Temporal Score


      vs


Shuffled Series

      |
      v

Null Distribution

Purpose:

Determine whether detected structure is real or caused by chance.

StabilityEvaluator

File:

stability.py

Analyzes:

rolling windows
score variation
regime changes

Purpose:

Answer:

Does the predictive structure remain consistent through time?

Forecastability Backend
Design Principle

Forecastability is intentionally separated from the core package.

The library does not assume:

TensorFlow
PyTorch
XGBoost
Transformer
LSTM

Instead it defines an interface:

ForecastBackend
        |
        |
        +----------------+
        |                |
TensorFlow Backend   Future Backend
Backend Interface

Location:

forecast/base.py

Concept:

class ForecastBackend:

    def evaluate(series):
        pass

Any model can be integrated by implementing this interface.

Examples:

Future implementations:

forecast/

├── tensorflow_backend.py

├── pytorch_backend.py

├── transformer_backend.py

└── tcn_backend.py
TensorFlow Backend

Location:

forecast/tensorflow_backend.py

Characteristics:

Optional dependency.
Installed separately.
Performs out-of-sample evaluation.
Returns normalized score [0,100].

The core package remains usable without TensorFlow.

Data Flow

Complete pipeline:

Input Series

      |
      v

Validation

      |
      v

Stationarity Evaluation

      |
      v

Temporal Dependency Evaluation

      |
      v

Null Significance Testing

      |
      v

Stability Analysis

      |
      v

Forecastability Evaluation

      |
      v

Score Aggregation

      |
      v

PredictabilityResult
Result Object

Location:

result.py

The output is a structured object:

PredictabilityResult

Contains:

total_score

stationarity

temporal

forecastability

null_significance

stability

details

This avoids returning unstructured dictionaries.

Extension Guide
Adding a New Statistical Evaluator

Example:

new_metric.py

Implement:

class NewEvaluator:


    def evaluate(series):

        return {

            "score": value,

            "details": {}

        }

Register in:

core.py
Adding a New Forecast Backend

Create:

forecast/new_backend.py

Implement:

class NewBackend(ForecastBackend):


    def evaluate(self, series):

        ...

No changes required in:

Stationarity
Temporal
Stability
Core scoring logic
Dependency Philosophy

Core dependencies:

numpy
scipy
pandas
statsmodels
scikit-learn

Optional:

tensorflow

The architecture avoids forcing heavy ML dependencies on users.

Testing Architecture

Tests are divided into:

tests/

├── unit tests

│
├── integration tests

│
└── optional backend tests

TensorFlow tests:

run only when TensorFlow exists.
never break the statistical core.
Design Principles

The project follows:

Modularity

Each evaluator has one responsibility.

Extensibility

New models can be added without changing the framework.

Reproducibility

Tests use fixed random seeds.

No Data Leakage

All evaluation methods are designed for time-ordered data.

Model Agnostic

The framework measures signal quality, not model preference.