# Scoring Methodology

## Overview

`predictability-score` produces a single score between:


0 ───────────────────── 100


The score represents the degree to which a time series contains stable, statistically significant, and exploitable predictive structure.

The score is not a trading signal.

It measures the intrinsic forecasting difficulty of the signal.

---

# Conceptual Model


A predictable time series should satisfy several conditions simultaneously:


             Predictability

                  |
                  |
-----------------------------------------
|          |          |          |
v          v          v          v

Stationary Temporal Forecastable Stable

                  |
                  v

          Statistically Significant


A signal can fail in different ways:

- It may be stationary but have no memory.
- It may have autocorrelation but be unstable.
- It may be predictable in-sample but impossible out-of-sample.
- It may appear structured only because of randomness.

The scoring system separates these effects.

---

# Final Score


The final score is a weighted combination:



Predictability Score =
Weighted Combination(
Stationarity,
Temporal Dependence,
Forecastability,
Null Significance,
Stability
)



Each component is normalized:


0 <= component_score <= 100



---

# Components


## 1. Stationarity Score


### Question

Does the statistical behavior of the signal remain consistent over time?


### Measurements


The evaluator uses:


### Augmented Dickey-Fuller Test (ADF)

Tests:


H0:
Series contains a unit root
(non-stationary)



A low p-value indicates evidence of stationarity.



### KPSS Test

Tests:



H0:
Series is stationary



Both tests are combined to reduce false decisions from using a single hypothesis test.



### Interpretation


High score:

- Stable mean behavior.
- Suitable for statistical modeling.


Low score:

- Trend dominated.
- Regime drift.
- Non-stationary behavior.



---

# 2. Temporal Dependence Score


## Question


Does the past contain information about future values?


The evaluator measures:


## Ljung-Box Test


Detects:

- autocorrelation
- linear temporal dependence


Example:


X(t-5)
|
v
X(t)




---

## BDS Test


Detects:

- nonlinear dependence
- deviation from IID behavior


Null hypothesis:



Series is independently and identically distributed




Rejecting IID suggests exploitable structure.



---

## Durbin-Watson


Measures residual autocorrelation characteristics.



---

# 3. Forecastability Score


## Question


Can a forecasting model actually extract the structure?


This is different from statistical dependency.


A series can have:


Temporal structure = High

but

Forecastability = Low



because the pattern may be:

- too complex.
- too noisy.
- difficult for the selected model.



---

# Backend Architecture


Forecastability is model-independent through a backend interface.


Example:



ForecastBackend

   |
   +----------------+
   |

TensorFlow Backend

   |
   |

LSTM/TCN/etc.




The backend calculates out-of-sample performance.



Important:

Only future unseen data is evaluated.



---

# 4. Null Significance Score


## Question


Is the detected structure real?


The evaluator uses permutation testing.



Process:



Original Series

   |
   v

Measure temporal score

   vs

Shuffle Series

   |
   v

Create Null Distribution




The final score depends on how far the original signal is from random expectation.



Advantages:

- Detects false patterns.
- Protects against over-interpreting noise.
- Helps identify data leakage.



---

# 5. Stability Score


## Question


Does predictability remain consistent through time?


The series is divided into windows:



|----|----|----|----|----|

W1 W2 W3 W4 W5



Each window receives a score.


A stable signal:



90
85
88
91
87



An unstable signal:



90
20
85
10
70




The second case receives a lower stability score.



---

# Why Multiple Components?


A single metric is insufficient.


Example:


## Case 1

High autocorrelation:


Temporal = 90



but:


Stability = 20



The structure exists but disappears over time.



---

## Case 2

Stable:


Stationarity = 90



but:


Temporal = 10



The series is stable random noise.



---

## Case 3

Good ML performance:


Forecastability = 85



but:



Null Significance = 20



The model may be exploiting leakage or randomness.



---

# Interpretation Guide


| Score | Meaning |
|---|---|
| 0-20 | Almost random |
| 20-40 | Weak predictive structure |
| 40-60 | Moderate structure |
| 60-80 | Strong predictive potential |
| 80-100 | Highly structured signal |


---

# Financial Time Series Notes


Financial markets require special care.


Raw prices:


Gold Price
1000
1001
1005
1010



are usually non-stationary.


Common transformations:


## Log Return



log(
Price(t)
/
Price(t-1)
)



## OHLC Derived Features


Examples:



log(High/Open)

log(Low/Open)

Range

Volatility




The score should be calculated on the actual target variable used by the forecasting model.



---

# Limitations


A high score does not guarantee:


- profitability.
- positive expected return.
- absence of market regime changes.
- robustness to transaction costs.


The score only measures predictive structure.



---

# Summary


Predictability Score answers:


> "How much reliable and extractable information about the future exists in this time series?"


It combines:

- statistical validity.
- temporal structure.
- machine learning extractability.
- significance testing.
- temporal stability.


The result is a model-agnostic measure of signal quality.