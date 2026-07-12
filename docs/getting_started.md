\# Getting Started



\## Overview



`predictability-score` is a modular Python library for evaluating the predictive quality of a time series.



The library produces a final score between:





0 ─────────────── 100





where:



\- `0` means the series has weak statistical structure and low predictability.

\- `100` means the series is highly structured, stable, and easier for forecasting models to learn.







The score combines several independent components:



\- Stationarity

\- Temporal dependence

\- Forecastability

\- Null significance

\- Stability







\---



\# Installation



\## Core installation



The statistical core does not require deep learning frameworks.



Install:



```bash

pip install predictability-score



This installs:



numpy

scipy

pandas

statsmodels

scikit-learn

Optional TensorFlow Backend



Forecastability evaluation can use a neural forecasting backend.



TensorFlow is optional.



Install:



pip install predictability-score\[tensorflow]



The TensorFlow backend evaluates how well a sequence model can predict future values using out-of-sample performance.



Basic Usage

Import

import numpy as np



from predictability\_score import PredictabilityScore

Create a time series



Example:



series = np.random.randn(2000)



The input must be:



one-dimensional

ordered by time

without future information leakage

Calculate Predictability Score

evaluator = PredictabilityScore()





result = evaluator.evaluate(

&#x20;   series

)



The result contains:



result.total\_score



Example:



Predictability Score: 63.7 / 100

Access Individual Components



The final score is composed of several evaluations:



result.stationarity



result.temporal



result.forecastability



result.null\_significance



result.stability



Each component provides:



score

diagnostic details



Example:



print(

&#x20;   result.temporal.score

)

Working With Financial Data



The library is designed for general time series.



For financial applications:



Examples:



OHLC prices

returns

volatility

spreads

indicators



Recommended:



Avoid directly evaluating raw prices:



Gold price

1000

1001

1005

1010

...



Prefer stationary transformations:



log return



log(Close\_t / Close\_t-1)



or normalized features.



Example: Gold OHLC

import pandas as pd





df = pd.read\_csv(

&#x20;   "gold.csv"

)





close\_return = (



&#x20;   np.log(

&#x20;       df\["Close"]

&#x20;       /

&#x20;       df\["Close"].shift(1)

&#x20;   )



&#x20;   .dropna()



&#x20;   .values



)





score = PredictabilityScore().evaluate(

&#x20;   close\_return

)





print(

&#x20;   score.total\_score

)

Without Forecasting Models



The package works without TensorFlow.



Example:



evaluator = PredictabilityScore(



&#x20;   config={



&#x20;       "forecast\_backend": None



&#x20;   }



)



In this mode:



Only statistical components are calculated.



With Forecasting Backend



Example:



from predictability\_score.forecast import (

&#x20;   TensorFlowBackend

)





backend = TensorFlowBackend(



&#x20;   window=64,



&#x20;   epochs=20



)





evaluator = PredictabilityScore(



&#x20;   forecast\_backend=backend



)





result = evaluator.evaluate(

&#x20;   series

)

Interpreting Scores



General interpretation:



Score	Interpretation

0-20	Mostly random

20-40	Weak structure

40-60	Moderate predictability

60-80	Strong structure

80-100	Highly predictable



The score is not a trading signal.



It measures the statistical and modeling difficulty of forecasting a series.



Important Notes

Avoid Data Leakage



The evaluation must use only information available at prediction time.



Do not include:



future values

centered indicators

future normalization statistics

Financial Markets



High predictability score does not guarantee profitability.



A predictable variable can still have:



transaction costs

low economic value

unstable regimes

