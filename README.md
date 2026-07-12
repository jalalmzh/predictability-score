\# Predictability Score



A Python framework for measuring time series predictability.



Predictability Score evaluates whether a time series has the properties required for successful forecasting models such as:



\- LSTM

\- TCN

\- Transformer

\- Classical forecasting models



The library produces a single score between:





0 -------------------- 100

Impossible Highly Predictable





\---



\# Why Predictability Score?



Not every time series is equally learnable.



A series may fail forecasting models because:



\- it is non-stationary,

\- it has weak temporal structure,

\- the existing structure is unstable,

\- apparent patterns are caused by randomness,

\- the forecasting model cannot extract available information.



Predictability Score separates these factors and evaluates them independently.



\---



\# Evaluation Architecture



The final score is a weighted combination of five independent components.



\\\[

Score =

0.20S\_{stationarity}

\+

0.25S\_{temporal}

\+

0.30S\_{forecastability}

\+

0.15S\_{null}

\+

0.10S\_{stability}

\\]



\---



\## Components



\## 1. Stationarity (20%)



Measures whether the statistical properties of the series are stable.



Methods:



\- Augmented Dickey-Fuller (ADF)

\- KPSS test





Output:





0 - 100





Higher values indicate stronger stationarity.



\---



\## 2. Temporal Dependence (25%)



Measures whether temporal structure exists independently of any forecasting model.



Methods:



\### Ljung-Box



Detects linear autocorrelation.



\### BDS Test



Detects nonlinear dependence and rejection of IID behavior.



\### Durbin-Watson



Measures short-term autocorrelation.





This component answers:



> "Does the series contain temporal information?"



It does not measure whether a model can exploit it.



\---



\## 3. Forecastability (30%)



Measures practical predictive performance.



This component is backend-based.



Currently supported:



\- TensorFlow LSTM backend





The backend interface allows future implementations:



\- PyTorch

\- Transformer

\- TCN

\- Classical ML models





Forecastability is measured using:



\- Out-of-sample R²



\---



\## 4. Null Significance (15%)



Tests whether detected temporal structure is real.



Method:



Permutation test.



Process:



1\. Shuffle the series.

2\. Destroy temporal ordering.

3\. Calculate temporal score repeatedly.

4\. Compare the real series against the null distribution.





A high score means:



> The observed structure is unlikely to be random.



\---



\## 5. Stability (10%)



Measures whether predictability remains consistent over time.



The series is divided into multiple windows.



Each window is evaluated separately.



A stable series has:



\- similar scores,

\- low variance,

\- consistent behavior.



\---



\# Installation



\## Basic installation



```bash

pip install -r requirements.txt

With TensorFlow Forecastability Backend

pip install -r requirements-tensorflow.txt

Quick Example

import numpy as np



from predictability\_score import PredictabilityScore





\# Example time series



series = np.random.randn(5000)





evaluator = PredictabilityScore()





result = evaluator.evaluate(series)





print(result)



Example output:



Predictability Score

====================



Total Score        : 73.8



Components

\----------



Stationarity       : 91.2

Temporal           : 65.4

Forecastability    : 70.8

Null Significance  : 88.5

Stability          : 54.1

Financial Time Series Example



For financial applications:



Recommended input:



log returns

volatility series

normalized OHLC transformations



Example:



returns = np.log(

&#x20;   close / close.shift(1)

)



result = evaluator.evaluate(

&#x20;   returns.dropna()

)

Optional TensorFlow Backend



TensorFlow is intentionally optional.



The core library works without TensorFlow.



When enabled:



config = PredictabilityConfig(

&#x20;   forecast\_backend="tensorflow"

)





evaluator = PredictabilityScore(

&#x20;   config=config

)



The TensorFlow backend trains a lightweight LSTM model and evaluates out-of-sample performance.



Project Structure

predictability-score/



│

├── predictability\_score/

│

│   ├── core.py

│   ├── config.py

│   ├── result.py

│   ├── utils.py

│   │

│   ├── stationarity.py

│   ├── temporal.py

│   ├── null\_significance.py

│   ├── stability.py

│   │

│   └── forecast/

│       ├── base.py

│       └── tensorflow\_backend.py

│

├── tests/

│

├── README.md

│

├── pyproject.toml

│

└── requirements.txt

Design Philosophy



Predictability Score follows three principles:



1\. Separate structure from model capability



A series can contain temporal information but still be difficult for a model.



Therefore:



Temporal Dependence

&#x20;       !=

Forecastability

2\. No mandatory machine learning dependency



Statistical evaluation works independently.



Deep learning backends are optional.



3\. Modular backend architecture



Forecasting engines can be added without changing the core framework.



Roadmap



Future backends:



PyTorch LSTM

Temporal Convolution Network (TCN)

Transformer Encoder

Classical ML backend



Future statistical modules:



Spectral predictability

Entropy measures

Complexity analysis

License



MIT License





\---



\## Project status



```text

predictability-score/



├── README.md                      ✅

├── pyproject.toml                 ✅

├── requirements.txt               ✅

├── requirements-tensorflow.txt    ✅

│

└── predictability\_score/

&#x20;   ├── \_\_init\_\_.py                ✅

&#x20;   ├── core.py                    ✅

&#x20;   ├── config.py                  ✅

&#x20;   ├── result.py                  ✅

&#x20;   ├── utils.py                   ✅

&#x20;   ├── stationarity.py            ✅

&#x20;   ├── temporal.py                ✅

&#x20;   ├── null\_significance.py       ✅

&#x20;   ├── stability.py               ✅

&#x20;   │

&#x20;   └── forecast/

&#x20;       ├── \_\_init\_\_.py            ✅

&#x20;       ├── base.py                ✅

&#x20;       └── tensorflow\_backend.py  ✅

