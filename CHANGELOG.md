\# Changelog



All notable changes to this project will be documented in this file.



The format is based on

\[Keep a Changelog](https://keepachangelog.com/)



This project follows

\[Semantic Versioning](https://semver.org/)







\# \[Unreleased]



\## Added



\- Future improvements will be listed here.







\# \[0.1.0] - 2026-07-11



Initial public release.





\## Added



\### Core Framework



\- Composite Predictability Score engine.

\- Unified evaluation interface.

\- Final score normalization to range \[0,100].





\### Statistical Evaluators



Added:



\- Stationarity evaluator:

&#x20; - Augmented Dickey-Fuller test.

&#x20; - KPSS test.





\- Temporal Dependence evaluator:

&#x20; - Ljung-Box autocorrelation analysis.

&#x20; - BDS nonlinear dependence test.

&#x20; - Durbin-Watson statistic.





\- Null Significance evaluator:



&#x20; - Permutation-based significance testing.

&#x20; - Comparison against shuffled null distribution.





\- Stability evaluator:



&#x20; - Window-based temporal consistency analysis.







\### Forecastability Architecture



Added:



\- Backend interface for forecasting models.

\- Optional TensorFlow backend.

\- LSTM-based out-of-sample forecast evaluation.





\### Package Infrastructure



Added:



\- Python package configuration.

\- PyPI-ready build system.

\- MIT License.

\- GitHub Actions CI.

\- Pytest test suite.

\- Documentation.







\## Design Decisions



\### Optional Deep Learning Dependencies



TensorFlow is not required for the core package.



Users can install:



```bash

pip install predictability-score\[tensorflow]

