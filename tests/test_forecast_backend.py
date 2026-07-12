"""
Tests for Forecastability Backends.

TensorFlow is optional.
Tests are skipped if TensorFlow
is not installed.
"""


import numpy as np

import pytest



# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def generate_predictable_series(
    n=600,
    seed=42,
):
    """
    Generate simple predictable AR series.
    """

    rng = np.random.RandomState(seed)


    x = np.zeros(n)


    noise = rng.normal(
        0,
        0.1,
        n
    )


    for i in range(1, n):

        x[i] = (

            0.8 * x[i-1]

            +

            noise[i]

        )


    return x



# ---------------------------------------------------
# Tests
# ---------------------------------------------------

def test_tensorflow_backend_import():

    """
    TensorFlow backend should be optional.
    """

    tf = pytest.importorskip(
        "tensorflow"
    )


    from predictability_score.forecast.tensorflow_backend import (
        TensorFlowBackend
    )


    assert TensorFlowBackend is not None



@pytest.mark.slow
def test_tensorflow_backend_execution():

    """
    Full backend execution test.

    Runs only when TensorFlow exists.
    """


    pytest.importorskip(
        "tensorflow"
    )


    from predictability_score.forecast.tensorflow_backend import (
        TensorFlowBackend
    )


    series = generate_predictable_series()



    backend = TensorFlowBackend(

        window=32,

        forecast_horizon=1,

        epochs=2,

        batch_size=32,

        verbose=0,

        random_state=42,

    )



    result = backend.evaluate(
        series
    )


    assert "score" in result

    assert "details" in result



    assert 0 <= result["score"] <= 100



def test_forecast_backend_interface():

    """
    Verify backend follows interface.
    """


    from predictability_score.forecast.base import (
        ForecastBackend
    )


    assert hasattr(
        ForecastBackend,
        "evaluate"
    )