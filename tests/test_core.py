"""
Tests for PredictabilityScore core engine.
"""


import numpy as np


from predictability_score import (
    PredictabilityScore
)


from predictability_score import (
    PredictabilityConfig
)



# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def generate_stationary_series(
    n=3000,
    seed=42,
):
    """
    Generate predictable stationary AR(1).
    """

    rng = np.random.RandomState(seed)


    x = np.zeros(n)


    noise = rng.normal(
        0,
        1,
        n
    )


    for i in range(1, n):

        x[i] = (

            0.85 * x[i-1]

            +

            noise[i]

        )


    return x



def generate_noise(
    n=3000,
    seed=42,
):
    """
    Generate random noise.
    """

    rng = np.random.RandomState(seed)


    return rng.normal(
        0,
        1,
        n
    )



# ---------------------------------------------------
# Tests
# ---------------------------------------------------

def test_core_output_type():

    series = generate_stationary_series()


    evaluator = PredictabilityScore()


    result = evaluator.evaluate(
        series
    )


    assert hasattr(
        result,
        "total_score"
    )



def test_total_score_range():

    series = generate_stationary_series()


    evaluator = PredictabilityScore()


    result = evaluator.evaluate(
        series
    )


    assert 0 <= result.total_score <= 100



def test_all_components_exist():

    series = generate_stationary_series()


    evaluator = PredictabilityScore()


    result = evaluator.evaluate(
        series
    )


    assert result.stationarity is not None

    assert result.temporal is not None

    assert result.forecastability is not None

    assert result.null_significance is not None

    assert result.stability is not None



def test_details_are_available():

    series = generate_stationary_series()


    evaluator = PredictabilityScore()


    result = evaluator.evaluate(
        series
    )


    assert isinstance(
        result.details,
        dict
    )


    assert "stationarity" in result.details

    assert "temporal" in result.details

    assert "null_significance" in result.details

    assert "stability" in result.details



def test_structured_series_scores_higher_than_noise():

    structured = generate_stationary_series()

    noise = generate_noise()


    evaluator = PredictabilityScore()


    structured_result = evaluator.evaluate(
        structured
    )


    noise_result = evaluator.evaluate(
        noise
    )


    assert (

        structured_result.total_score

        >

        noise_result.total_score

    )



def test_custom_config_without_forecast_backend():

    config = PredictabilityConfig(

        forecast_backend=None

    )


    evaluator = PredictabilityScore(
        config=config
    )


    result = evaluator.evaluate(

        generate_stationary_series()

    )


    assert result.forecastability == 0.0