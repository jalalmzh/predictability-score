"""
Tests for TemporalEvaluator.
"""


import numpy as np


from predictability_score.temporal import (
    TemporalEvaluator
)



# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def generate_ar1(
    n=2000,
    phi=0.8,
    seed=42,
):
    """
    Generate AR(1) stationary process.
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
            phi * x[i-1]
            +
            noise[i]
        )


    return x



def generate_white_noise(
    n=2000,
    seed=42,
):
    """
    Generate IID Gaussian noise.
    """

    rng = np.random.RandomState(seed)


    return rng.normal(
        0,
        1,
        n
    )



def generate_nonlinear_series(
    n=2000,
):
    """
    Logistic-map based nonlinear sequence.

    Used only to test nonlinear dependency.
    """

    x = np.zeros(n)


    x[0] = 0.2


    r = 3.8


    for i in range(1, n):

        x[i] = (

            r
            *
            x[i-1]
            *
            (1-x[i-1])

        )


    return x



# ---------------------------------------------------
# Tests
# ---------------------------------------------------

def test_temporal_output_structure():

    series = generate_ar1()


    evaluator = TemporalEvaluator()


    result = evaluator.evaluate(
        series
    )


    assert "score" in result

    assert "details" in result



def test_score_range():

    series = generate_ar1()


    evaluator = TemporalEvaluator()


    score = (

        evaluator.evaluate(series)

        ["score"]

    )


    assert 0 <= score <= 100



def test_temporal_structure_detected():

    dependent = generate_ar1()

    noise = generate_white_noise()


    evaluator = TemporalEvaluator()


    dependent_score = (

        evaluator.evaluate(dependent)

        ["score"]

    )


    noise_score = (

        evaluator.evaluate(noise)

        ["score"]

    )


    assert dependent_score > noise_score



def test_ljung_box_details_exist():

    series = generate_ar1()


    evaluator = TemporalEvaluator()


    details = (

        evaluator.evaluate(series)

        ["details"]

    )


    assert "ljung_box" in details

    assert "bds" in details

    assert "durbin_watson" in details



def test_nonlinear_dependency_execution():

    series = generate_nonlinear_series()


    evaluator = TemporalEvaluator()


    result = evaluator.evaluate(
        series
    )


    assert isinstance(
        result["score"],
        float
    )


    assert "bds" in result["details"]