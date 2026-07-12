"""
Tests for StabilityEvaluator.
"""


import numpy as np


from predictability_score.temporal import (
    TemporalEvaluator
)


from predictability_score.stability import (
    StabilityEvaluator
)



# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def generate_ar1(
    n=5000,
    phi=0.8,
    seed=42,
):
    """
    Generate stationary AR(1).
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



def generate_regime_change(
    n=5000,
    seed=42,
):
    """
    Generate series with changing dynamics.

    First half:
        AR(1)

    Second half:
        White noise
    """

    rng = np.random.RandomState(seed)


    half = n // 2


    first = generate_ar1(
        n=half,
        phi=0.9,
        seed=seed
    )


    second = rng.normal(

        0,

        1,

        half

    )


    return np.concatenate(

        [
            first,
            second
        ]

    )



# ---------------------------------------------------
# Tests
# ---------------------------------------------------

def test_stability_output_structure():

    series = generate_ar1()


    evaluator = StabilityEvaluator(

        temporal_evaluator=
        TemporalEvaluator(),

        windows=8

    )


    result = evaluator.evaluate(
        series
    )


    assert "score" in result

    assert "details" in result



def test_score_range():

    series = generate_ar1()


    evaluator = StabilityEvaluator()


    score = (

        evaluator.evaluate(series)

        ["score"]

    )


    assert 0 <= score <= 100



def test_stable_series_has_high_score():

    stable = generate_ar1()


    evaluator = StabilityEvaluator(

        windows=8

    )


    score = (

        evaluator.evaluate(stable)

        ["score"]

    )


    assert score > 40



def test_regime_change_reduces_stability():

    stable = generate_ar1()

    unstable = generate_regime_change()


    evaluator = StabilityEvaluator(

        windows=8

    )


    stable_score = (

        evaluator.evaluate(stable)

        ["score"]

    )


    unstable_score = (

        evaluator.evaluate(unstable)

        ["score"]

    )


    assert stable_score > unstable_score



def test_window_scores_exist():

    series = generate_ar1()


    evaluator = StabilityEvaluator()


    details = (

        evaluator.evaluate(series)

        ["details"]

    )


    assert "window_scores" in details

    assert "mean" in details

    assert "std" in details

    assert "coefficient_of_variation" in details
#Tests for TemporalEvaluator.


import numpy as np


from predictability_score.temporal import (
    TemporalEvaluator
)



# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def generate_ar1(
    n=5000,
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
    n=5000,
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
    n=5000,
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