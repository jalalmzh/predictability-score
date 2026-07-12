"""
Tests for NullSignificanceEvaluator.
"""


import numpy as np


from predictability_score.temporal import (
    TemporalEvaluator
)


from predictability_score.null_significance import (
    NullSignificanceEvaluator
)



# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def generate_ar1(
    n=1500,
    phi=0.8,
    seed=42,
):
    """
    Generate stationary AR(1) process.
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
    n=1500,
    seed=42,
):
    """
    Generate IID noise.
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

def test_null_output_structure():

    series = generate_ar1()


    evaluator = NullSignificanceEvaluator(

        temporal_evaluator=
        TemporalEvaluator(),

        iterations=20

    )


    result = evaluator.evaluate(
        series
    )


    assert "score" in result

    assert "details" in result



def test_score_range():

    series = generate_ar1()


    evaluator = NullSignificanceEvaluator(

        iterations=20

    )


    score = (

        evaluator.evaluate(series)

        ["score"]

    )


    assert 0 <= score <= 100



def test_dependent_series_has_high_significance():

    series = generate_ar1()


    evaluator = NullSignificanceEvaluator(

        iterations=50,

        random_state=123

    )


    result = evaluator.evaluate(
        series
    )


    score = result["score"]


    assert score > 80



def test_white_noise_has_lower_significance():

    dependent = generate_ar1()

    noise = generate_white_noise()


    evaluator = NullSignificanceEvaluator(

        iterations=30,

        random_state=42

    )


    dependent_score = (

        evaluator.evaluate(dependent)

        ["score"]

    )


    noise_score = (

        evaluator.evaluate(noise)

        ["score"]

    )


    assert dependent_score > noise_score



def test_details_content():

    series = generate_ar1()


    evaluator = NullSignificanceEvaluator(

        iterations=10

    )


    details = (

        evaluator.evaluate(series)

        ["details"]

    )


    assert "real_temporal_score" in details

    assert "null_mean" in details

    assert "null_std" in details

    assert "iterations" in details

    assert "percentile" in details