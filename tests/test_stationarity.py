"""
Tests for StationarityEvaluator.
"""

import numpy as np


from predictability_score.stationarity import (
    StationarityEvaluator
)



# ---------------------------------------------------
# Helpers
# ---------------------------------------------------

def generate_stationary_series(
    n=2000,
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
            0.5 * x[i-1]
            +
            noise[i]
        )


    return x



def generate_random_walk(
    n=2000,
    seed=42,
):
    """
    Generate non-stationary random walk.
    """

    rng = np.random.RandomState(seed)


    steps = rng.normal(
        0,
        1,
        n
    )


    return np.cumsum(steps)



# ---------------------------------------------------
# Tests
# ---------------------------------------------------

def test_stationarity_output_structure():

    series = generate_stationary_series()


    evaluator = StationarityEvaluator()


    result = evaluator.evaluate(
        series
    )


    assert "score" in result

    assert "details" in result


    assert isinstance(
        result["score"],
        float
    )



def test_score_range():

    series = generate_stationary_series()


    evaluator = StationarityEvaluator()


    result = evaluator.evaluate(
        series
    )


    score = result["score"]


    assert 0 <= score <= 100



def test_stationary_series_has_high_score():

    series = generate_stationary_series()


    evaluator = StationarityEvaluator()


    result = evaluator.evaluate(
        series
    )


    score = result["score"]


    assert score > 50



def test_random_walk_has_lower_score():

    stationary = generate_stationary_series()

    random_walk = generate_random_walk()


    evaluator = StationarityEvaluator()


    s1 = evaluator.evaluate(
        stationary
    )["score"]


    s2 = evaluator.evaluate(
        random_walk
    )["score"]


    assert s1 > s2



def test_stationarity_details_exist():

    series = generate_stationary_series()


    evaluator = StationarityEvaluator()


    result = evaluator.evaluate(
        series
    )


    details = result["details"]


    assert "adf" in details

    assert "kpss" in details


    assert "p_value" in details["adf"]

    assert "p_value" in details["kpss"]