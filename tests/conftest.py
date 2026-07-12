"""
Shared pytest fixtures.

Common synthetic time series generators
used across test modules.
"""


import numpy as np

import pytest



# ---------------------------------------------------
# Random seed fixture
# ---------------------------------------------------

@pytest.fixture
def random_seed():

    return 42



# ---------------------------------------------------
# Stationary AR(1)
# ---------------------------------------------------

@pytest.fixture
def stationary_series():

    rng = np.random.RandomState(
        42
    )


    n = 3000


    x = np.zeros(n)


    noise = rng.normal(
        0,
        1,
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
# Strong dependency AR(1)
# ---------------------------------------------------

@pytest.fixture
def predictable_series():

    rng = np.random.RandomState(
        42
    )


    n = 3000


    x = np.zeros(n)


    noise = rng.normal(
        0,
        0.1,
        n
    )


    for i in range(1, n):

        x[i] = (

            0.9 * x[i-1]

            +

            noise[i]

        )


    return x



# ---------------------------------------------------
# White noise
# ---------------------------------------------------

@pytest.fixture
def white_noise_series():

    rng = np.random.RandomState(
        42
    )


    return rng.normal(

        0,

        1,

        3000

    )



# ---------------------------------------------------
# Random walk
# ---------------------------------------------------

@pytest.fixture
def random_walk_series():

    rng = np.random.RandomState(
        42
    )


    steps = rng.normal(

        0,

        1,

        3000

    )


    return np.cumsum(
        steps
    )



# ---------------------------------------------------
# Regime change series
# ---------------------------------------------------

@pytest.fixture
def regime_change_series():

    rng = np.random.RandomState(
        42
    )


    n = 3000


    half = n // 2


    first = np.zeros(
        half
    )


    noise = rng.normal(
        0,
        1,
        half
    )


    for i in range(1, half):

        first[i] = (

            0.9 * first[i-1]

            +

            noise[i]

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
# Short series for fast tests
# ---------------------------------------------------

@pytest.fixture
def short_series():

    rng = np.random.RandomState(
        42
    )


    return rng.normal(

        0,

        1,

        500

    )