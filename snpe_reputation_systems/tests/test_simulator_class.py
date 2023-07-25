from typing import Deque, List, Optional, Union

import hypothesis
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy import float

import numpy as np

import pandas as pd
import pytest


from ..snpe_reputation_systems.simulations.simulator_class import (
    BaseSimulator,
    DoubleRhoSimulator,
    HerdingSimulator,
    SingleRhoSimulator,
)

# class TestBaseSimulator
#############################################


# @pytest.fixture
# def yield_BaseSimulator():
#    params = {
#        "review_prior": np.ones(5),
#        "tendency_to_rate": 0.05,
#        "simulation_type": "timeseries",
#    }
#    return BaseSimulator(params)


@given(arrays(float, 5), arrays(float, 6), st.none())
def test_convolve_prior_with_existing_reviews(arr1, arr2, none_value):
    # BaseSimulator instance
    params = {
        "review_prior": np.ones(5),
        "tendency_to_rate": 0.05,
        "simulation_type": "timeseries",
    }
    base_simulator = BaseSimulator(params)

    # Test of correct sum
    result = base_simulator.convolve_prior_with_existing_reviews(arr1)
    assert np.array_equal(result, np.ones(5) + arr1)

    # Input shape test
    with pytest.raises(ValueError):
        base_simulator.convolve_prior_with_existing_reviews(arr2)

    # Output type test
    assert isinstance(result, np.ndarray)

    # Null input test
    with pytest.raises(AttributeError):
        base_simulator.convolve_prior_with_existing_reviews(none_value)


def test_simulate():
    pass


def test_yield_simulation_param_per_visitor():
    pass


@pytest.fixture
def generate_mock_simulation_parameters(test_base_simulator):
    pass


def test_save_simulations():
    pass


def test_load_simulations():
    pass


# class TestSingleRhoSimulator:
#############################################


def yield_SingleRhoSimulator():
    params = {
        "review_prior": np.array([1, 1, 1, 1, 1]),
        "tendency_to_rate": 1.0,
        "simulation_type": "histogram",
    }
    return SingleRhoSimulator(params)


@settings(max_examples=20)
@given(
    experience=st.integers(min_value=1, max_value=5),
    expected_experience=st.integers(min_value=1, max_value=5),
)
def test_mismatch_calculator(experience, expected_experience):
    simulator = yield_SingleRhoSimulator()

    # Test of correct substraction
    assert simulator.mismatch_calculator(experience, expected_experience) == (
        experience - expected_experience
    )

    # Output type test
    assert isinstance(
        simulator.mismatch_calculator(experience, expected_experience), int
    )

    # Null input test
    with pytest.raises(AssertionError):
        simulator.mismatch_calculator(None, None)


@settings(max_examples=20)
@given(
    delta=st.floats(min_value=-4, max_value=4), simulation_id=st.integers(min_value=0)
)
def test_rating_calculator(delta, simulation_id):
    simulator = yield_SingleRhoSimulator()
    result = simulator.rating_calculator(delta, simulation_id)

    # Return type test
    assert isinstance(result, int), "Result is not an integer"

    # Result range test
    assert 0 <= result <= 4, "Result is out of expected range"

    # Test of correct output
    if delta <= -1.5:
        assert result == 0
    elif delta > -1.5 and delta <= -0.5:
        assert result == 1
    elif delta > -0.5 and delta <= 0.5:
        assert result == 2
    elif delta > 0.5 and delta <= 1.5:
        assert result == 3
    else:
        assert result == 4


# class TestDoubleRhoSimulator:
#############################################


def test_double_rho_simulator_generate_simulation_parameters():
    num_simulations = 10
    params = DoubleRhoSimulator.generate_simulation_parameters(num_simulations)

    assert isinstance(params, dict), "Result is not a dictionary"
    assert "rho" in params, "Result does not contain rho"
    assert params["rho"].shape == (
        10,
        num_simulations,
        2,
    ), "Result does not have correct shape"


# class TestHerdingSimulator:
#############################################


def test_herding_simulator_generate_simulation_parameters():
    num_simulations = 10
    params = HerdingSimulator.generate_simulation_parameters(num_simulations)

    assert isinstance(params, dict)
    assert "rho" in params
    assert "h_p" in params
    assert params["rho"].shape == (10, num_simulations, 2)
    assert params["h_p"].shape == (10, num_simulations)
