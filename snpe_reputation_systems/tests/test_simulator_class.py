from typing import Deque, List, Optional, Union

import hypothesis
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import none, text, floats
from numpy import float64

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


@given(
    arrays(float64, 5, elements=floats(min_value=-100, max_value=100)),
    arrays(float64, 6, elements=floats(min_value=-100, max_value=100)),
    arrays(float64, 5, elements=floats(allow_nan=True, allow_infinity=True)),
    arrays(float64, 0),
    none(),
    text(),
)
def test_convolve_prior_with_existing_reviews(
    arr1, arr2, nan_or_inf_arr, empty_arr, none_value, string_value
):
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

    # NaN or Inf input test
    with pytest.raises(ValueError):
        base_simulator.convolve_prior_with_existing_reviews(nan_or_inf_arr)

    # Empty array input test
    with pytest.raises(ValueError):
        base_simulator.convolve_prior_with_existing_reviews(empty_arr)

    # Output type test
    assert isinstance(result, np.ndarray)

    # Null input test
    with pytest.raises(AttributeError):
        base_simulator.convolve_prior_with_existing_reviews(none_value)

    # String input test
    with pytest.raises(TypeError):
        base_simulator.convolve_prior_with_existing_reviews(string_value)



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


@settings(max_examples=10)
@given(
    experience=st.integers(min_value=1, max_value=5),
    expected_experience=st.floats(min_value=1, max_value=5),
    wrong_experience=st.integers(min_value=6, max_value=10),
    wrong_expected_experience=st.floats(min_value=6, max_value=10),
)
def test_mismatch_calculator(
    experience, expected_experience, wrong_experience, wrong_expected_experience
):
    simulator = yield_SingleRhoSimulator()

    # Test of correct substraction
    assert simulator.mismatch_calculator(experience, expected_experience) == (
        experience - expected_experience
    )

    # Output type test
    assert isinstance(
        simulator.mismatch_calculator(experience, expected_experience), float
    )

    # out-of-range experience test
    with pytest.raises(ValueError):
        simulator.mismatch_calculator(wrong_experience, expected_experience)

    # out-of-range expected experience test
    with pytest.raises(ValueError):
        simulator.mismatch_calculator(experience, wrong_expected_experience)


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
