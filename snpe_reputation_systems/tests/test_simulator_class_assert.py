from typing import Deque, List, Optional, Union

import hypothesis
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, none, text, tuples
from numpy import float64

from ..snpe_reputation_systems.simulations.simulator_class import (
    BaseSimulator,
    DoubleRhoSimulator,
    HerdingSimulator,
    SingleRhoSimulator,
)

# class TestBaseSimulator
#############################################


class TestBaseSimulator:
    def get_base_simulator(
        review_prior=np.array([1, 1, 1, 1, 1]),
        tendency_to_rate=0.05,
        simulation_type="timeseries",
    ):
        """
        Returns a functional instance of BaseSimulator to use for the different
        tests for its methods.

        Although in most cases the default values set for `params` would work,
        the option to manually modify these has been considered in case such
        flexibility is necesary later in the testing design and implementation
        process.
        """

        params = {
            "review_prior": review_prior,
            "tendency_to_rate": tendency_to_rate,
            "simulation_type": simulation_type,
        }

        return BaseSimulator(params)

    @settings(max_examples=10)
    @given(
        arrays(int, 5, elements=integers(min_value=0, max_value=100)),
        arrays(
            int,
            shape=tuples(integers(1, 10)),
            elements=integers(min_value=0, max_value=100),
        ),
        text(min_size=3, max_size=15),
    )
    def test___init__(array_int5, array_not5, random_string):
        """
        Testing builder method by providing it with innapropriate paramerters
        according to the former "assert"cases provided for BaseSimulator
        in simulator_class.py
        """

        # Hypothesis rule so array_not5 cannot take the "correct" shape (5,)
        assume(array_not5.shape != (5,))

        # Testing correct cases

        assert isinstance(TestBaseSimulator.get_base_simulator(), BaseSimulator)

        assert isinstance(
            TestBaseSimulator.get_base_simulator(review_prior=array_int5), BaseSimulator
        )

        assert isinstance(
            TestBaseSimulator.get_base_simulator(simulation_type="histogram"),
            BaseSimulator,
        )

        # Testing incorrect shape of "review_prior"

        with pytest.raises(
            ValueError,
            match="Prior Dirichlet distribution of simulated reviews needs to have 5 parameters",
        ):
            TestBaseSimulator.get_base_simulator(review_prior=array_not5)

        # Testing incorrect values for "simulation type"

        with pytest.raises(
            ValueError, match="Can only simulate review histogram or timeseries"
        ):
            TestBaseSimulator.get_base_simulator(simulation_type=random_string)

    @settings(max_examples=10)
    @given(
        arrays(
            dtype=int,
            shape=tuples(integers(1, 10)),
            elements=integers(min_value=0, max_value=100),
        ),
        arrays(dtype=int, shape=5, elements=integers(min_value=0, max_value=100)),
        arrays(int, 0),
    )
    def test_convolve_prior_with_existing_reviews(
        array_not5, array_int5, empty_arr, none_value
    ):
        """
        Testing "convolve_prior_with_existing_reviews"
        according to the former "assert"cases provided for this
        BaseSimulator method in simulator_class.py
        """

        # Hypothesis rule so array_not5 cannot take the "correct" shape (5,)
        assume(array_not5.shape != (5,))

        # Instanciate base simulator
        base_simulator = TestBaseSimulator.get_base_simulator()

        # Testing correct cases
        result = base_simulator.convolve_prior_with_existing_reviews(array_int5)

        assert np.array_equal(result, np.ones(5) + array_int5)

        assert isinstance(result, np.ndarray)

        # Testing incorrect cases (1)
        with pytest.raises(ValueError):
            base_simulator.convolve_prior_with_existing_reviews(array_not5)

        # Testing  incorrect cases (2)
        with pytest.raises(ValueError):
            base_simulator.convolve_prior_with_existing_reviews(empty_arr)
