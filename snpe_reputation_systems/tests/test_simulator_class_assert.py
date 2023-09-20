from typing import Deque, List, Optional, Union

import hypothesis
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, composite, given, settings
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
    def test___init__(self, array_int5, array_not5, random_string):
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
        self, array_not5, array_int5, empty_arr
    ):
        """
        Testing "convolve_prior_with_existing_reviews"
        according to the former "assert"cases provided for this
        BaseSimulator method in simulator_class.py
        """

        # Hypothesis rule so array_not5 cannot take the "correct" shape (5,)
        assume(array_not5.shape != (5,))

        # Instantiate base simulator
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

    def gen_random_existing_reviews(self, num_products: int, depth: int):
        """
        Assistant function for "test_simulate" method, generates a random array
        which is valid to be passed as the "existing_reviews" parameter. The number
        of reviews is fixed by the parameter "depth" while the number of products is
        be adjusted through the parameter "num_products". It returns an array of shape
        (num_products, depth, 5) where the first row of each product is [1, 1, 1, 1, 1]
        """

        # Initialize array with shape (num_products, time, 5)
        existing_reviews = np.zeros((num_products, depth, 5), dtype=int)

        # Fill array
        for i in range(num_products):
            # First row of each product
            existing_reviews[i, 0] = np.array([1, 1, 1, 1, 1])

            # Adding the subsequent lines with reviews being added randomly
            for j in range(1, depth):
                add_index = np.random.choice(5)
                existing_reviews[i, j] = existing_reviews[i, j - 1] + np.array(
                    [1 if k == add_index else 0 for k in range(5)]
                )

        return existing_reviews

    @composite
    def _integer_and_array(self, draw):
        """
        Function for composite hypothesis strategy.

        This is required as in the "simulate" method, num_reviews_per_simulation
        is expected to have a length equal to num_simulations.

        Accordingly, the function return the value for num_simulations and an appropriate
        num_reviews_per_simulation array
        """
        n = draw(integers(min_value=1, max_value=50))
        array = draw(arrays(int, n, elements=integers(min_value=0, max_value=50)))
        return n, array  # num_simulations, num_reviews_per_simulation

    @settings(max_examples=10)
    @given(
        _integer_and_array(),
        arrays(int, 5, elements=integers(min_value=0, max_value=5)),
        st.lists(
            single_array_strategy=arrays(
                dtype=np.int32, shape=st.integers(min_value=1, max_value=5)
            ),
            min_size=1,
            max_size=10,
        ),
    )
    def test_simulate(self, int_and_array, array):
        """
        Testing "simulate" method according to the former "assert"cases provided for this
        BaseSimulator method in simulator_class.py
        """

        num_simulations, num_reviews_per_simulation = int_and_array

        # Instantiate base simulator
        base_simulator = TestBaseSimulator.get_base_simulator()

        # If existing_reviews exists:

        # Expect ValueError if simulation_parameters is None
        with pytest.raises(ValueError):
            base_simulator.simulate(
                existing_reviews=self.gen_random_existing_reviews(num_simulations, 10)
            )

        # Expect ValueError if num_reviews_per_simulation is None
        with pytest.raises(ValueError):
            base_simulator.simulate(
                existing_reviews=self.gen_random_existing_reviews(num_simulations, 10),
                simulation_parameters={},
            )

        # If all three exist: code continues

        # If num_reviews_per_simulation exists:

        # Expect ValueError if len(num_reviews_per_simulation) != num_simulations
        # base_simulator.simulate(num_simulations, num_reviews_per_simulation)

        # If simulation_parameters exists:

        # Expect KeyError if set(simulation_parameters) != set(dummy_parameters)
