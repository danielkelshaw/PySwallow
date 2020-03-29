import pytest
import numpy as np

import pyswallow as ps
from pyswallow.utils.functions.single_objective import sphere


@pytest.fixture
def basic_optimiser():

    """Returns basic Swarm object."""

    opt = ps.Swarm(obj_function=sphere,
                   n_swallows=30,
                   n_iterations=1000,
                   lb=[-50, -50],
                   ub=[50, 50],
                   constraints=None,
                   debug=False)
    return opt


def test_sphere_optimizer(basic_optimiser):
    basic_optimiser.optimise()

    target_fitness = 0.0
    target_position = np.array([0.0, 0.0])

    assert np.isclose(basic_optimiser.gbest_fitness,
                      target_fitness,
                      rtol=1e-3)

    assert np.allclose(basic_optimiser.gbest_position,
                       target_position,
                       rtol=1e-3)
