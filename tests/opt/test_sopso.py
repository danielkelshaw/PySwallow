import numpy as np
import pytest

import pyswallow as ps
from pyswallow.utils.functions.single_objective import sphere


class TestSOSwarm:

    @pytest.fixture
    def optimiser(self):
        bounds = {
            'x0': [-50.0, -50.0],
            'x1': [50.0, 50.0]
        }

        opt = ps.Swarm(
            n_swallows=30,
            bounds=bounds,
            n_iterations=1000,
            debug=False
        )

        return opt

    @pytest.fixture
    def swallow(self):
        bounds = {
            'x0': [-50.0, -50.0],
            'x1': [50.0, 50.0]
        }

        swallow = ps.Swallow(bounds)
        return swallow

    def test_reset_environment(self, optimiser):
        optimiser.optimise(sphere)
        optimiser.reset_environment()

        assert optimiser.iteration == 0
        assert optimiser.population == []

    def test_initialise_swarm(self, optimiser):
        optimiser.population = []
        optimiser.initialise_swarm()

        assert len(optimiser.population) == optimiser.n_swallows
        for swallow in optimiser.population:
            assert isinstance(swallow, ps.Swallow)

    def test_evaluate_fitness(self, optimiser, swallow):
        target_fit = sphere(swallow.position)
        optimiser.evaluate_fitness(swallow, sphere)

        assert target_fit == swallow.fitness

    @pytest.mark.parametrize('f', [50, 0, -50])
    def test_pbest_update(self, optimiser, swallow, f):
        swallow.pbest_fitness = 100
        swallow.pbest_position = np.array([10, 10])
        swallow.fitness = f
        swallow.position = np.array([5, 5])
        optimiser.pbest_update(swallow)

        assert np.array_equal(swallow.pbest_fitness, swallow.fitness)
        assert np.array_equal(swallow.pbest_position, swallow.position)

    @pytest.mark.parametrize('f', [50, 0, -50])
    def test_gbest_update(self, optimiser, swallow, f):
        optimiser.gbest_fitness = 100
        optimiser.gbest_position = np.array([10, 10])
        swallow.fitness = f
        swallow.position = np.array([5, 5])
        optimiser.gbest_update(swallow)

        assert np.array_equal(optimiser.gbest_swallow.fitness, swallow.fitness)
        assert np.array_equal(optimiser.gbest_swallow.position, swallow.position)

    def test_optimise(self, optimiser):
        target_fit = 0.0
        target_pos = np.array([0, 0])

        optimiser.optimise(sphere)

        assert np.allclose(optimiser.gbest_swallow.fitness,
                           target_fit,
                           rtol=1e-3)
        assert np.allclose(optimiser.gbest_swallow.position,
                           target_pos,
                           rtol=1e-3)
