import pytest

import pyswallow as ps
from pyswallow.utils.history import *


class TestGeneralHistory:

    @pytest.fixture
    def optimiser(self):
        bounds = {
            'x0': [0.0, 10.0],
            'x1': [0.0, 10.0]

        }
        optimiser = ps.Swarm(n_swallows=30, n_iterations=100, bounds=bounds)
        optimiser.initialise_swarm()

        optimiser.gbest_swallow = ps.Swallow(bounds)
        optimiser.gbest_swallow.fitness = 0.5

        for idx, swallow in enumerate(optimiser.population):
            swallow.fitness = 5.0

        return optimiser

    def test_write_history(self, optimiser):
        hist = SOHistory(optimiser)
        hist.write_history()

        assert isinstance(hist.arr_best_fitness, list)
        assert isinstance(hist.arr_mean_fitness, list)

        assert len(hist.arr_best_fitness) == len(hist.arr_mean_fitness) == 1
        assert hist.arr_best_fitness[0] == 0.5
        assert hist.arr_mean_fitness[0] == 5.0
