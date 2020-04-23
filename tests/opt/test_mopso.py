import pytest

import pyswallow as ps
from pyswallow.handlers.archive import Archive
from pyswallow.utils.functions.multi_objective import schaffer_n1


class TestMOSwarm:

    @pytest.fixture
    def optimiser(self):
        bounds = {
            'x0': [0.0, 50.0],
        }

        opt = ps.MOSwarm(
            n_swallows=30,
            bounds=bounds,
            n_iterations=100,
            debug=False
        )

        return opt

    @pytest.fixture
    def swallow(self):
        bounds = {
            'x0': [0.0, 50.0],
        }

        swallow = ps.MOSwallow(bounds, n_obj=2)
        return swallow

    def test_reset_environment(self, optimiser):
        optimiser.optimise(schaffer_n1())
        optimiser.reset_environment()

        assert optimiser.iteration == 0
        assert len(optimiser.population) == 0
        assert len(optimiser.archive.population) == 0

    def test_initialise_swarm(self, optimiser):
        optimiser.n_objs = 2
        optimiser.initialise_swarm()

        assert isinstance(optimiser.population, list)
        for swallow in optimiser.population:
            assert isinstance(swallow, ps.MOSwallow)

    def test_initialise_archive(self, optimiser):
        optimiser.initialise_archive()

        assert isinstance(optimiser.archive, Archive)
        assert optimiser.archive.population == []

    def test_evaluate_fitness(self, optimiser, swallow):
        swallow.position[0] = 0.0
        optimiser.evaluate_fitness(swallow, schaffer_n1())

        assert swallow.fitness[0] == 0
        assert swallow.fitness[1] == 4

    def test_update_pbest(self, optimiser, swallow):
        swallow.fitness = [5.0, 5.0]
        swallow.pbest_fitness = [10.0, 10.0]

        optimiser.update_pbest(swallow)
        assert swallow.pbest_fitness == [5.0, 5.0]
