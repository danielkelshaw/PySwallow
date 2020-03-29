import pytest
import numpy as np

import pyswallow as ps
import pyswallow.handlers.boundary_handler as psbh
from pyswallow.handlers.archive import Archive
from pyswallow.utils.functions.multi_objective import schaffer_n1


class TestMOSwallow:

    @pytest.fixture
    def swallow(self):
        lb = np.array([-50, 50])
        ub = np.array([50, 50])
        swallow = ps.MOSwallow(lb, ub, n_obj=2)
        return swallow

    @pytest.mark.parametrize('vel', [np.array([5, 5]), np.array([-5, -5])])
    def test_move(self, swallow, vel):
        swallow.position = np.array([0, 0])
        swallow.velocity = vel
        swallow.move(psbh.StandardBH())

        assert np.array_equal(swallow.position, vel)

    @pytest.mark.parametrize('opp_fit', [[5, 10], [10, 5], [10, 10]])
    def test_dominate(self, opp_fit):
        swallow = ps.MOSwallow(np.array([0, 0]), np.array([1, 1]), 2)
        opp_swallow = ps.MOSwallow(np.array([0, 0]), np.array([1, 1]), 2)

        swallow.fitness = [5, 5]
        opp_swallow.fitness = opp_fit

        assert swallow.dominate(opp_swallow)

    @pytest.mark.parametrize('fit', [[5, 10], [10, 5], [-5, -5]])
    def test_self_dominate(self, swallow, fit):
        swallow.fitness = fit
        swallow.pbest_fitness = [10, 10]

        assert swallow.self_dominate()


class TestMOSwarm:

    @pytest.fixture
    def optimiser(self):
        opt = ps.MOSwarm(obj_functions=schaffer_n1(),
                         n_swallows=30,
                         n_iterations=100,
                         lb=[-50],
                         ub=[50])
        return opt

    @pytest.fixture
    def swallow(self):
        swallow = ps.MOSwallow(np.array([-50]), np.array([50]), 2)
        return swallow

    def test_initialise_swarm(self, optimiser):
        optimiser.population = []
        optimiser.initialise_swarm()

        assert len(optimiser.population) == optimiser.n_swallows
        for swallow in optimiser.population:
            assert isinstance(swallow, ps.MOSwallow)

    def test_initialise_archive(self, optimiser):
        optimiser.archive = None
        optimiser.initialise_archive()
        assert isinstance(optimiser.archive, Archive)

    def test_evaluate_fitness(self, optimiser, swallow):
        swallow.position = np.array([0])
        target_fitness = np.array([0, 4])
        optimiser.evaluate_fitness(swallow)
        print(swallow.fitness)

        assert np.array_equal(swallow.fitness, target_fitness)

    def test_update_pbest(self, optimiser, swallow):
        swallow.position = np.array([10, 10])
        swallow.fitness = np.array([50, 50])
        swallow.pbest_position = np.array([5, 5])
        swallow.pbest_fitness = np.array([100, 100])
        optimiser.update_pbest(swallow)

        assert np.array_equal(swallow.position, swallow.pbest_position)
        assert np.array_equal(swallow.fitness, swallow.pbest_fitness)

    @pytest.mark.parametrize('iteration', [100, 200])
    def test_termination_check(self, optimiser, iteration):
        optimiser.n_iterations = 100
        optimiser.iteration = 200
        terminate = optimiser.termination_check()

        assert not terminate

    def test_optimise(self, optimiser):
        assert len(optimiser.archive.population) == 0
        optimiser.optimise()
        assert len(optimiser.archive.population) == optimiser.n_swallows
