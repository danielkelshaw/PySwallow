import pytest
import numpy as np

import pyswallow as ps
import pyswallow.handlers.boundary_handler as psbh
from pyswallow.utils.functions.single_objective import sphere


class TestSOSwallow:

    @pytest.fixture
    def swallow(self):
        lb = np.array([-50, 50])
        ub = np.array([50, 50])
        swallow = ps.Swallow(lb, ub)
        return swallow

    def test_move(self, swallow):
        swallow.position = np.array([0, 0])
        swallow.velocity = np.array([5, 5])
        swallow.move(psbh.StandardBH())

        assert np.array_equal(swallow.position, np.array([5, 5]))


class TestSOSwarm:

    @pytest.fixture
    def optimiser(self):
        opt = ps.Swarm(obj_function=sphere,
                       n_swallows=30,
                       n_iterations=1000,
                       lb=[-50, 50],
                       ub=[50, 50],
                       constraints=None,
                       debug=False)
        return opt

    @pytest.fixture
    def constrained_optimiser(self, optimiser):

        def applied_constraints(position):
            if np.logical_or((position[0] >= 0) and (position[1] <= 0),
                             (position[1] >= 0) and (position[0] <= 0)):
                return True
            else:
                return False

        optimiser.constraints = applied_constraints
        return optimiser

    @pytest.fixture
    def swallow(self):
        lb = np.array([-50, 50])
        ub = np.array([50, 50])
        swallow = ps.Swallow(lb, ub)
        return swallow

    @pytest.fixture
    def reset_optimiser(self, optimiser):
        optimiser.optimise()
        optimiser.reset_environment()
        return optimiser

    def test_check_reset(self, reset_optimiser):
        assert reset_optimiser.iteration == 0
        assert reset_optimiser.population == []

    def test_initialise_swarm(self, optimiser):
        optimiser.population = []
        optimiser.initialise_swarm()

        assert len(optimiser.population) == optimiser.n_swallows
        for swallow in optimiser.population:
            assert isinstance(swallow, ps.Swallow)

    def test_eval_fitness(self, optimiser, swallow):
        target_fit = sphere(swallow.position)
        optimiser.evaluate_fitness(swallow)
        assert target_fit == swallow.fitness

    @pytest.mark.parametrize('f', [50, 0, -50])
    def test_unconstrained_pbest_update(self, optimiser, swallow, f):
        swallow.pbest_fitness = 100
        swallow.pbest_position = np.array([10, 10])
        swallow.fitness = f
        swallow.position = np.array([5, 5])
        optimiser.pbest_update(swallow)

        assert np.array_equal(swallow.pbest_fitness, swallow.fitness)
        assert np.array_equal(swallow.pbest_position, swallow.position)

    @pytest.mark.parametrize('f', [50, 0, -50])
    def test_unconstrained_gbest_update(self, optimiser, swallow, f):
        optimiser.gbest_fitness = 100
        optimiser.gbest_position = np.array([10, 10])
        swallow.fitness = f
        swallow.position = np.array([5, 5])
        optimiser.gbest_update(swallow)

        assert np.array_equal(optimiser.gbest_swallow.fitness, swallow.fitness)
        assert np.array_equal(optimiser.gbest_swallow.position, swallow.position)

    def test_sphere_opt_result(self, optimiser):
        target_fit = 0.0
        target_pos = np.array([0, 0])

        optimiser.optimise()

        assert np.allclose(optimiser.gbest_swallow.fitness,
                           target_fit,
                           rtol=1e-3)
        assert np.allclose(optimiser.gbest_swallow.position,
                           target_pos,
                           rtol=1e-3)
