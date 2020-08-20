import time

import pytest

import pyswallow as ps
from pyswallow.utils.termination_manager import (
    IterationTerminationManager,
    TimeTerminationManager,
    EvaluationTerminationManager,
    ErrorTerminationManager
)


@pytest.fixture
def optimiser():
    bounds = {
        'x0': [0.0, 10.0],
        'x1': [0.0, 10.0]
    }

    optimiser = ps.Swarm(n_swallows=10, n_iterations=100, bounds=bounds)
    return optimiser


class TestIterationTerminationManager:

    def test_termination_check(self, optimiser):
        optimiser.iteration = 150
        tm = IterationTerminationManager(optimiser)
        ret_bool = tm.termination_check()

        assert ret_bool


class TestTimeTerminationManager:

    def test_termination_check(self):
        tm = TimeTerminationManager(t_budget=1)
        tm.t_start = time.time()
        time.sleep(2)
        ret_bool = tm.termination_check()

        assert ret_bool


class TestEvaluationTerminationManager:

    def test_termination_check(self, optimiser):
        optimiser.iteration = 11
        tm = EvaluationTerminationManager(optimiser, n_evaluations=100)
        ret_bool = tm.termination_check()

        assert ret_bool


class TestErrorTerminationManager:

    def test_termination_check(self, optimiser):
        bounds = {
            'x0': [0.0, 10.0],
            'x1': [0.0, 10.0]
        }

        best = ps.Swallow(bounds)
        best.fitness = 0.0001
        optimiser.gbest_swallow = best

        tm = ErrorTerminationManager(optimiser, 0.0, 1e-3)
        ret_bool = tm.termination_check()

        assert ret_bool
