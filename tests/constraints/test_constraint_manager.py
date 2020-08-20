import pytest

from pyswallow.opt.sopso import Swarm
from pyswallow.constraints.base_constraints import PositionConstraint
from pyswallow.constraints.constraint_manager import ConstraintManager


class TestConstraintManager:

    @pytest.fixture
    def swarm(self):
        swarm = Swarm(
            n_swallows=10,
            n_iterations=10,
            bounds={'x0': [-20.0, 20.0], 'x1': [-20.0, 20.0]}
        )

        swarm.initialise_swarm()

        return swarm

    @pytest.fixture
    def pos_constraint(self):
        class TestConstraint(PositionConstraint):

            def constrain(self, position):
                return position['x0'] > 0.0

        return TestConstraint

    def test_violates_position(self, swarm, pos_constraint):
        swallow = swarm.population[0]
        swallow.position[0] = -10.0

        constraints_manager = ConstraintManager(swarm)
        constraints_manager.register_constraint(pos_constraint())
        assert constraints_manager.violates_position(swallow)

    def test_register_constraint(self, swarm, pos_constraint):
        constraints_manager = ConstraintManager(swarm)
        constraints_manager.register_constraint(pos_constraint())

        assert len(constraints_manager.constraints) == 1
