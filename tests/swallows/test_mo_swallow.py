import numpy as np
import pytest

import pyswallow as ps
import pyswallow.handlers.boundary_handler as psbh


class TestMOSwallow:

    @pytest.fixture
    def swallow(self):
        bounds = {
            'x0': [-50.0, 50.0],
            'x1': [-50.0, 50.0]
        }

        swallow = ps.MOSwallow(bounds, n_obj=2)
        return swallow

    @pytest.fixture
    def opp_swallow(self):
        bounds = {
            'x0': [-50.0, 50.0],
            'x1': [-50.0, 50.0]
        }

        opp_swallow = ps.MOSwallow(bounds, n_obj=2)
        return opp_swallow

    def test_move(self, swallow):
        swallow.position = np.array([0.0, 0.0])
        swallow.velocity = np.array([10.0, 10.0])

        bh = psbh.StandardBH()
        swallow.move(bh)

        assert np.array_equal(swallow.position, swallow.velocity)

    def test_dominate(self, swallow, opp_swallow):
        opp_swallow.fitness = [50.0, 50.0]
        swallow.fitness = [5.0, 5.0]

        ret_bool = swallow.dominate(opp_swallow)

        assert ret_bool

    def test_self_dominate(self, swallow):
        swallow.fitness = [5.0, 5.0]
        swallow.pbest_fitness = [50.0, 50.0]

        ret_bool = swallow.self_dominate()

        assert ret_bool
