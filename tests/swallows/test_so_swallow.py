import numpy as np
import pytest

import pyswallow as ps
import pyswallow.handlers.boundary_handler as psbh


class TestSOSwallow:

    @pytest.fixture
    def swallow(self):

        bounds = {
            'x0': [-50.0, -50.0],
            'x01': [50.0, 50.0]
        }

        swallow = ps.Swallow(bounds)
        return swallow

    def test_move(self, swallow):
        swallow.position = np.array([0, 0])
        swallow.velocity = np.array([5, 5])
        swallow.move(psbh.StandardBH())

        assert np.array_equal(swallow.position, np.array([5, 5]))
