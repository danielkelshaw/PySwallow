import numpy as np
import pytest

import pyswallow.utils.functions.single_objective as fx


class TestSingleObjective:

    def test_ackley(self):
        pos = np.array([0.0, 0.0, 0.0])
        assert fx.ackley(pos) == pytest.approx(0.0, 1e-6)

    def test_beale(self):
        pos = np.array([3.0, 0.5])
        assert fx.beale(pos) == pytest.approx(0.0, 1e-6)

    def test_booth(self):
        pos = np.array([1.0, 3.0])
        assert fx.booth(pos) == pytest.approx(0.0, 1e-6)

    def test_goldsteinprice(self):
        pos = np.array([0.0, -1.0])
        assert fx.goldsteinprice(pos) == pytest.approx(3.0, 1e-6)

    def test_rastrigin(self):
        pos = np.array([0.0, 0.0, 0.0])
        assert fx.rastrigin(pos) == pytest.approx(0.0, 1e-6)

    def test_sphere(self):
        pos = np.array([0.0, 0.0, 0.0])
        assert fx.sphere(pos) == pytest.approx(0.0, 1e-6)
