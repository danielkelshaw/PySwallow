import numpy as np
import pytest

from pyswallow.handlers.velocity_handler import (
    StandardVH, ClampedVH, InvertVH, ZeroVH
)


class TestVelocityHandler:

    @pytest.fixture
    def bounds(self):
        lb = np.array([0, 0])
        ub = np.array([10, 10])
        return lb, ub

    @pytest.mark.parametrize('vel', [[-5, -5], [5, 5], [15, 15]])
    def test_standard_vh(self, vel):
        vel = np.array(vel, dtype=np.float32)
        vh = StandardVH()
        ret_vel = vh(vel)

        assert np.array_equal(ret_vel, vel)

    @pytest.mark.parametrize('vel', [[-5, -5], [5, 5], [15, 15]])
    def test_clamped_vh(self, bounds, vel):
        lb, ub = bounds
        arr_vel = np.array(vel, dtype=np.float32)
        vh = ClampedVH(lb, ub)
        ret_vel = vh(arr_vel)

        assert np.logical_and(ret_vel >= lb, ret_vel <= ub).all()
        if vel == [-5, -5]:
            assert np.array_equal(ret_vel, lb)
        elif vel == [5, 5]:
            assert np.array_equal(ret_vel, arr_vel)
        elif vel == [15, 15]:
            assert np.array_equal(ret_vel, ub)

    @pytest.mark.parametrize('vel', [[-5, -5], [5, 5], [15, 15]])
    def test_invert_vh(self, bounds, vel):
        lb, ub = bounds
        arr_vel = np.array(vel, dtype=np.float32)
        vh = InvertVH(lb, ub)
        ret_vel = vh(arr_vel, 0.5)

        if vel == [-5, -5]:
            assert np.array_equal(ret_vel, np.array([2.5, 2.5]))
        elif vel == [5, 5]:
            assert np.array_equal(ret_vel, arr_vel)
        elif vel == [15, 15]:
            assert np.array_equal(ret_vel, np.array([-7.5, -7.5]))

    @pytest.mark.parametrize('vel', [[-5, -5], [5, 5], [15, 15]])
    def test_zero_vh(self, bounds, vel):
        lb, ub = bounds
        arr_vel = np.array(vel, dtype=np.float32)
        vh = ZeroVH(lb, ub)
        ret_vel = vh(arr_vel)

        if vel == [-5, -5]:
            assert np.array_equal(ret_vel, np.zeros(2))
        elif vel == [5, 5]:
            assert np.array_equal(ret_vel, arr_vel)
        elif vel == [15, 15]:
            assert np.array_equal(ret_vel, np.zeros(2))
