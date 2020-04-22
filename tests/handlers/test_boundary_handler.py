import numpy as np
import pytest

from pyswallow.handlers.boundary_handler import (
    StandardBH, NearestBH, ReflectiveBH, RandomBH
)


class TestBoundaryHandler:

    @pytest.fixture
    def standard_bh(self):
        return StandardBH()

    @pytest.fixture
    def bounds(self):
        lb = np.array([0, 0])
        ub = np.array([10, 10])

        return lb, ub

    @pytest.mark.parametrize('pos', [[-5, -5], [5, 5], [15, 15]])
    def test_standard(self, standard_bh, pos):
        pos = np.array(pos, dtype=np.float32)
        ret_pos = standard_bh(pos)
        assert np.array_equal(pos, ret_pos)

    # TODO >> Write better tests for these boundary handlers.
    @pytest.mark.parametrize('pos', [[-5, -5], [5, 5], [15, 15]])
    def test_nearest(self, bounds, pos):
        lb, ub = bounds
        arr_pos = np.array(pos, dtype=np.float32)
        bh = NearestBH(lb, ub)
        ret_pos = bh(arr_pos)

        assert len(pos) == len(ret_pos)
        assert np.logical_and(ret_pos >= lb, ret_pos <= ub).all()

        if pos == [-5, -5]:
            assert np.array_equal(ret_pos, lb)
        elif pos == [5, 5]:
            assert np.array_equal(ret_pos, arr_pos)
        elif pos == [15, 15]:
            assert np.array_equal(ret_pos, ub)

    @pytest.mark.parametrize('pos', [[-5, -5], [5, 5], [15, 15]])
    def test_reflective(self, bounds, pos):
        lb, ub = bounds
        arr_pos = np.array(pos, dtype=np.float32)
        bh = ReflectiveBH(lb, ub)
        ret_pos = bh(arr_pos)

        assert len(pos) == len(ret_pos)
        assert np.logical_and(ret_pos >= lb, ret_pos <= ub).all()
        if pos == [-5, -5]:
            assert np.array_equal(ret_pos, np.array([5, 5]))
        elif pos == [5, 5]:
            assert np.array_equal(ret_pos, arr_pos)
        elif pos == [15, 15]:
            assert np.array_equal(ret_pos, np.array([5, 5]))

    @pytest.mark.parametrize('pos', [[-5, -5], [5, 5], [15, 15]])
    def test_random(self, bounds, pos):
        lb, ub = bounds
        arr_pos = np.array(pos, dtype=np.float32)
        bh = RandomBH(lb, ub)
        ret_pos = bh(arr_pos)

        assert len(pos) == len(ret_pos)
        assert np.logical_and(ret_pos >= lb, ret_pos <= ub).all()
