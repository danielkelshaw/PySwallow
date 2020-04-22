import pytest

from pyswallow.handlers.inertia_handler import StandardIWH, LinearIWH


class TestInertiaWeightHandler:

    @pytest.fixture
    def standard_iwh(self):
        return StandardIWH(0.7)

    @pytest.mark.parametrize('iteration', [0, 5, 10])
    def test_standard_iwh(self, standard_iwh, iteration):
        ret_it = standard_iwh(iteration)
        assert ret_it == 0.7

    @pytest.mark.parametrize('iteration', [0, 5, 10])
    def test_linear_iwh(self, iteration):
        iw_handler = LinearIWH(0.7, 0.3, 10)
        ret_it = iw_handler(iteration)

        if iteration == 0:
            assert ret_it == 0.7
        elif iteration == 5:
            assert ret_it == 0.5
        elif iteration == 10:
            assert ret_it == 0.3
