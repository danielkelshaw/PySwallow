import pytest
import numpy as np

import pyswallow as ps
from pyswallow.handlers.archive import Archive


class TestArchive:

    @pytest.fixture
    def archive(self):
        return Archive(n_objectives=2)

    @pytest.fixture
    def pop_archive(self, archive):
        for i in range(30):
            lb = np.array([0, 0])
            ub = np.array([5, 5])
            swallow = ps.MOSwallow(lb, ub, 2)
            swallow.fitness = [i, i]
            archive.add_swallow(swallow)
        return archive

    @pytest.fixture
    def swallow(self):
        lb = np.array([-50, -50])
        ub = np.array([50, 50])
        return ps.MOSwallow(lb, ub, 2)

    def test_add_swallow(self, archive, swallow):
        assert len(archive.population) == 0
        archive.add_swallow(swallow)
        assert len(archive.population) == 1

    def test_assign_sparsity(self, pop_archive):
        pop_archive.assign_sparsity()
        for idx, swallow in enumerate(pop_archive.population):
            if idx == 0 or idx == 29:
                assert swallow.sparsity == float('inf')
            else:
                assert swallow.sparsity == 4

    @pytest.mark.parametrize('n_limit', [15,  30, 45])
    def test_sparsity_limit(self, pop_archive, n_limit):
        pop_archive.assign_sparsity()
        pop_archive.sparsity_limit(n_limit)
        _s = sorted([s.sparsity for s in pop_archive.population], reverse=True)

        assert len(pop_archive.population) <= n_limit
        assert _s[:2] == [float('inf'), float('inf')]

    @pytest.mark.parametrize('method', [0, 1])
    def test_choose_leader(self, pop_archive, method):
        pop_archive.assign_sparsity()
        leader = pop_archive.choose_leader(method)

        assert isinstance(leader, ps.MOSwallow)

        if method == 1:
            assert leader.sparsity == 4
