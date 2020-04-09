import copy
import numpy as np


class Archive:

    def __init__(self, n_objectives):

        self.population = []
        self.n_objectives = n_objectives

    def add_swallow(self, swallow):
        self.population.append(swallow)

    def pareto_front(self):

        pf = []

        for idx, swallow in enumerate(self.population):

            pf.append(swallow)

            for opp_idx, opp_swallow in enumerate(pf[:-1]):
                if opp_swallow.dominate(swallow):
                    del pf[-1]
                    break
                elif swallow.dominate(opp_swallow):
                    del pf[opp_idx]

        self.population = pf

    def assign_sparsity(self):

        _population = copy.deepcopy(self.population)

        for swallow in _population:
            swallow.sparsity = 0

        for obj in range(self.n_objectives):
            _population = sorted(_population, key=lambda x: x.fitness[obj])
            _population[0].sparsity = float('inf')
            _population[-1].sparsity = float('inf')

            for i in range(1, len(_population) - 1):
                _sparse = (_population[i + 1].fitness[obj]
                           - _population[i - 1].fitness[obj])

                _population[i].sparsity += _sparse

        self.population = _population

    def sparsity_limit(self, n_limit):

        if len(self.population) > n_limit:
            self.population = sorted(self.population,
                                     key=lambda x: x.sparsity,
                                     reverse=True)[:n_limit]

    def choose_leader(self, method=0):

        if method == 0:
            return copy.deepcopy(np.random.choice(self.population))

        if method == 1:
            if len(self.population) <= self.n_objectives:
                return copy.deepcopy(np.random.choice(self.population))
            else:
                sparsist_leader = sorted(self.population,
                                         key=lambda x: x.sparsity,
                                         reverse=True)[self.n_objectives]
                return copy.deepcopy(sparsist_leader)
