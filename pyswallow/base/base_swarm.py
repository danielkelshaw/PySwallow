import numpy as np
from abc import ABC, abstractmethod


class BaseSwarm(ABC):

    def __init__(self, n_swallows, lb, ub, w, c1, c2):

        self.n_swallows = n_swallows

        self.lb = np.array(lb)
        self.ub = np.array(ub)

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.gbest_position = np.random.uniform(self.lb, self.ub)
        self.gbest_fitness = float('inf')

        self.population = []

    @abstractmethod
    def reset_environment(self):
        raise NotImplementedError('BaseSwarm::reset_environment()')

    @abstractmethod
    def termination_check(self):
        raise NotImplementedError('BaseSwarm::termination_check()')

    @abstractmethod
    def initialise_swarm(self):
        raise NotImplementedError('BaseSwarm::initialise_swarm()')

    @abstractmethod
    def update_velocity(self, swallow):
        raise NotImplementedError('BaseSwarm::update_velocity()')

    @abstractmethod
    def evaluate_fitness(self, swallow):
        raise NotImplementedError('BaseSwarm::evaluate_fitness()')

    @abstractmethod
    def pbest_update(self, swallow):
        raise NotImplementedError('BaseSwarm::pbest_update()')

    @abstractmethod
    def gbest_update(self, swallow):
        raise NotImplementedError('BaseSwarm::gbest_update()')

    def swarm_update_velocity(self):
        for swallow in self.population:
            self.update_velocity(swallow)

    def swarm_evaluate_fitness(self):
        for swallow in self.population:
            self.evaluate_fitness(swallow)

    def swarm_update_best(self):
        for swallow in self.population:
            self.pbest_update(swallow)
            self.gbest_update(swallow)
