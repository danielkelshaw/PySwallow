import numpy as np
from abc import ABC, abstractmethod


class BaseSwarm(ABC):

    def __init__(self, n_swallows, bounds, w, c1, c2):

        self.n_swallows = n_swallows

        if not isinstance(bounds, dict):
            raise TypeError('bounds must be dict.')

        self.bounds = bounds
        self.pnames = list(bounds.keys())

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.population = []

    @abstractmethod
    def reset_environment(self):
        raise NotImplementedError('BaseSwarm::reset_environment()')

    @abstractmethod
    def initialise_swarm(self):
        raise NotImplementedError('BaseSwarm::initialise_swarm()')

    @abstractmethod
    def update_velocity(self, swallow):
        raise NotImplementedError('BaseSwarm::update_velocity()')

    @staticmethod
    @abstractmethod
    def evaluate_fitness(*args):
        raise NotImplementedError('BaseSwarm::evaluate_fitness()')

    @abstractmethod
    def step_optimise(self, fn):
        raise NotImplementedError('BaseSwarm::step_optimise()')

    @abstractmethod
    def optimise(self, fn):
        raise NotImplementedError('BaseSwarm::optimise()')
