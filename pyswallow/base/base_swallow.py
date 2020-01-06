import numpy as np
from abc import ABC, abstractmethod


class BaseSwallow(ABC):

    def __init__(self, lb, ub):

        self.lb = lb
        self.ub = ub

        self.position = np.random.uniform(lb, ub, size=lb.shape[0])
        self.velocity = np.random.uniform(lb, ub, size=lb.shape[0])
        self.fitness = None

        self.pbest_position = self.position
        self.pbest_fitness = float('inf')

    @abstractmethod
    def move(self):
        raise NotImplementedError('BaseSwallow::move()')
