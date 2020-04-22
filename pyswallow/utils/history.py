import abc

import numpy as np


class BaseHistory(abc.ABC):

    def __init__(self, swarm):
        self.swarm = swarm

    @abc.abstractmethod
    def write_history(self):
        raise NotImplementedError('BaseHistory::write_history()')


class SOHistory(BaseHistory):

    def __init__(self, swarm):
        super().__init__(swarm)

        self.arr_best_fitness = []
        self.arr_mean_fitness = []

    def write_history(self):

        best_fitness = self.swarm.gbest_swallow.fitness
        self.arr_best_fitness.append(best_fitness)

        mean_fitness = np.mean([i.fitness for i in self.swarm.population])
        self.arr_mean_fitness.append(mean_fitness)


class MOHistory(BaseHistory):

    def __init__(self, swarm):
        super().__init__(swarm)

        self.arr_mean_fitness = []

    def write_history(self):

        mean_fitness = np.mean([i.fitness for i in self.swarm.population])
        self.arr_mean_fitness.append(mean_fitness)
