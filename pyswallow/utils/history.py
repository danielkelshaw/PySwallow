import abc
from typing import NoReturn

import numpy as np

from ..opt.base_swarm import BaseSwarm
from ..opt.mopso import MOSwarm
from ..opt.sopso import Swarm


class BaseHistory(abc.ABC):

    def __init__(self, swarm: BaseSwarm) -> None:

        """BaseHistory Class.

        Parameters
        ----------
        swarm : BaseSwarm
            Swarm for which to record the history.
        """

        self.swarm = swarm

    @abc.abstractmethod
    def write_history(self) -> NoReturn:

        """Records the history for the swarm."""

        raise NotImplementedError('BaseHistory::write_history()')


class SOHistory(BaseHistory):

    def __init__(self, swarm: Swarm) -> None:
        super().__init__(swarm)

        self.arr_best_fitness = []
        self.arr_mean_fitness = []

    def write_history(self) -> None:
        best_fitness = self.swarm.gbest_swallow.fitness
        self.arr_best_fitness.append(best_fitness)

        mean_fitness = np.mean([i.fitness for i in self.swarm.population])
        self.arr_mean_fitness.append(mean_fitness)


class MOHistory(BaseHistory):

    def __init__(self, swarm: MOSwarm) -> None:
        super().__init__(swarm)

        self.arr_mean_fitness = []

    def write_history(self) -> None:
        mean_fitness = np.mean([i.fitness for i in self.swarm.population])
        self.arr_mean_fitness.append(mean_fitness)
