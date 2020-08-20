from abc import ABC, abstractmethod
from typing import NoReturn

import numpy as np

from ..handlers.base_handler import BaseHandler


class BaseSwallow(ABC):

    def __init__(self, bounds: dict) -> None:

        """BaseSwallow Class.

        Parameters
        ----------
        bounds : dict
            Bounds to impose on the search space.
        """

        if not isinstance(bounds, dict):
            raise TypeError('bounds must be dict.')

        self._pnames = list(bounds.keys())
        _bounds = np.asarray(list(bounds.values()))

        self.lb = _bounds[:, 0]
        self.ub = _bounds[:, 1]

        self.position = np.random.uniform(self.lb, self.ub)
        self.velocity = np.random.uniform(self.lb, self.ub)
        self.fitness = None

        self.pbest_position = self.position
        self.pbest_fitness = float('inf')

    @abstractmethod
    def move(self, bh: BaseHandler) -> NoReturn:
        raise NotImplementedError('BaseSwallow::move()')
