from abc import ABC, abstractmethod

import numpy as np


class BaseSwallow(ABC):

    def __init__(self, bounds):

        """
        Initialiser for BaseSwallow class.

        Parameters
        ----------
        bounds : dict
            Bounds to impose on the search space.

        Attributes
        ----------
        lb : np.ndarray
            Lower bound of search space.
        ub : np.ndarray
            Upper bound of search space.
        position : np.ndarray
            Current position of the swallow.
        velocity : np.ndarray
            Current velocity of the swallow.
        fitness : float or list
            Current fitness of the swallow.
        pbest_position : np.ndarray
            The position of the swallow at the best fitness evaluation.
        pbest_fitness : float or list
            The current best fitness evaluation.
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
    def move(self, bh):
        raise NotImplementedError('BaseSwallow::move()')
