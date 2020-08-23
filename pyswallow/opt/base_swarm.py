from abc import ABC, abstractmethod
from typing import Any, Callable, NoReturn

import numpy as np

from ..swallows.base_swallow import BaseSwallow


class BaseSwarm(ABC):

    def __init__(self,
                 bounds: dict,
                 n_swallows: int,
                 w: float,
                 c1: float,
                 c2: float) -> None:

        """BaseSwarm Class.

        Parameters
        ----------
        bounds : dict
            Bounds to impose on the search space.
        n_swallows : int
            Number of swallows for use in the population.
        w : float
            Inertia weight.
        c1 : float
            Cognitive weight.
        c2 : float
            Social weight.
        """

        self.n_swallows = n_swallows

        if not isinstance(bounds, dict):
            raise TypeError('bounds must be dict.')

        self.bounds = bounds
        _bounds = np.asarray(list(bounds.values()))
        self.lb = _bounds[:, 0]
        self.ub = _bounds[:, 1]

        self.pnames = list(bounds.keys())

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.iteration = None
        self.n_iterations = None

        self.population = []

    @abstractmethod
    def reset_environment(self) -> NoReturn:
        raise NotImplementedError('BaseSwarm::reset_environment()')

    @abstractmethod
    def initialise_swarm(self) -> NoReturn:
        raise NotImplementedError('BaseSwarm::initialise_swarm()')

    @abstractmethod
    def update_velocity(self, swallow: BaseSwallow) -> NoReturn:
        raise NotImplementedError('BaseSwarm::update_velocity()')

    @staticmethod
    @abstractmethod
    def evaluate_fitness(*args: Any) -> NoReturn:
        raise NotImplementedError('BaseSwarm::evaluate_fitness()')

    @abstractmethod
    def step_optimise(self, fn: Callable[[Any], Any]) -> NoReturn:
        raise NotImplementedError('BaseSwarm::step_optimise()')

    @abstractmethod
    def optimise(self, fn: Callable[[Any], Any]) -> NoReturn:
        raise NotImplementedError('BaseSwarm::optimise()')
