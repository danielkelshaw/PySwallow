from __future__ import annotations

from .base_swallow import BaseSwallow
from ..handlers.boundary_handler import BaseBoundaryHandler


class MOSwallow(BaseSwallow):

    def __init__(self, bounds: dict, n_obj: int) -> None:

        """MOSwallow Class.

        Parameters
        ----------
        bounds : dict
            Provides the upper and lower bounds of the search space.
        n_obj : int
            Number of objectives.
        """

        super().__init__(bounds)
        self.n_obj = n_obj

        self.fitness = [None] * n_obj
        self.pbest_fitness = [float('inf')] * n_obj
        self.sparsity = 0

    def move(self, bh: BaseBoundaryHandler) -> None:

        """Responsible for moving the swallow in the search space.

        Parameters
        ----------
        bh : BaseHandler
            Alters the position according to the strategy.
        """

        self.position += self.velocity
        self.position = bh(self.position)

    def dominate(self, opponent: MOSwallow) -> bool:

        """Returns the dominant swallow.

        Parameters
        ----------
        opponent : MOSwallow
            The swallow with which to compare fitness.

        Returns
        -------
        self_dominates : bool
            True if self dominates opponent, False otherwise.
        """

        self_dominates = False

        for i in range(self.n_obj):
            if self.fitness[i] < opponent.fitness[i]:
                self_dominates = True
            elif opponent.fitness[i] < self.fitness[i]:
                return False

        return self_dominates

    def self_dominate(self) -> bool:

        """Determines if new fitness dominates pbest_fitness.

        Returns
        -------
        new_fitness_dominates : bool
            True if fitness dominates pbest_fitness, False otherwise.
        """

        new_fitness_dominates = False

        for i in range(self.n_obj):
            if self.fitness[i] < self.pbest_fitness[i]:
                new_fitness_dominates = True
            elif self.pbest_fitness[i] < self.fitness[i]:
                return False

        return new_fitness_dominates
