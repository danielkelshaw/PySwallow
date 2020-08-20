import copy
import logging
from typing import Callable

import numpy as np

from pyswallow.opt.base_swarm import BaseSwarm
from ..constraints.constraint_manager import ConstraintManager
from ..handlers.boundary_handler import StandardBH
from ..handlers.inertia_handler import StandardIWH
from ..handlers.velocity_handler import StandardVH
from ..swallows.so_swallow import Swallow
from ..utils.history import SOHistory
from ..utils.reporter import Reporter
from ..utils.termination_manager import IterationTerminationManager


class Swarm(BaseSwarm):

    def __init__(self,
                 bounds: dict,
                 n_swallows: int,
                 n_iterations: int,
                 w: float = 0.7,
                 c1: float = 2.0,
                 c2: float = 2.0,
                 debug: bool = False) -> None:

        """Swarm Class.

        Parameters
        ----------
        bounds : dict
            Provides the upper and lower bounds of the search space.
        n_swallows : int
            Population size.
        n_iterations : int
            Number of iterations to run optimisation for.
        w : float
            Inertia weight.
        c1 : float
            Cognitive weight.
        c2 : float
            Social weight.
        debug : bool
            True if you want to log debugging, False otherwise.
        """

        super().__init__(bounds, n_swallows, w, c1, c2)

        self.gbest_swallow = None

        log_level = logging.DEBUG if debug else logging.INFO
        self.rep = Reporter(lvl=log_level)

        self.iteration = 0
        self.n_iterations = n_iterations

        self.bh = StandardBH()
        self.vh = StandardVH()
        self.iwh = StandardIWH(self.w)

        self.history = SOHistory(self)

        self.constraints_manager = ConstraintManager(self)
        self.termination_manager = IterationTerminationManager(self)

        self.rep.log(
            f'Swarm::__init__('
            f'n_swallows={n_swallows},'
            f'n_iterations={n_iterations},'
            f'bounds={bounds}'
            f')', lvl=logging.DEBUG)

    def reset_environment(self) -> None:

        """Responsible for resetting the optimisation environment."""

        self.iteration = 0
        self.gbest_swallow = None
        self.population = []
        self.rep.log('Swarm::reset_environment()', lvl=logging.DEBUG)

    def initialise_swarm(self) -> None:

        """Initialises the population with Swallow objects."""

        self.population = [Swallow(self.bounds) for _ in range(self.n_swallows)]
        self.rep.log('Swarm::initialise_swarm()', lvl=logging.DEBUG)

    @staticmethod
    def evaluate_fitness(swallow: Swallow, fn: Callable[[np.ndarray], np.ndarray]) -> None:

        """Assesses the fitness of the swallow.

        Parameters
        ----------
        swallow : MOSwallow
            Swallow for which to assess the fitness.
        fn : Callable[[np.ndarray], np.ndarray]
            Function to use in order to assess the fitness.
        """

        swallow.fitness = fn(swallow.position)

    def update_velocity(self, swallow: Swallow) -> None:

        """Updates the velocity of a given swallow.

        Parameters
        ----------
        swallow : MOSwallow
            Swallow for which to update the velocity.
        """

        def inertial() -> np.ndarray:
            return self.w * swallow.velocity

        def cognitive() -> np.ndarray:
            return (self.c1 * np.random.uniform()
                    * (swallow.pbest_position - swallow.position))

        def social() -> np.ndarray:
            return (self.c2 * np.random.uniform()
                    * (self.gbest_swallow.position - swallow.position))

        swallow.velocity = inertial() + cognitive() + social()
        swallow.velocity = self.vh(swallow.velocity)

        self.rep.log(
            f'Swarm::update_velocity(swallow={swallow})\t'
            f'velocity={swallow.velocity}',
            lvl=logging.DEBUG
        )

    @staticmethod
    def pbest_update(swallow: Swallow) -> None:

        """Updates the pbest values of the swallow.

        Parameters
        ----------
        swallow : Swallow
            Swallow for which to update the pbest_fitness.
        """

        if swallow.fitness < swallow.pbest_fitness:
            swallow.pbest_fitness = swallow.fitness
            swallow.pbest_position = swallow.position

    def gbest_update(self, swallow: Swallow) -> None:

        """Updates the gbest value of the swarm.

        Parameters
        ----------
        swallow : Swallow
            Swallow with which to update the gbest_swallow.
        """

        if (self.gbest_swallow is None or
                swallow.fitness < self.gbest_swallow.fitness):
            self.gbest_swallow = copy.deepcopy(swallow)

        self.rep.log(
            f'Swarm::gbest_update({swallow})\t'
            f'gbest_swallow={self.gbest_swallow}',
            lvl=logging.DEBUG
        )

    def step_optimise(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:

        """Runs one iteration of the optimisation process.

        Parameters
        ----------
        fn : Callable[[np.ndarray], np.ndarray]
            Funnction to optimise for.
        """

        self.w = self.iwh(self.iteration)

        for swallow in self.population:
            self.evaluate_fitness(swallow, fn)

            if not self.constraints_manager.violates_position(swallow):
                self.gbest_update(swallow)
                self.pbest_update(swallow)

        for swallow in self.population:
            self.update_velocity(swallow)
            swallow.move(self.bh)

        self.history.write_history()

        mean_fitness = np.mean([s.fitness for s in self.population])
        self.rep.log(
            f'iteration={self.iteration:05}\t'
            f'mean_fitness={mean_fitness:.3f}\t'
            f'gbest_fitness={self.gbest_swallow.fitness:.3f}\t'
            f'gbest_position={self.gbest_swallow.position}'
        )

    def optimise(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:

        """Runs the entire optimisation process.

        Parameters
        ----------
        fn : Callable[[np.ndarray], np.ndarray]
            Function to optimise for.
        """

        self.reset_environment()
        self.initialise_swarm()

        while not self.termination_manager.termination_check():
            self.step_optimise(fn)
            self.iteration += 1

        self.rep.log('Optimisation complete...')

        self.rep.log(
            f'\tgbest_fitness={self.gbest_swallow.fitness:.3f}\n'
            f'\tgbest_position={self.gbest_swallow.position}'
        )
