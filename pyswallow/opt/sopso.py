import copy
import logging

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

    def __init__(self, n_swallows, n_iterations, bounds,
                 w=0.7, c1=2.0, c2=2.0, debug=False):

        """
        Initialiser for the Swarm class.

        Parameters
        ----------
        n_swallows : int
            Population size.
        n_iterations : int
            Number of iterations to run optimisation for.
        bounds : dict
            Provides the upper and lower bounds of the search space.
        w : float
            Inertia weight.
        c1 : float
            Cognitive weight.
        c2 : float
            Social weight.
        debug : bool
            True if you want to log debugging, False otherwise.

        Attributes
        ---------
        rep : Reporter
            Provides ability to log / debug the optimisation.å
        gbest_swallow : Swallow
            Swallow with the current best fitness.
        iteration : int
            Current iteration of the optimisation procedure.
        bh : BaseHandler
            Manipulates position dependant on boundary interactions.
        vh : BaseHandler
            Manipulates the velocity dependant on boundary interactions.
        iwh : InertiaWeightHandler
            Manipulates the inertia weight.
        history : MOHistory
            Records the optimisation history.
        constraint_manager : ConstraintManager
            Determines if imposed constraints have been violated.
        termination_manager : BaseTerminationManager
            Determines whether termination criteria have been fulfilled.
        """

        super().__init__(n_swallows, bounds, w, c1, c2)

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

    def reset_environment(self):

        """Responsible for resetting the optimisation environment."""

        self.iteration = 0
        self.gbest_swallow = None
        self.population = []
        self.rep.log('Swarm::reset_environment()', lvl=logging.DEBUG)

    def initialise_swarm(self):

        """Initialises the population with Swallow objects."""

        self.population = [Swallow(self.bounds) for _ in range(self.n_swallows)]
        self.rep.log('Swarm::initialise_swarm()', lvl=logging.DEBUG)

    @staticmethod
    def evaluate_fitness(swallow, fn):

        """
        Assesses the fitness of the swallow.

        Parameters
        ----------
        swallow : MOSwallow
            Swallow for which to assess the fitness.
        fn : function
            Function to use in order to assess the fitness.
        """

        swallow.fitness = fn(swallow.position)

    def update_velocity(self, swallow):

        """
        Updates the velocity of a given swallow.

        Parameters
        ----------
        swallow : MOSwallow
            Swallow for which to update the velocity.
        """

        def inertial():
            return self.w * swallow.velocity

        def cognitive():
            return (self.c1 * np.random.uniform()
                    * (swallow.pbest_position - swallow.position))

        def social():
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
    def pbest_update(swallow):

        """
        Updates the pbest values of the swallow.

        Parameters
        ----------
        swallow : Swallow
            Swallow for which to update the pbest_fitness.
        """

        if swallow.fitness < swallow.pbest_fitness:
            swallow.pbest_fitness = swallow.fitness
            swallow.pbest_position = swallow.position

    def gbest_update(self, swallow):

        """
        Updates the gbest value of the swarm.

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

    def step_optimise(self, fn):

        """
        Runs one iteration of the optimisation process.

        Parameters
        ----------
        fn : function
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

    def optimise(self, fn):

        """
        Runs the entire optimisation process.

        Parameters
        ----------
        fn : function
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

