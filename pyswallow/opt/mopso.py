import logging

import numpy as np

from pyswallow.opt.base_swarm import BaseSwarm
from ..constraints.constraint_manager import ConstraintManager
from ..handlers.archive import Archive
from ..handlers.boundary_handler import StandardBH
from ..handlers.inertia_handler import StandardIWH
from ..handlers.velocity_handler import StandardVH
from ..swallows.mo_swallow import MOSwallow
from ..utils.history import MOHistory
from ..utils.reporter import Reporter
from ..utils.termination_manager import IterationTerminationManager


class MOSwarm(BaseSwarm):

    def __init__(self, bounds, n_swallows, n_iterations,
                 w=0.7, c1=2.0, c2=2.0, debug=False):

        """
        Initialiser for the MOSwarm class.

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

        Attributes
        ----------
        rep : Reporter
            Provides ability to log / debug the optimisation.
        n_objs : int
            Number of objectives.
        archive : Archive
            Class to deal with Pareto dominance.
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

        super().__init__(bounds, n_swallows, w, c1, c2)

        log_level = logging.DEBUG if debug else logging.INFO
        self.rep = Reporter(lvl=log_level)

        self.n_objs = None
        self.archive = None

        self.iteration = 0
        self.n_iterations = n_iterations

        self.bh = StandardBH()
        self.vh = StandardVH()
        self.iwh = StandardIWH(self.w)

        self.history = MOHistory(self)

        self.constraint_manager = ConstraintManager(self)
        self.termiation_manager = IterationTerminationManager(self)

        self.rep.log(
            f'MOSwarm::__init__('
            f'n_swallows={n_swallows},'
            f'n_iterations={n_iterations},'
            f'bounds={bounds}'
            f')', lvl=logging.DEBUG
        )

    def reset_environment(self):

        """Responsible for resetting the optimisation environment."""

        self.iteration = 0
        self.population = []
        self.archive = Archive(self.n_objs)
        self.rep.log('MOSwarm::reset_environment()', lvl=logging.DEBUG)

    def initialise_swarm(self):

        """Initialises the population with MOSwallow objects."""

        self.population = [MOSwallow(self.bounds, self.n_objs)
                           for _ in range(self.n_swallows)]
        self.rep.log('MOSwarm::initialise_swarm()', lvl=logging.DEBUG)

    def initialise_archive(self):

        """Instantiates an Archive instance."""

        self.archive = Archive(self.n_objs)
        self.rep.log('MOSwarm::initialise_archive()', lvl=logging.DEBUG)

    @staticmethod
    def evaluate_fitness(swallow, fns):

        """
        Assesses the fitness of the swallow.

        Parameters
        ----------
        swallow : MOSwallow
            Swallow for which to assess the fitness.
        fns : list
            Functions to use in order to assess the fitness.
        """

        for idx, function in enumerate(fns):
            swallow.fitness[idx] = function(swallow.position)

    def update_velocity(self, swallow):

        """
        Updates the velocity of a given swallow.

        Parameters
        ----------
        swallow : MOSwallow
            Swallow for which to update the velocity.
        """

        _leader = self.archive.choose_leader()

        def inertial():
            return self.w * swallow.velocity

        def cognitive():
            return (self.c1 * np.random.uniform()
                    * (swallow.pbest_position - swallow.position))

        def social():
            return (self.c2 * np.random.uniform()
                    * (_leader.pbest_position - swallow.position))

        swallow.velocity = inertial() + cognitive() + social()
        swallow.velocity = self.vh(swallow.velocity)

        self.rep.log(
            f'MOSwarm::update_velocity(swallow={swallow})\t'
            f'velocity={swallow.velocity}',
            lvl=logging.DEBUG
        )

    @staticmethod
    def update_pbest(swallow):

        """
        Updates the pbest values of the swallow.

        Parameters
        ----------
        swallow : MOSwallow
            Swallow for which to update the pbest_fitness.
        """

        if swallow.self_dominate():
            swallow.pbest_position = swallow.position
            swallow.pbest_fitness = swallow.fitness

    def step_optimise(self, fns):

        """
        Runs one iteration of the optimisation process.

        Parameters
        ----------
        fns : list
            List of functions to optimise for.
        """

        self.w = self.iwh(self.iteration)

        for swallow in self.population:
            self.evaluate_fitness(swallow, fns)

            if not self.constraint_manager.violates_position(swallow):
                self.update_pbest(swallow)

        for swallow in self.population:
            if not self.constraint_manager.violates_position(swallow):
                self.archive.add_swallow(swallow)

        self.archive.pareto_front()
        self.archive.assign_sparsity()
        self.archive.sparsity_limit(n_limit=self.n_swallows)

        for swallow in self.population:
            self.update_velocity(swallow)
            swallow.move(self.bh)

        self.history.write_history()

        self.rep.log(
            f'iteration={self.iteration:05}\t'
            f'archive_length={len(self.archive.population):03}'
        )

    def optimise(self, fns):

        """
        Runs the entire optimisation process.

        Parameters
        ----------
        fns : list
            List of functions to optimise for.
        """

        self.reset_environment()
        self.n_objs = len(fns)

        self.initialise_swarm()
        self.initialise_archive()

        while not self.termiation_manager.termination_check():
            self.step_optimise(fns)
            self.iteration += 1

        self.rep.log('Optimisation complete...')
        for idx, swallow in enumerate(self.archive.population):
            self.rep.log(
                f'archive_swallow={idx:03}\t'
                f'fitness={swallow.fitness}\t'
                f'position={swallow.position}'
            )
