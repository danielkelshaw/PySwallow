import copy
import logging
import numpy as np

from ..base.base_swarm import BaseSwarm
from ..base.base_swallow import BaseSwallow

from ..constraints.constraint_manager import ConstraintManager

from ..utils.reporter import Reporter
from ..utils.history import SOHistory
from ..utils.termination_manager import IterationTerminationManager

from ..handlers.boundary_handler import StandardBH
from ..handlers.velocity_handler import StandardVH
from ..handlers.inertia_handler import StandardIWH


class Swallow(BaseSwallow):

    def __init__(self, bounds):
        super().__init__(bounds)

    def move(self, bh):
        self.position += self.velocity
        self.position = bh(self.position)


class Swarm(BaseSwarm):

    def __init__(self, n_swallows, n_iterations, bounds,
                 w=0.7, c1=2.0, c2=2.0, debug=False):

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

        self.rep.log('Swarm::__init__()')

    def reset_environment(self):
        self.iteration = 0
        self.gbest_swallow = None
        self.population = []
        self.rep.log('Swarm::reset_environment()', lvl=logging.DEBUG)

    def initialise_swarm(self):
        self.population = [Swallow(self.bounds) for _ in range(self.n_swallows)]
        self.rep.log('Swarm::initialise_swarm()', lvl=logging.DEBUG)

    @staticmethod
    def evaluate_fitness(swallow, fn):
        swallow.fitness = fn(swallow.position)

    def update_velocity(self, swallow):

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

    @staticmethod
    def pbest_update(swallow):
        if swallow.fitness < swallow.pbest_fitness:
            swallow.pbest_fitness = swallow.fitness
            swallow.pbest_position = swallow.position

    def gbest_update(self, swallow):
        if (self.gbest_swallow is None or
                swallow.fitness < self.gbest_swallow.fitness):
            self.gbest_swallow = copy.deepcopy(swallow)

    def step_optimise(self, fn):

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
