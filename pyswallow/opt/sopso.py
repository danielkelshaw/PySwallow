from ..base.base_swarm import BaseSwarm
from ..base.base_swallow import BaseSwallow

from ..constraints.constraint_manager import ConstraintManager

from ..utils.reporter import Reporter
from ..utils.history import GeneralHistory
from ..utils.termination_manager import IterationTerminationManager

from ..handlers.boundary_handler import StandardBH
from ..handlers.velocity_handler import StandardVH
from ..handlers.inertia_handler import StandardIWH

import numpy as np
import logging
import copy


class Swallow(BaseSwallow):

    def __init__(self, bounds):
        super().__init__(bounds)

    def move(self, bh):
        self.position += self.velocity
        self.position = bh(self.position)


class Swarm(BaseSwarm):

    def __init__(self, obj_function, n_swallows, n_iterations, bounds,
                 w=0.7, c1=2.0, c2=2.0, debug=False):

        super().__init__(n_swallows, bounds, w, c1, c2)

        self.gbest_swallow = None

        log_debug = logging.DEBUG if debug else logging.INFO
        self.rep = Reporter(lvl=log_debug)

        self.obj_function = obj_function

        self.iteration = 0
        self.n_iterations = n_iterations

        self.bh = StandardBH()
        self.vh = StandardVH()
        self.iwh = StandardIWH(self.w)

        self.history = GeneralHistory(self)

        self.constraints_manager = ConstraintManager(self)
        self.termination_manager = IterationTerminationManager(self)

        self.rep.log('Swarm initialised successfully')

    # Reset Methods
    def reset_environment(self):
        self.iteration = 0
        self.gbest_swallow = None

        self.reset_populations()
        self.rep.log('Environment reset', lvl=logging.DEBUG)

    def reset_populations(self):
        self.population = []
        self.rep.log('Populations reset', lvl=logging.DEBUG)

    # Initialisation
    def initialise_swarm(self):
        self.population = [Swallow(self.bounds) for _ in range(self.n_swallows)]
        self.rep.log('Population initialised', lvl=logging.DEBUG)

    # Update Methods
    def evaluate_fitness(self, swallow):
        swallow.fitness = self.obj_function(swallow.position)

    def swarm_evaluate_fitness(self):
        for swallow in self.population:
            self.evaluate_fitness(swallow)

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

    def swarm_update_velocity(self):
        for swallow in self.population:
            self.update_velocity(swallow)

    def swarm_move(self):
        for swallow in self.population:
            swallow.move(self.bh)

    def pbest_update(self, swallow):
        if swallow.fitness < swallow.pbest_fitness:
            swallow.pbest_fitness = copy.deepcopy(swallow.fitness)
            swallow.pbest_position = copy.deepcopy(swallow.position)

    def gbest_update(self, swallow):
        if self.gbest_swallow is None:
            self.gbest_swallow = copy.deepcopy(swallow)
        elif swallow.fitness < self.gbest_swallow.fitness:
            self.gbest_swallow = copy.deepcopy(swallow)

    def swarm_update_best(self):
        for swallow in self.population:
            if not self.constraints_manager.violates_position(swallow):
                self.pbest_update(swallow)
                self.gbest_update(swallow)

    def step_optimise(self):
        self.w = self.iwh(self.iteration)

        self.swarm_evaluate_fitness()
        self.swarm_update_best()

        self.swarm_update_velocity()
        self.swarm_move()

        self.history.write_history()

        self.rep.log(
            'Iteration {}\t'
            'GBest Fitness = {:.3f}\t'
            'GBest Position = {}'
            ''.format(self.iteration,
                      self.gbest_swallow.fitness,
                      self.gbest_swallow.position)
        )

    def optimise(self):

        self.reset_environment()
        self.initialise_swarm()

        while not self.termination_manager.termination_check():
            self.step_optimise()
            self.iteration += 1

        self.rep.log('Optimisation complete...')
        self.rep.log('GBest Fitness = {:.3f}\t'
                     'GBest Position = {}'.format(self.gbest_swallow.fitness,
                                                  self.gbest_swallow.position))
