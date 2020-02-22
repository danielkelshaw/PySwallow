from ..base.base_swarm import BaseSwarm
from ..base.base_swallow import BaseSwallow
from ..utils.reporter import Reporter
from ..handlers.boundary_handler import StandardBH
from ..handlers.velocity_handler import StandardVH
from ..handlers.inertia_handler import StandardIWH

import numpy as np
import logging
import copy


class Swallow(BaseSwallow):

    def __init__(self, lb, ub):
        super().__init__(lb, ub)

    def move(self, bh):
        self.position += self.velocity
        self.position = bh(self.position)


class Swarm(BaseSwarm):

    def __init__(self, obj_function, n_swallows, n_iterations, lb, ub,
                 constraints=None, w=0.7, c1=2.0, c2=2.0, debug=False):

        super().__init__(n_swallows, lb, ub, w, c1, c2)

        self.gbest_position = np.random.uniform(self.lb, self.ub)
        self.gbest_fitness = float('inf')

        log_debug = logging.DEBUG if debug else logging.INFO
        self.rep = Reporter(lvl=log_debug)

        self.obj_function = obj_function
        self.constraints = constraints

        self.iteration = 0
        self.n_iterations = n_iterations

        self.bh = StandardBH()
        self.vh = StandardVH()
        self.iwh = StandardIWH(self.w)

        self.rep.log('Swarm initialised successfully')

    # Reset Methods
    def reset_environment(self):
        self.iteration = 0
        self.gbest_position = np.random.uniform(self.lb, self.ub,
                                                size=self.lb.shape[0])
        self.gbest_fitness = float('inf')
        self.reset_populations()
        self.rep.log('Environment reset', lvl=logging.DEBUG)

    def reset_populations(self):
        self.population = []
        self.rep.log('Populations reset', lvl=logging.DEBUG)

    # Initialisation
    def initialise_swarm(self):
        self.population = [Swallow(self.lb, self.ub)
                           for _ in range(self.n_swallows)]
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
                    * (self.gbest_position - swallow.position))

        swallow.velocity = inertial() + cognitive() + social()
        swallow.velocity = self.vh(swallow.velocity)

    def swarm_update_velocity(self):
        for swallow in self.population:
            self.update_velocity(swallow)

    def swarm_move(self):
        for swallow in self.population:
            swallow.move(self.bh)

    def pbest_update(self, swallow):
        if self.constraints is not None:
            if self.constraints(swallow.position):
                if swallow.fitness < swallow.pbest_fitness:
                    swallow.pbest_fitness = copy.deepcopy(swallow.fitness)
                    swallow.pbest_position = copy.deepcopy(swallow.position)
        else:
            if swallow.fitness < swallow.pbest_fitness:
                swallow.pbest_fitness = copy.deepcopy(swallow.fitness)
                swallow.pbest_position = copy.deepcopy(swallow.position)

    def gbest_update(self, swallow):
        if self.constraints is not None:
            if self.constraints(swallow.position):
                if swallow.fitness < self.gbest_fitness:
                    self.gbest_fitness = copy.deepcopy(swallow.fitness)
                    self.gbest_position = copy.deepcopy(swallow.position)
        else:
            if swallow.fitness < self.gbest_fitness:
                self.gbest_fitness = copy.deepcopy(swallow.fitness)
                self.gbest_position = copy.deepcopy(swallow.position)

    def swarm_update_best(self):
        for swallow in self.population:
            self.pbest_update(swallow)
            self.gbest_update(swallow)

    # Optimise
    def termination_check(self):
        if self.iteration >= self.n_iterations:
            return False
        else:
            return True

    def optimise(self):
        self.reset_environment()

        self.initialise_swarm()

        while self.termination_check():

            print('Iteration {0}: {1}'.format(self.iteration,
                                              self.gbest_fitness))

            self.w = self.iwh(self.iteration)

            self.swarm_evaluate_fitness()
            self.swarm_update_best()

            self.swarm_update_velocity()
            self.swarm_move()

            self.rep.log(
                'Iteration {}\t'
                'GBest Fitness = {:.3f}\t'
                'GBest Position = {}'
                ''.format(self.iteration,
                          self.gbest_fitness,
                          self.gbest_position)
            )

            self.iteration += 1

        self.rep.log('Optimisation complete...')
        self.rep.log('GBest Fitness = {:.3f}\t'
                     'GBest Position = {}'.format(self.gbest_fitness,
                                                  self.gbest_position))
