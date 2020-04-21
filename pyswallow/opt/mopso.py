from ..base.base_swarm import BaseSwarm
from ..base.base_swallow import BaseSwallow

from ..constraints.constraint_manager import ConstraintManager

from ..utils.reporter import Reporter
from ..utils.history import GeneralHistory
from ..utils.termination_manager import IterationTerminationManager

from ..handlers.boundary_handler import StandardBH
from ..handlers.velocity_handler import StandardVH
from ..handlers.inertia_handler import StandardIWH
from ..handlers.archive import Archive

import numpy as np
import matplotlib.pyplot as plt
import logging


class MOSwallow(BaseSwallow):

    def __init__(self, bounds, n_obj):

        super().__init__(bounds)
        self.n_obj = n_obj

        self.fitness = [None] * n_obj
        self.pbest_fitness = [float('inf')] * n_obj
        self.sparsity = 0

    def move(self, bh):

        self.position += self.velocity
        self.position = bh(self.position)

    def dominate(self, opponent):

        self_dominates = False

        for i in range(self.n_obj):
            if self.fitness[i] < opponent.fitness[i]:
                self_dominates = True
            elif opponent.fitness[i] < self.fitness[i]:
                return False

        return self_dominates

    def self_dominate(self):

        new_fitness_dominates = False

        for i in range(self.n_obj):
            if self.fitness[i] < self.pbest_fitness[i]:
                new_fitness_dominates = True
            elif self.pbest_fitness[i] < self.fitness[i]:
                return False

        return new_fitness_dominates


class MOSwarm(BaseSwarm):

    def __init__(self, obj_functions, n_swallows, n_iterations, bounds,
                 w=0.7, c1=2.0, c2=2.0, debug=False):

        super().__init__(n_swallows, bounds, w, c1, c2)

        log_debug = logging.DEBUG if debug else logging.INFO
        self.rep = Reporter(lvl=log_debug)

        self.obj_functions = obj_functions
        self.n_objs = len(self.obj_functions)

        self.archive = Archive(self.n_objs)

        self.iteration = 0
        self.n_iterations = n_iterations

        self.bh = StandardBH()
        self.vh = StandardVH()
        self.iwh = StandardIWH(self.w)

        self.history = GeneralHistory(self)

        self.constraint_manager = ConstraintManager(self)
        self.termiation_manager = IterationTerminationManager(self)

        self.rep.log('Swarm initialised successfully')

    # Reset methods
    def reset_environment(self):
        self.iteration = 0
        self.reset_populations()
        self.rep.log('Environment reset', lvl=logging.DEBUG)

    def reset_populations(self):
        self.population = []
        self.archive = Archive(self.n_objs)
        self.rep.log('Populations reset', lvl=logging.DEBUG)

    # Initialisation
    def initialise_swarm(self):
        self.population = [MOSwallow(self.bounds, self.n_objs)
                           for _ in range(self.n_swallows)]
        self.rep.log('Population initialised', lvl=logging.DEBUG)

    def initialise_archive(self):
        self.archive = Archive(self.n_objs)

    # Update Methods
    def evaluate_fitness(self, swallow):
        for idx, function in enumerate(self.obj_functions):
            swallow.fitness[idx] = function(swallow.position)

    def swarm_evaluate_fitness(self):
        for swallow in self.population:
            self.evaluate_fitness(swallow)

    def update_velocity(self, swallow):

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

    def swarm_update_velocity(self):
        for swallow in self.population:
            self.update_velocity(swallow)

    def swarm_move(self):
        for swallow in self.population:
            swallow.move(self.bh)

    @staticmethod
    def update_pbest(swallow):
        # TODO [CONSIDER] >> Should the Swallow class handle this?
        if swallow.self_dominate():
            swallow.pbest_position = swallow.position
            swallow.pbest_fitness = swallow.fitness

    def swarm_update_pbest(self):
        for swallow in self.population:
            if not self.constraint_manager.violates_position(swallow):
                self.update_pbest(swallow)

    def step_optimise(self):

        self.w = self.iwh(self.iteration)

        self.swarm_evaluate_fitness()
        self.swarm_update_pbest()

        for swallow in self.population:
            self.archive.add_swallow(swallow)

        self.archive.pareto_front()
        self.archive.assign_sparsity()
        self.archive.sparsity_limit(n_limit=self.n_swallows)

        self.swarm_update_velocity()
        self.swarm_move()

        self.history.write_history()

        self.rep.log(
            'Iteration {}\t'
            'Archive Length = {:03}\t'
            ''.format(self.iteration,
                      len(self.archive.population))
        )

    def optimise(self):

        self.reset_environment()
        self.initialise_swarm()
        self.initialise_archive()

        while not self.termiation_manager.termination_check():
            self.step_optimise()
            self.iteration += 1

        self.rep.log('Optimisation complete...')
        self.rep.log('Archive contents:')

        for idx, swallow in enumerate(self.archive.population):
            self.rep.log('Swallow {:03}\t'
                         'Fitness = {}\t'
                         'Position = {}'
                         ''.format(idx, swallow.fitness, swallow.position))

    def plot_archive(self, save=False):

        if self.n_objs == 2:

            f1 = [swallow.fitness[0] for swallow in self.archive.population]
            f2 = [swallow.fitness[1] for swallow in self.archive.population]

            fig = plt.figure(figsize=(16, 10))
            plt.scatter(f1, f2, s=5)

            if save:
                plt.savefig('plot.png')
            else:
                plt.show()

        else:
            raise ValueError('Must be 2 objectives...')
