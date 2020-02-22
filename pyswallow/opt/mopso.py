from ..base.base_swarm import BaseSwarm
from ..base.base_swallow import BaseSwallow
from ..utils.reporter import Reporter
from ..handlers.boundary_handler import StandardBH
from ..handlers.velocity_handler import StandardVH
from ..handlers.inertia_handler import StandardIWH

import numpy as np
import logging
import copy


class MOSwallow(BaseSwallow):

    def __init__(self, lb, ub, n_obj):

        super().__init__(lb, ub)
        self.n_obj = n_obj

        self.fitness = [None] * n_obj
        self.pbest_fitness = [float('inf')] * n_obj

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

    def __init__(self, obj_functions, n_swallows, n_iterations, lb, ub,
                 constraints=None, w=0.7, c1=2.0, c2=2.0, debug=False):

        super().__init__(n_swallows, lb, ub, w, c1, c2)

        log_debug = logging.DEBUG if debug else logging.INFO
        self.rep = Reporter(lvl=log_debug)

        self.archive = []

        self.obj_functions = obj_functions
        self.n_objs = len(self.obj_functions)
        self.constraints = constraints

        self.iteration = 0
        self.n_iterations = n_iterations

        self.bh = StandardBH()
        self.vh = StandardVH()
        self.iwh = StandardIWH(self.w)

        self.rep.log('Swarm initialised successfully')

    # Reset methods
    def reset_environment(self):
        self.iteration = 0
        self.reset_populations()
        self.rep.log('Environment reset', lvl=logging.DEBUG)

    def reset_populations(self):
        self.population = []
        self.archive = []
        self.rep.log('Populations reset', lvl=logging.DEBUG)

    # Initialisation
    def initialise_swarm(self):
        self.population = [MOSwallow(self.lb, self.ub, self.n_objs)
                           for _ in range(self.n_swallows)]
        self.rep.log('Population initialised', lvl=logging.DEBUG)

    # Update Methods
    def evaluate_fitness(self, swallow):
        for idx, function in enumerate(self.obj_functions):
            swallow.fitness[idx] = function(swallow.position)

    def swarm_evaluate_fitness(self):
        for swallow in self.population:
            self.evaluate_fitness(swallow)

    def update_velocity(self, swallow):

        _leader = self.choose_leader()

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

    def choose_leader(self):
        return copy.deepcopy(np.random.choice(self.archive))

    def swarm_move(self):
        for swallow in self.population:
            swallow.move(self.bh)

    @staticmethod
    def update_pbest(swallow):
        if swallow.self_dominate():
            swallow.pbest_position = swallow.position
            swallow.pbest_fitness = swallow.fitness

    def swarm_update_pbest(self):
        for swallow in self.population:
            self.update_pbest(swallow)

    # Archive Based Functions
    @staticmethod
    def pareto_front(population):

        pf = []
        _population = copy.deepcopy(population)

        for idx, particle in enumerate(_population):

            pf.append(particle)

            for opp_idx, opp_particle in enumerate(pf[:-1]):
                if opp_particle.dominate(particle):
                    del pf[-1]
                    break
                elif particle.dominate(opp_particle):
                    del pf[opp_idx]

        return pf

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

            print('Iteration: {0}: Archive Length: {1:05}'
                  ''.format(self.iteration, len(self.archive)))

            self.w = self.iwh(self.iteration)

            self.swarm_evaluate_fitness()
            self.swarm_update_pbest()

            self.archive.extend(self.population)
            self.archive = self.pareto_front(self.archive)

            self.swarm_update_velocity()
            self.swarm_move()

            self.rep.log(
                'Iteration {}\t'
                'Archive Length = {:03}\t'
                ''.format(self.iteration,
                          len(self.archive))
            )

            self.iteration += 1

        self.rep.log('Optimisation complete...')
        self.rep.log('Archive contents:')

        for idx, swallow in enumerate(self.archive):
            self.rep.log('Swallow {:03}\t'
                         'Fitness = {}\t'
                         'Position = {}'
                         ''.format(idx, swallow.fitness, swallow.position))
