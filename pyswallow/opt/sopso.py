import numpy as np
import copy


class Swallow:

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

        self.position = np.random.uniform(lb, ub, size=self.lb.shape[0])
        self.velocity = np.random.uniform(lb, ub, size=self.lb.shape[0])
        self.fitness = None

        self.pbest_position = self.position
        self.pbest_fitness = float('inf')

    def move(self):
        self.position += self.velocity


class Swarm:

    def __init__(self, obj_function, n_swallows, n_iterations,
                 lb, ub, constraints=None, w=0.7, c1=2.0, c2=2.0):

        self.obj_function = obj_function
        self.constraints = constraints

        self.n_swallows = n_swallows
        self.n_iterations = n_iterations

        self.lb = np.array(lb)
        self.ub = np.array(ub)
        assert self.lb.shape[0] == self.ub.shape[0]

        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.iteration = 0
        self.population = []

        self.gbest_position = np.random.uniform(lb, ub, size=self.lb.shape[0])
        self.gbest_fitness = float('inf')

    # Reset Methods
    def reset_environment(self):
        self.iteration = 0
        self.gbest_position = np.random.uniform(self.lb, self.ub,
                                                size=self.lb.shape[0])
        self.gbest_fitness = float('inf')
        self.reset_populations()

    def reset_populations(self):
        self.population = []

    # Initialisation
    def initialise_swarm(self):
        self.population = [Swallow(self.lb, self.ub)
                           for _ in range(self.n_swallows)]

    # Update Methods
    def eval_fitness(self, swallow):
        swallow.fitness = self.obj_function(swallow.position)

    def swarm_eval_fitness(self):
        for swallow in self.population:
            self.eval_fitness(swallow)

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

    def swarm_update_velocity(self):
        for swallow in self.population:
            self.update_velocity(swallow)

    def swarm_move(self):
        for swallow in self.population:
            swallow.move()

    def pbest_update(self, swallow):
        if self.constraints is not None:
            if self.constraints(swallow.position):
                if swallow.fitness < swallow.pbest_fitness:
                    swallow.pbest_fitness = swallow.fitness
                    swallow.pbest_position = swallow.position
        else:
            if swallow.fitness < swallow.pbest_fitness:
                swallow.pbest_fitness = swallow.fitness
                swallow.pbest_position = swallow.position

    def swarm_pbest_update(self):
        for swallow in self.population:
            self.pbest_update(swallow)

    def gbest_update(self):
        for swallow in self.population:
            if self.constraints is not None:
                if self.constraints(swallow.position):
                    if swallow.fitness < self.gbest_fitness:
                        self.gbest_fitness = copy.copy(swallow.fitness)
                        self.gbest_position = copy.copy(swallow.position)
            else:
                if swallow.fitness < self.gbest_fitness:
                    self.gbest_fitness = copy.copy(swallow.fitness)
                    self.gbest_position = copy.copy(swallow.position)

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

            print('Iteration {0}: {1}'.format(self.iteration, self.gbest_fitness))

            self.swarm_eval_fitness()
            self.swarm_pbest_update()
            self.gbest_update()

            self.swarm_update_velocity()
            self.swarm_move()

            self.iteration += 1
