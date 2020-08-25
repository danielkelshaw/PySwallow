import multiprocessing as mp
from typing import Callable

from ..opt.sopso import Swarm
from ..swallows.so_swallow import Swallow


class MPSwarm(Swarm):

    def __init__(self,
                 bounds: dict,
                 n_swallows: int,
                 n_iterations: int,
                 cores: int,
                 w: float = 0.7,
                 c1: float = 2.0,
                 c2: float = 2.0,
                 debug: bool = False) -> None:

        """Multiprocessing Swarm.

        Parameters
        ----------
        bounds : dict
            Provides the upper and lower bounds of the search space.
        n_swallows : int
            Population size.
        n_iterations : int
            Number of iterations to run optimisation for.
        cores : int
            Number of cores to use for multiprocessing.
        w : float
            Inertia weight.
        c1 : float
            Cognitive weight.
        c2 : float
            Social weight.
        debug : bool
            True if you want to log debugging, False otherwise.
        """

        super().__init__(bounds, n_swallows, n_iterations, w, c1, c2, debug)

        self.cores = cores
        self.pool = mp.Pool(processes=self.cores)

    def step_optimise(self, fn: Callable[[Swallow], Swallow]) -> None:

        """Runs one iteration of the optimisation process.

        Parameters
        ----------
        fn : Callable[[Swallow], Swallow]
            Function to optimise for.
        """

        self.w = self.iwh(self.iteration)

        for swallow in self.population:
            swallow.swallow_iteration = self.iteration

        self.population = self.pool.map(fn, self.population)

        for swallow in self.population:

            if not self.constraints_manager.violates_position(swallow):
                self.gbest_update(swallow)
                self.pbest_update(swallow)

        for swallow in self.population:

            self.update_velocity(swallow)
            swallow.move(self.bh)

        self.history.write_history()

    def optimise(self, fn: Callable[[Swallow], Swallow]) -> None:

        """Runs the entire optimisation process.

        Parameters
        ----------
        fn : Callable[[Swallow], Swallow]
            Function to optimise for.
        """

        self.reset_environment()
        self.initialise_swarm()

        while not self.termination_manager.termination_check():
            self.step_optimise(fn)

            if self.checkpointer(self.iteration):
                self.save_swarm()

            self.iteration += 1

        self.rep.log('Optimisation complete...')
