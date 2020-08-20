from ..opt.sopso import Swarm
from ..swallows.so_swallow import Swallow

from typing import Any, Callable
import multiprocessing as mp


class MPSwarm(Swarm):

    def __init__(self, bounds, n_swallows, n_iterations, cores, w=0.7, c1=2.0, c2=2.0, debug=False) -> None:

        super().__init__(bounds, n_swallows, n_iterations, w, c1, c2, debug)

        self.cores = cores
        self.pool = mp.Pool(processes=self.cores)

    def step_optimise(self, fn: Callable[[Swallow], Swallow]) -> None:

        self.w = self.iwh(self.iteration)

        self.population = self.pool.map(fn, self.population)

        for swallow in self.population:

            if not self.constraints_manager.violates_position(swallow):
                self.gbest_update(swallow)
                self.pbest_update(swallow)

        for swallow in self.population:

            self.update_velocity(swallow)
            swallow.move(self.bh)

        self.history.write_history()
