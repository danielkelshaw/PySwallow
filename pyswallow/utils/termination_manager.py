import abc
import time

from ..opt.base_swarm import BaseSwarm
from ..opt.sopso import Swarm

from typing import NoReturn


class BaseTerminationManager(abc.ABC):

    @abc.abstractmethod
    def termination_check(self) -> NoReturn:

        """Checks if optimisation process is complete."""

        raise NotImplementedError(
            'BaseTerminationManager::termination_check()'
        )


class IterationTerminationManager(BaseTerminationManager):

    """Terminates optimisation process after n_iterations."""

    def __init__(self, swarm: BaseSwarm) -> None:

        """Iteration Termination Manager Class.

        Parameters
        ----------
        swarm : BaseSwarm
            Swarm to manage.
        """

        self.swarm = swarm

    def termination_check(self) -> bool:

        if self.swarm.iteration > self.swarm.n_iterations:
            return True
        else:
            return False


class TimeTerminationManager(BaseTerminationManager):

    """Terminates optimisation process after N seconds."""

    def __init__(self, t_budget: int) -> None:

        """Time Termination Manager Class.

        Parameters
        ----------
        t_budget : int
            Number of seconds to terminate after.
        """

        self.t_budget = t_budget
        self.t_start = None

    def termination_check(self) -> bool:

        if self.t_start is None:
            self.t_start = time.time()

        t_elapsed = time.time() - self.t_start

        if t_elapsed > self.t_budget:
            return True
        else:
            return False


class EvaluationTerminationManager(BaseTerminationManager):

    """Terminates optimisation process after N function evaluations."""

    def __init__(self, swarm: BaseSwarm, n_evaluations: int) -> None:

        """Evaluation Termination Manager Class.

        Parameters
        ----------
        swarm : BaseSwarm
            Swarm to manage.
        n_evaluations : int
            Total number of function evaluations allowed.
        """

        self.swarm = swarm
        self.n_iterations = n_evaluations // swarm.n_swallows

    def termination_check(self) -> bool:

        if self.swarm.iteration > self.n_iterations:
            return True
        else:
            return False


class ErrorTerminationManager(BaseTerminationManager):

    """Terminates optimisation process if target is reached."""

    def __init__(self, swarm: Swarm, target: float, threshold: float) -> None:

        """Error Termination Manager Class.

        Parameters
        ----------
        swarm : Swarm
            Swarm to manage.
        target : float
            Optimisation target to reach before termination.
        threshold : float
            Threshold within which to consider the target reached.
        """

        self.swarm = swarm
        self.target = target
        self.threshold = threshold

    def termination_check(self) -> bool:

        if self.swarm.gbest_swallow is None:
            return False
        elif self._in_threshold(self.swarm.gbest_swallow.fitness):
            return True
        else:
            return False

    def _in_threshold(self, val: float) -> bool:

        """Helper method to determine if the value has reached the target.

        Parameters
        ----------
        val : float
            Value to evaluate as having the reached the target or not.

        Returns
        -------
        bool
            True if val within the target threshold, False otherwise.
        """

        return self.target - self.threshold < val < self.target + self.threshold
