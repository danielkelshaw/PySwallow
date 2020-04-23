import abc
import time


class BaseTerminationManager(abc.ABC):

    @abc.abstractmethod
    def termination_check(self):

        """
        Checks if optimisation process is complete.

        Returns
        -------
        bool
            True if complete, False if incomplete.
        """

        raise NotImplementedError(
            'BaseTerminationManager::termination_check()'
        )


class IterationTerminationManager(BaseTerminationManager):

    """Terminates optimisation process after n_iterations."""

    def __init__(self, swarm):

        """
        Initialises IterationTerminationManager.

        Parameters
        ----------
        swarm : BaseSwarm
            Swarm to manage.
        """

        self.swarm = swarm

    def termination_check(self):
        if self.swarm.iteration > self.swarm.n_iterations:
            return True
        else:
            return False


class TimeTerminationManager(BaseTerminationManager):

    """Terminates optimisation process after N seconds."""

    def __init__(self, t_budget):

        """
        Initialises TimeTerminationManager.

        Parameters
        ----------
        t_budget : int
            Number of seconds to terminate after.

        Attributes
        ----------
        t_start : float
            Time that the optimisation procedure started.
        """

        self.t_budget = t_budget
        self.t_start = None

    def termination_check(self):
        if self.t_start is None:
            self.t_start = time.time()

        t_elapsed = time.time() - self.t_start

        if t_elapsed > self.t_budget:
            return True
        else:
            return False


class EvaluationTerminationManager(BaseTerminationManager):

    """Terminates optimisation process after N function evaluations."""

    def __init__(self, swarm, n_evaluations):

        """
        Initialises EvaluationTerminationManager.

        Parameters
        ----------
        swarm : BaseSwarm
            Swarm to manage.
        n_evaluations : int
            Total number of function evaluations allowed.
        """

        self.swarm = swarm
        self.n_iterations = n_evaluations // swarm.n_swallows

    def termination_check(self):
        if self.swarm.iteration > self.n_iterations:
            return True
        else:
            return False


class ErrorTerminationManager(BaseTerminationManager):

    """Terminates optimisation process if target is reached."""

    def __init__(self, swarm, target, threshold):

        """
        Initialises ErrorTerminationManager.

        Parameters
        ----------
        swarm : BaseSwarm
            Swarm to manage.
        target : float
            Optimisation target to reach before termination.
        threshold : float
            Threshold within which to consider the target reached.
        """

        self.swarm = swarm
        self.target = target
        self.threshold = threshold

    def termination_check(self):

        if self.swarm.best_individual is None:
            return False
        elif self._in_threshold(self.swarm.best_individual.fitness):
            return True
        else:
            return False

    def _in_threshold(self, val):

        """
        Helper method to determine if the value has reached the target.

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
