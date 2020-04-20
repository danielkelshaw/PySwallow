from .base_constraints import BaseConstraint, PositionConstraint


class ConstraintManager:

    def __init__(self, swarm):

        """
        Initialiser for ConstraintManager.

        Parameters
        ----------
        swarm : BaseSwarm
            Instance of BaseSwarm to be used.
        """

        self.swarm = swarm
        self.constraints = []

    def violates_position(self, swallow):

        """
        Checks if position constraints have been violated.

        Parameters
        ----------
        swallow : BaseSwallow
            The swallow for which to check the constraints.

        Returns
        -------
        bool
            True if position constraints are violated, False otherwise.
        """

        position = dict(zip(self.swarm.pnames, swallow.position))

        within_constraints = True
        for constraint in self.constraints:
            if isinstance(constraint, PositionConstraint):
                within_constraints = constraint.constrain(position)

            if not within_constraints:
                return True

        return False

    def register_constraint(self, constraint):

        """
        Adds a constraint to be tested.

        Parameters
        ----------
        constraint : BaseConstraint
            The constraint to check.
        """

        if not isinstance(constraint, BaseConstraint):
            raise TypeError('constraint must inherit from BaseConstraint')

        self.constraints.append(constraint)
