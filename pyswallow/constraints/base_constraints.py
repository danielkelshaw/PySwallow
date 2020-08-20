import abc
from typing import NoReturn


class BaseConstraint(abc.ABC):

    @abc.abstractmethod
    def constrain(self, *args: dict) -> NoReturn:

        """Determines whether the associated arg violates constraints.

        Parameters
        ----------
        args : dict
            Arguments to determine if constraints are violated.

        Raises
        ------
        NotImplementedError
            Raises when this function has not yet been implemented.
        """

        raise NotImplementedError('BaseConstraint::constrained()')


class PositionConstraint(BaseConstraint):

    @abc.abstractmethod
    def constrain(self, position: dict) -> NoReturn:

        """Determines whether the associated position violates constraints.

        Parameters
        ----------
        position : dict
            Current position of the Swallow in dictionary form.

        Raises
        ------
        NotImplementedError
            Raises when this function has not yet been implemented.
        """

        raise NotImplementedError('PositionConstraint::constrained()')
