import abc
from typing import Any, NoReturn

from ..swallows.base_swallow import BaseSwallow


class BaseConstraint(abc.ABC):

    @abc.abstractmethod
    def constrain(self, *args: Any) -> NoReturn:

        """Determines whether the associated arg violates constraints.

        Parameters
        ----------
        args : Any
            Arguments to determine if constraints are violated.

        Raises
        ------
        NotImplementedError
            Raises when this function has not yet been implemented.
        """

        raise NotImplementedError('BaseConstraint::constrained()')


class PositionConstraint(BaseConstraint):

    @abc.abstractmethod
    def constrain(self, swallow: BaseSwallow) -> NoReturn:

        """Determines whether the associated position violates constraints.

        Parameters
        ----------
        swallow : BaseSwallow
            Swallow to check constraints for.

        Raises
        ------
        NotImplementedError
            Raises when this function has not yet been implemented.
        """

        raise NotImplementedError('PositionConstraint::constrain()')


class FitnessConstraint(BaseConstraint):

    @abc.abstractmethod
    def constrain(self, swallow: BaseSwallow) -> NoReturn:

        """Determines whether thw associated fitness violates constraints.

        Parameters
        ----------
        swallow : BaseSwallow
            Swallow to check constraints for.

        Raises
        ------
        NotImplementedError
            Raises when this function has not yet been implemented.
        """

        raise NotImplementedError('FitnessConstraint::constrain()')
