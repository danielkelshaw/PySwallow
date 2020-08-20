from .base_swallow import BaseSwallow
from ..handlers.boundary_handler import BaseBoundaryHandler


class Swallow(BaseSwallow):

    def __init__(self, bounds: dict) -> None:

        """Swallow Class.

        Parameters
        ----------
        bounds : dict
            Provides the upper and lower bounds of the search space.
        """

        super().__init__(bounds)

    def move(self, bh: BaseBoundaryHandler) -> None:

        """Responsible for moving the swallow in the search space.

        Parameters
        ----------
        bh : BaseHandler
            Alters the position according to the strategy.
        """

        self.position += self.velocity
        self.position = bh(self.position)
