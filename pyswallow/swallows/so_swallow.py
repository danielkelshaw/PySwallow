from .base_swallow import BaseSwallow


class Swallow(BaseSwallow):

    def __init__(self, bounds):

        """
        Initialiser for the Swallow class.

        Parameters
        ----------
        bounds : dict
            Provides the upper and lower bounds of the search space.
        """

        super().__init__(bounds)

    def move(self, bh):

        """
        Responsible for moving the swallow in the search space.

        Parameters
        ----------
        bh : BaseHandler
            Alters the position according to the strategy.

        Returns
        -------

        """

        self.position += self.velocity
        self.position = bh(self.position)
