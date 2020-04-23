import numpy as np

from .base_handler import BaseHandler


class StandardBH(BaseHandler):

    def __init__(self):
        super().__init__()

    def __call__(self, position):

        """
        Returns the position unchanged.

        Parameters
        ----------
        position : np.ndarray
            Position to return.

        Returns
        -------
        np.ndarray
            Original position unchanged.
        """

        return position


class NearestBH(BaseHandler):

    def __init__(self, lb, ub):
        super().__init__()
        self.lb = lb
        self.ub = ub

    def __call__(self, position):

        """
        Clips the position according to the imposed bounds.

        Parameters
        ----------
        position : np.ndarray
            Position to clip.

        Returns
        -------
        np.ndarray
            Clipped position.
        """

        return np.clip(position, self.lb, self.ub)


class ReflectiveBH(BaseHandler):

    def __init__(self, lb, ub):

        """
        Initialiser for the ReflectiveBH class.

        Parameters
        ----------
        lb : np.ndarray
            Lower bound.
        ub : np.ndarray
            Upper bound.
        """

        super().__init__()
        self.lb = lb
        self.ub = ub

    def __call__(self, position):

        """
        Reflects position back within the imposed bounds.

        Parameters
        ----------
        position : np.ndarray
            Position to reflect within bounds.

        Returns
        -------
        np.ndarray
            Position after being reflected within the bounds.
        """

        ltb, gtb = self._out_of_bounds(position, self.lb, self.ub)

        while (ltb.shape[0] > 0) or (gtb.shape[0] > 0):

            if ltb.shape[0] > 0:
                position[ltb] = 2 * self.lb[ltb] - position[ltb]

            if gtb.shape[0] > 0:
                position[gtb] = 2 * self.ub[gtb] - position[gtb]

            ltb, gtb = self._out_of_bounds(position, self.lb, self.ub)

        return position


class RandomBH(BaseHandler):

    def __init__(self, lb, ub):

        """
        Initialiser for the RandomBH class.

        Parameters
        ----------
        lb : np.ndarray
            Lower bound.
        ub : np.ndarray
            Upper bound.
        """

        super().__init__()
        self.lb = lb
        self.ub = ub

    def __call__(self, position):

        """
        Returns random position within range for exceeded boundaries.

        Parameters
        ----------
        position : np.ndarray
            Position to alter according to bounds.

        Returns
        -------
        np.ndarray
            Altered position.
        """

        ltb, gtb = self._out_of_bounds(position, self.lb, self.ub)

        position[ltb] = np.random.uniform(self.lb[ltb], self.ub[ltb])
        position[gtb] = np.random.uniform(self.lb[gtb], self.ub[gtb])

        return position
