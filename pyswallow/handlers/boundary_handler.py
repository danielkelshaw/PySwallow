import numpy as np

from .base_handler import BaseHandler


class BaseBoundaryHandler(BaseHandler):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: np.ndarray) -> np.ndarray:
        raise NotImplementedError('BaseBoundaryHandler::__call__()')


class StandardBH(BaseBoundaryHandler):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, position: np.ndarray) -> np.ndarray:

        """Returns the position unchanged.

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


class NearestBH(BaseBoundaryHandler):

    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:

        """Nearest Boundary Handler.

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

    def __call__(self, position: np.ndarray) -> np.ndarray:

        """Clips the position according to the imposed bounds.

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


class ReflectiveBH(BaseBoundaryHandler):

    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:

        """Reflective Boundary Handler.

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

    def __call__(self, position: np.ndarray) -> np.ndarray:

        """Reflects position back within the imposed bounds.

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


class RandomBH(BaseBoundaryHandler):

    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:

        """Random Boundary Handler.

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

    def __call__(self, position: np.ndarray) -> np.ndarray:

        """Returns random position within range for exceeded boundaries.

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
