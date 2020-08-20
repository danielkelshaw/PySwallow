import numpy as np

from .base_handler import BaseHandler


class StandardVH(BaseHandler):

    def __init__(self) -> None:

        """Standard Velocity Handler."""

        super().__init__()

    def __call__(self, velocity: np.ndarray) -> np.ndarray:

        """Returns the velocity completely unaltered.

        Parameters
        ----------
        velocity : np.ndarray
            Velocity to pass back.

        Returns
        -------
        np.ndarray
            Unaltered velocity.
        """

        return velocity


class ClampedVH(BaseHandler):

    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:

        """Clamped Velocity Handler.

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

    def __call__(self, velocity: np.ndarray) -> np.ndarray:

        """Clips the velocity according to the imposed bounds.

        Parameters
        ----------
        velocity : np.ndarrary
            Velocity to clip.

        Returns
        -------
        np.ndarray
            Clipped velocity.
        """

        return np.clip(velocity, self.lb, self.ub)


class InvertVH(BaseHandler):

    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:

        """Invert Velocity Handler.

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

    def __call__(self, velocity: np.ndarray, z: float = 0.5) -> np.ndarray:

        """Inverts velocity according to scaling factor, z, and bounds.

        Parameters
        ----------
        velocity : np.ndarray
            Velocity to invert.
        z : float
            Inversion scaling factor.

        Returns
        -------
        np.ndarray
            Inverted velocity vector.
        """

        ltb, gtb = self._out_of_bounds(velocity, self.lb, self.ub)

        velocity[ltb] = (-z) * velocity[ltb]
        velocity[gtb] = (-z) * velocity[gtb]

        return velocity


class ZeroVH(BaseHandler):

    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:

        """Zero Velocity Handler.

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

    def __call__(self, velocity: np.ndarray) -> np.ndarray:

        """Zeros the velocty for any dimension exceeding the bounds.

        Parameters
        ----------
        velocity : np.ndarray
            Velocity to zero.

        Returns
        -------
        np.ndarray
            Velocity with values zeroed if outside bounds.
        """

        ltb, gtb = self._out_of_bounds(velocity, self.lb, self.ub)

        velocity[np.concatenate((ltb, gtb))] = 0.0
        return velocity
