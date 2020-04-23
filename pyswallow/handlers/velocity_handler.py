import numpy as np

from .base_handler import BaseHandler


class StandardVH(BaseHandler):

    def __init__(self):
        super().__init__()

    def __call__(self, velocity):

        """
        Returns the velocity completely unaltered.

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

    def __init__(self, lb, ub):

        """
        Initialiser for ClampedVH class.

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

    def __call__(self, velocity):

        """
        Clips the velocity according to the imposed bounds.

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

    def __init__(self, lb, ub):

        """
        Initialiser for InvertVH class.

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

    def __call__(self, velocity, z=0.5):

        """
        Inverts velocity according to scaling factor, z, and bounds.

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

    def __init__(self, lb, ub):

        """
        Initialier for ZeroVH class.

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

    def __call__(self, velocity):

        """
        Zeros the velocty for any dimension exceeding the bounds.

        Parameters
        ----------
        velocity : np.ndarray

        Returns
        -------
        np.ndarray
            Velocity with values zeroed if outside bounds.
        """

        ltb, gtb = self._out_of_bounds(velocity, self.lb, self.ub)

        velocity[np.concatenate((ltb, gtb))] = 0.0
        return velocity
