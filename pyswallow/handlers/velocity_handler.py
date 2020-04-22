import numpy as np

from .base_handler import BaseHandler


class StandardVH(BaseHandler):

    def __init__(self):
        super().__init__()

    def __call__(self, velocity):
        return velocity


class ClampedVH(BaseHandler):

    def __init__(self, lb, ub):
        super().__init__()
        self.lb = lb
        self.ub = ub

    def __call__(self, velocity):
        return np.clip(velocity, self.lb, self.ub)


class InvertVH(BaseHandler):

    def __init__(self, lb, ub):
        super().__init__()

        # TODO >> Need to make sure np array passed in has type(np.float32)
        self.lb = lb
        self.ub = ub

    def __call__(self, velocity, z=0.5):
        ltb, gtb = self._out_of_bounds(velocity, self.lb, self.ub)

        velocity[ltb] = (-z) * velocity[ltb]
        velocity[gtb] = (-z) * velocity[gtb]

        return velocity


class ZeroVH(BaseHandler):

    def __init__(self, lb, ub):
        super().__init__()
        self.lb = lb
        self.ub = ub

    def __call__(self, velocity, z=0.5):
        ltb, gtb = self._out_of_bounds(velocity, self.lb, self.ub)

        velocity[np.concatenate((ltb, gtb))] = 0.0
        return velocity
