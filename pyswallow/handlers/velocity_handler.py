from .base_handler import BaseHandler
import numpy as np


class VelocityHandler(BaseHandler):

    def __init__(self, strategy='standard'):

        self.strategy = strategy
        self.strategy_dict = self._get_all_strategies()

    def __call__(self, velocity, lb, ub, **kwargs):
        try:
            new_velocity = self.strategy_dict[self.strategy](
                velocity, lb, ub, **kwargs
            )
        except KeyError:
            raise KeyError('Invalid VelocityHandler strategy.')
        else:
            return new_velocity

    @staticmethod
    def standard(velocity, lb, ub, **kwargs):
        return velocity

    @staticmethod
    def clamped(velocity, lb, ub, **kwargs):
        return np.clip(velocity, lb, ub)

    def invert(self, velocity, lb, ub, **kwargs):
        ltb, gtb = self._out_of_bounds(kwargs['position'], lb, ub)

        if 'z' not in kwargs:
            z = 0.5
        else:
            z = kwargs['z']

        velocity[ltb] = (-z) * velocity[ltb]
        velocity[gtb] = (-z) * velocity[gtb]

        return velocity

    def zero(self, velocity, lb, ub, **kwargs):
        ltb, gtb = self._out_of_bounds(kwargs['position'], lb, ub)

        velocity[np.concatenate((ltb, gtb))] = 0.0
        return velocity
