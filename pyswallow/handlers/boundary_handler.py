from .base_handler import BaseHandler
import numpy as np


class BoundaryHandler(BaseHandler):

    def __init__(self, strategy='standard'):

        self.strategy = strategy
        self.strategy_dict = self._get_all_strategies()

    def __call__(self, position, lb, ub, **kwargs):
        try:
            new_position = self.strategy_dict[self.strategy](
                position, lb, ub, **kwargs
            )
        except KeyError:
            raise KeyError('Invalid BoundaryHandler strategy.')
        else:
            return new_position

    @staticmethod
    def standard(position, lb, ub, **kwargs):
        return position

    @staticmethod
    def nearest(position, lb, ub, **kwargs):
        return np.clip(position, lb, ub)

    def reflective(self, position, lb, ub, **kwargs):
        ltb, gtb = self._out_of_bounds(position, lb, ub)

        while (ltb.shape[0] > 0) or (gtb.shape[0] > 0):

            if ltb.shape[0] > 0:
                position[ltb] = 2 * lb[ltb] - position[ltb]

            if gtb.shape[0] > 0:
                position[gtb] = 2 * ub[gtb] - position[gtb]

            ltb, gtb = self._out_of_bounds(position, lb, ub)

        return position

    def random(self, position, lb, ub, **kwargs):
        ltb, gtb = self._out_of_bounds(position, lb, ub)

        position[ltb] = np.random.uniform(lb[ltb], ub[ltb])
        position[gtb] = np.random.uniform(lb[gtb], ub[gtb])

        return position
