from .base_handler import BaseHandler
import numpy as np


class StandardBH(BaseHandler):

    def __init__(self):
        super().__init__()

    def __call__(self, position):
        return position


class NearestBH(BaseHandler):

    def __init__(self, lb, ub):
        super().__init__()
        self.lb = lb
        self.ub = ub

    def __call__(self, position):
        return np.clip(position, self.lb, self.ub)


class ReflectiveBH(BaseHandler):

    def __init__(self, lb, ub):
        super().__init__()
        self.lb = lb
        self.ub = ub

    def __call__(self, position):
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
        super().__init__()
        self.lb = lb
        self.ub = ub

    def __call__(self, position):
        ltb, gtb = self._out_of_bounds(position, self.lb, self.ub)

        position[ltb] = np.random.uniform(self.lb[ltb], self.ub[ltb])
        position[gtb] = np.random.uniform(self.lb[gtb], self.ub[gtb])

        return position
