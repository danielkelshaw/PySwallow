import inspect
import numpy as np


class BaseHandler:

    def _get_all_strategies(self):
        return {
            k: v
            for k, v in inspect.getmembers(self, predicate=inspect.isroutine)
            if not k.startswith(('_', '__'))
        }

    @staticmethod
    def _out_of_bounds(vector, lb, ub):
        ltb = np.nonzero(vector < lb)[0]
        gtb = np.nonzero(vector > ub)[0]
        return ltb, gtb
