import numpy as np


class BaseHandler:

    @staticmethod
    def _out_of_bounds(vector, lb, ub):
        ltb = np.nonzero(vector < lb)[0]
        gtb = np.nonzero(vector > ub)[0]
        return ltb, gtb
