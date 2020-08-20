from typing import Tuple

import numpy as np


class BaseHandler:

    @staticmethod
    def _out_of_bounds(vector: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """
        Determines if a vector is out of imposed bounds.

        Parameters
        ----------
        vector : np.ndarray
            Vector to see if within bounds.
        lb : np.ndarray
            Lower bound.
        ub : np.ndarray
            Upper bound.

        Returns
        -------
        ltb : np.ndarray
            Boolean arary determining whether vector < lb.
        gtb : np.ndarray
            Boolean arary determining whether vector > ub.
        """

        ltb = np.nonzero(vector < lb)[0]
        gtb = np.nonzero(vector > ub)[0]

        return ltb, gtb
