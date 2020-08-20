class StandardIWH:

    def __init__(self, w: float) -> None:

        """Standard Inertia Weight Handler.

        Parameters
        ----------
        w : float
            Inertia weight.
        """

        self.w = w

    def __call__(self, iteration: int) -> float:

        """Returns the inertia weight unchanged.

        Parameters
        ----------
        iteration : int
            Current iteration of the optimisation process.

        Returns
        -------
        float
            Original inertia weight, unchanged.
        """

        return self.w


class LinearIWH:

    def __init__(self, w_init: float, w_end: float, n_iterations: int) -> None:

        """Linear Inertia Weight Handler.

        Parameters
        ----------
        w_init : float
            Inertia weight at initialisation.
        w_end : float
            Inertia weight at end of optimisation.
        n_iterations : int
            Number of iterations.
        """

        self.w_init = w_init
        self.w_end = w_end
        self.w_diff = self.w_init - self.w_end
        self.n_iterations = n_iterations

    def __call__(self, iteration: int) -> float:

        """Calculated linearly interpolated intertia weight.

        Parameters
        ----------
        iteration : int
            Iteration for which to calculate the inertia weight.

        Returns
        -------
        float
            Inertia weight interpolated according to iteration.
        """

        curr_w = self.w_init - self.w_diff * (iteration / self.n_iterations)
        return curr_w
