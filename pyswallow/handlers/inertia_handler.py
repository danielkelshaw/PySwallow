class StandardIWH:

    def __init__(self, w):
        self.w = w

    def __call__(self, iteration):
        return self.w


class LinearIWH:

    def __init__(self, w_init, w_end, n_iterations):
        self.w_init = w_init
        self.w_end = w_end
        self.w_diff = self.w_init - self.w_end
        self.n_iterations = n_iterations

    def __call__(self, iteration):
        curr_w = self.w_init - self.w_diff * (iteration / self.n_iterations)
        return curr_w
