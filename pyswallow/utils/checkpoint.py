from typing import Optional, Union


class Checkpointer:

    def __init__(self,
                 freq: Union[int, None] = None,
                 save_final: Union[bool, None] = False,
                 n_iterations: Optional[int] = None) -> None:

        """Checkpointer Class.

        Parameters
        ----------
        freq : Union[int, None]
            Checkpointing frequency.
        save_final : Union[bool, None]
            If True, the final iteration will be checkpointed.
        n_iterations : Optional[int]
            Number of iterations.
        """

        self.freq = freq
        self.save_final = save_final
        self.n_iterations = n_iterations

        if self.save_final and n_iterations is None:
            raise ValueError('If save_final=True, n_iterations must be defined.')

    def __call__(self, iteration: int) -> bool:

        if self.freq is None:
            return False

        if iteration % self.freq == 0:
            return True
        elif self.save_final and iteration == self.n_iterations:
            return True
        else:
            return False
