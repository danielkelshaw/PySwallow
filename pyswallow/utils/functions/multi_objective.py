import numpy as np


def binh_korn():
    def func_one(position):
        if not position.shape[0] == 2:
            raise IndexError('Binh & Korn only takes two-dimensional inputs.')
        if not np.logical_and(position[0] >= 0, position[0] <= 5):
            raise ValueError(
                'x-input for Binh & Korn function must be within [0, 5].'
            )
        if not np.logical_and(position[1] >= 0, position[1] <= 3):
            raise ValueError(
                'y-input for Binh & Korn function must be within [0, 3].'
            )

        x = position[0]
        y = position[1]

        return 4 * np.square(x) + 4 * np.square(y)

    def func_two(position):
        if not position.shape[0] == 2:
            raise IndexError('Binh & Korn only takes two-dimensional inputs.')
        if not np.logical_and(position[0] >= 0, position[0] <= 5):
            raise ValueError(
                'x-input for Binh & Korn function must be within [0, 5].'
            )
        if not np.logical_and(position[1] >= 0, position[1] <= 3):
            raise ValueError(
                'y-input for Binh & Korn function must be within [0, 3].'
            )

        x = position[0]
        y = position[1]

        return np.square(x - 5) + np.square(y - 5)

    def constraints(position):

        x = position[0]
        y = position[1]

        def con_one():
            return np.square(x - 5) + np.square(y)

        def con_two():
            return np.square(x - 8) + np.square(y + 3)

        if np.logical_and(con_one() <= 25,
                          con_two() >= 7.7):
            return True
        else:
            return False

    return [func_one, func_two], constraints


def schaffer_n1():
    def func_one(position):
        if not position.shape[0] == 1:
            raise IndexError('Schaffer N1 only takes one-dimensional input.')

        return np.square(position[0])

    def func_two(position):
        if not position.shape[0] == 1:
            raise IndexError('Schaffer N1 only takes one-dimensional input.')

        return np.square(position[0] - 2)

    return [func_one, func_two]
