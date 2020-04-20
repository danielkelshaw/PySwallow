import numpy as np


def ackley(position):

    """
    Implementation of the Ackley Function.

    Bounds  = [-32.0, 32.0] for N dimensions.
    Optimum = 0.0 at position = [0.0, ..., 0.0]

    Parameters
    ----------
    position : np.ndarray
        Position at which to evaluate the function.

    Returns
    -------
    val : float
        Function evaluation, f(position)
    """

    if not np.logical_and(position >= -32, position <= 32).all():
        raise ValueError('Input for Ackley function must be within [-32, 32].')

    dims = position.shape[0]
    val = (-20.0 * np.exp(-0.2 * np.sqrt((1 / dims)
                                         * (position ** 2).sum(axis=0)))
           - np.exp((1 / float(dims))
                    * np.cos(2 * np.pi * position).sum(axis=0))
           + 20.0
           + np.exp(1))

    return val


def beale(position):

    """
    Implementation of the Beale Function.

    Bounds  = [-4.5, 4.5] in 2D.
    Optimum = 0.0 at position = [3.0, 0.5]

    Parameters
    ----------
    position : np.ndarray
        Position at which to evaluate the function.

    Returns
    -------
    val : float
        Function evaluation, f(position)
    """

    if not position.shape[0] == 2:
        raise IndexError('Beale function only takes two-dimensional input.')
    if not np.logical_and(position >= -4.5, position <= 4.5).all():
        raise ValueError('Input for Beale function must be within [-4.5, 4.5].')

    x = position[0]
    y = position[1]
    val = ((1.5 - x + x * y) ** 2.0
           + (2.25 - x + x * y ** 2.0) ** 2.0
           + (2.625 - x + x * y ** 3.0) ** 2.0)

    return val


def booth(position):

    """
    Implementation of the Booth Function.

    Bounds  = [-10.0, 10.0] in 2D.
    Optimum = 0.0 at position = [1.0, 3.0]

    Parameters
    ----------
    position : np.ndarray
        Position at which to evaluate the function.

    Returns
    -------
    val : float
        Function evaluation, f(position)
    """

    if not position.shape[0] == 2:
        raise IndexError('Booth function only takes two-dimensional input.')
    if not np.logical_and(position >= -10, position <= 10).all():
        raise ValueError('Input for Booth function must be within [-10, 10].')

    x = position[0]
    y = position[1]
    val = (x + 2 * y - 7) ** 2.0 + (2 * x + y - 5) ** 2.0

    return val


def goldsteinprice(position):

    """
    Implementation of the Goldenstein-Price Function.

    Bounds  = [-2.0, 2.0] in 2D.
    Optimum = 3.0 at position = [0.0, -1.0]

    Parameters
    ----------
    position : np.ndarray
        Position at which to evaluate the function.

    Returns
    -------
    val : float
        Function evaluation, f(position)
    """

    if not position.shape[0] == 2:
        raise IndexError('Goldstein function only takes two-dimensional input.')
    if not np.logical_and(position >= -2, position <= 2).all():
        raise ValueError('Input for Goldstein-Price '
                         'function must be within [-2, 2].')

    x = position[0]
    y = position[1]
    val = ((1
           + (x + y + 1) ** 2.0
           * (19
              - 14 * x
              + 3 * x ** 2.0
              - 14 * y
              + 6 * x * y
              + 3 * y ** 2.0
              ))
           * (30
              + (2 * x - 3 * y) ** 2.0
              * (18
                 - 32 * x
                 + 12 * x ** 2.0
                 + 48 * y
                 - 36 * x * y
                 + 27 * y ** 2.0
                 )
              ))

    return val


def rastrigin(position):

    """
    Implementation of the Rastrigin Function.

    Bounds  = [-5.12, 5.12] in N dimensions.
    Optimum = 0.0 at position = [0.0, ..., 0.0]

    Parameters
    ----------
    position : np.ndarray
        Position at which to evaluate the function.

    Returns
    -------
    val : float
        Function evaluation, f(position)
    """

    if not np.logical_and(position >= -5.12, position <= 5.12).all():
        raise ValueError('Input for Rastrigin function '
                         'must be within [-5.12, 5.12].')

    dims = position.shape[0]
    val = 10.0 * dims + (position ** 2.0
                         - 10.0 * np.cos(2.0 * np.pi * position)).sum(axis=0)

    return val


def sphere(position):

    """
    Implementation of the Sphere Function.

    Bounds  = [-inf, inf] in N dimensions.
    Optimum = 0.0 at position = [0.0, ..., 0.0]

    Parameters
    ----------
    position : np.ndarray
        Position at which to evaluate the function.

    Returns
    -------
    val : float
        Function evaluation, f(position)
    """

    val = np.sum(np.square(position))
    return val
