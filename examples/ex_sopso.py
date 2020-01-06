import pyswallow as ps
import numpy as np


def objective_function(position):
    return np.square(position[0]) + np.square(position[1]) + 1


def applied_constraints(position):
    if np.logical_or((position[0] >= 0) and (position[1] <= 0),
                     (position[1] >= 0) and (position[0] <= 0)):
        return True
    else:
        return False


print('PySwallow: Example SOPSO')

swallows = 30
iterations = 1000

lbound = [-50, -50]
ubound = [50, 50]

swarm = ps.Swarm(obj_function=objective_function,
                 n_swallows=swallows,
                 n_iterations=iterations,
                 lb=lbound,
                 ub=ubound,
                 constraints=applied_constraints)

swarm.optimise()
