import pyswallow as ps
import numpy as np


def obj_one(position):
    return np.square(position[0])


def obj_two(position):
    return np.square(position[0] - 2)


objectives = [obj_one, obj_two]

print('PySwallow: Example MOPSO')

swallows = 30
iterations = 100

lbound = [-500]
ubound = [500]

swarm = ps.MOSwarmArchive(obj_functions=objectives,
                          n_swallows=swallows,
                          n_iterations=iterations,
                          lb=lbound,
                          ub=ubound,
                          constraints=None,
                          w=0.7,
                          c1=2.0,
                          c2=2.0,
                          debug=False)

swarm.optimise()

swarm.plot_archive()
