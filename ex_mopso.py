import pyswallow as ps
from pyswallow.utils.functions.multi_objective import schaffer_n1


objectives = schaffer_n1()

print('PySwallow: Example MOPSO')

swallows = 30
iterations = 100

lbound = [-500]
ubound = [500]

swarm = ps.MOSwarm(obj_functions=objectives,
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
