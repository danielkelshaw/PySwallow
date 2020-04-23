# PySwallow

[![Build Status](https://travis-ci.org/danielkelshaw/PySwallow.svg?branch=master)](https://travis-ci.org/danielkelshaw/PySwallow)

PySwallow is an extensible toolkit for PSO.

The library aims to provide a high-level declarative interface which
ensures that PSOs can be implemented and customised with ease. PySwallow 
features an extensible framework which allows researchers to provide 
custom implementations which interface with existing functionality.

- **License:** MIT
- **Python Versions:** 3.6+

## **Features:**
- [x] High-level module for Particle Swarm Optimisation.
- [x] Extensible API for implementing new functionality.

## **Basic Usage:**
PySwallow aims to provide a high-level interface for PSO - the code 
below demonstrates just how easy running an optimisation procedure
can be.

```python
import pyswallow as ps
from pyswallow.utils.functions import single_objective as fx


bounds = {
    'x0': [-1e6, 1e6],
    'x1': [-1e6, 1e6],
    'x2': [-1e6, 1e6]
}

optimiser = ps.Swarm(bounds=bounds, n_swallows=30, n_iterations=100)
optimiser.optimise(fx.sphere)
```

## **History:**
The optimisation history is written to a ```History``` data structure
to allow the user to further investigate the optimisation procedure 
upon completion. This is a powerful tool, letting the user define custom
history classes which can record whichever data the user desires.

Tracking the history of the optimisation process allows for plotting
of the results, an example demonstration is seen in the
```plot_fitness_history``` function - this can be further customised
through the designation of a ```PlotDesigner``` object which provides
formatting instructions for the graphing tools.

## **Constraints:**
PySwallow allows the user to define a set of constraints for the 
optimisation problem - this is achieved through inheriting a template 
class and implementing the designated method. An example of which is 
demonstrated below:

```python
from pyswallow.constraints.base_constraints import PositionConstraint


class UserConstraint(PositionConstraint):

    def constrain(self, position):
        return position['x0'] > 0 and position['x1'] < 0


optimiser.constraint_manager.register_constraint(UserConstraint())
```

This provides the user with a large amount of freedom to define the
appropriate constraints and allows the `ConstraintManager` to deal with
the relevant constraints at the appropriate time.

## **Customisation:**
Though the base `Swarm` is very effective, there may be aspects that the
user wishes to change, such as the boundary handler / inertia weight
methods. The library provides an extensible API which allows the user
to implement a variety of functions as well as develop their own with
templates provided in the form of *Abstract Base Classes*.

Attributes of the `Swarm` instance can be modified to alter how the
optimisation process will work, this is demonstrated below:

```python
# altering the boundary handling method
from pyswallow.handlers.boundary_handler import NearestBH
optimiser.bh = NearestBH(lb, ub)
```
```python
# altering the inertia weight handler
from pyswallow.handlers.inertia_handler import LinearIWH
optimiser.iwh = LinearIWH(w_init=0.7, w_end=0.4, n_iterations=100)
```

It is also possible to define alternative termination criteria through
implementation of a ```TerminationManager``` class, a couple of examples
are demonstrated below:

```python
# using elapsed time as the termination criteria
from pyswallow.utils.termination_manager import TimeTerminationManager
optimiser.termination_manager = TimeTerminationManager(t_budget=10_000)
```

```python
# using error as the termination criteria
from pyswallow.utils.termination_manager import ErrorTerminationManager
optimiser.termination_manager = ErrorTerminationManager(
    optimiser, target=0.0, threshold=1e-3
)
```

###### Author: Daniel Kelshaw
