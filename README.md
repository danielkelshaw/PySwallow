# PySwallow

[![Build Status](https://travis-ci.org/danielkelshaw/PySwallow.svg?branch=master)](https://travis-ci.org/danielkelshaw/PySwallow)

**PySwallow is a library of Particle Swarm Optimisation algorithms implemented in Python.**

## Features

- [x] High-level module for optimisation.
- [x] Extensible tools for researchers and developers.
- [x] Built-in objective functions for testing.

*And many more...*

## Usage

```python
import pyswallow as ps
from pyswallow.utils.functions.single_objective import sphere

swallows = 30
iterations = 100

lbound = [-50, -50]
ubound = [50, 50]

swarm = ps.Swarm(obj_function=sphere,
                 n_swallows=swallows,
                 n_iterations=iterations,
                 lb=lbound,
                 ub=ubound)

swarm.optimise()
```
