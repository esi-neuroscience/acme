# -*- coding: utf-8 -*-
#
# Simple script for testing acme w/o pip-installing it
#

# Builtin/3rd party package imports
import numpy as np

# Add acme to Python search path
import os
import sys
acme_path = os.path.abspath(".." + os.sep + "..")
if acme_path not in sys.path:
    sys.path.insert(0, acme_path)

# Import package
from acme import ParallelMap

def simple_func(x, y, z=3):
    return (x + y) * z

def medium_func(x, y, z=3, w=np.ones((3, 3))):
    return (sum(x) + y) * z * w.max()

def hard_func(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return (sum(x) + y) * z * w.max()


# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    sys.exit()

    # Test stuff within here...
    pmap = ParallelMap(simple_func, [2, 4, 6, 8], 4)
    pmap = ParallelMap(simple_func, 0, 4, z=[3, 4, 5, 6])
    pmap = ParallelMap(simple_func, [2, 4, 6, 8], [2, 2], n_inputs=2)

    pmap = ParallelMap(medium_func, [2, 4, 6, 8], [2, 2], n_inputs=2)
    pmap = ParallelMap(medium_func, None, None, w=[np.ones((3, 3)), 2 * np.ones((3,3))])
    pmap = ParallelMap(medium_func, None, None, z=np.zeros((3,)))
    pmap = ParallelMap(medium_func, None, None, z=np.zeros((3, 1)))

    pmap = ParallelMap(hard_func, [2, 4, 6, 8], 2, w=np.ones((3,)))
    pmap = ParallelMap(hard_func, [2, 4, 6, 8], 2, w=np.ones((7, 1)))
    pmap = ParallelMap(hard_func, np.ones((3,)), 1, w=np.ones((7, 1)))
    pmap = ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], z=np.array([1, 2]), w=np.ones((8, 1)), n_inputs=2)
    pmap = ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], w=np.ones((8, 1)), n_inputs=4)

    # # Errors
    # pmap = ParallelMap(simple_func, 4)
    # pmap = ParallelMap(simple_func, 4, 4, z=3, w=4)
    # pmap = ParallelMap(simple_func, [2, 4, 6, 8], [2, 2])
    # pmap = ParallelMap(simple_func, [2, 4, 6, 8], [2, 2], n_inputs=3)
    # pmap = ParallelMap(medium_func, [1, 2, 3], None, w=[np.ones((3,3)), 2 * np.ones((3,3))])
    # pmap = ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], w=np.ones((8, 1)), n_inputs=8)

