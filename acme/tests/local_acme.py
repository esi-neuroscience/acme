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

def f(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return (sum(x) + y) * z * w.max()


# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    # Test stuff within here...

    pmap = ParallelMap(f, [2, 4, 6, 8], [2, 2], z=np.array([1, 2]), w=np.ones((8, 1)), n_inputs=2)
    with pmap as p:
        p.compute()


