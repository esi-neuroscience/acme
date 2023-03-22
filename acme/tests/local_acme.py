#
# Simple script for testing acme w/o pip-installing it
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
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
    return (x + y) * z * w.max()

import time

def g(x, y, z=3):
    fSize = np.dtype("float").itemsize
    arrSize = 4
    time.sleep(10)
    arr = np.ones((int(arrSize * 1024**3 / fSize), ))
    time.sleep(300)
    return (sum(x) + y) * z * arr.max()

def arr_test(x, y):
    return x + y

# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    # Test stuff within here...
    # pass

    with ParallelMap(arr_test, [np.ones((20,)), 2 * np.ones((20,)), 3 * np.ones((20,))], 4, result_shape=(None, 20), verbose=True, logfile=True, single_file=True) as pmap:
        results = pmap.compute()

    1/0

    # pmap = ParallelMap(g, np.arange(100), 2)
    # pmap.daemon.estimate_memuse()

    # # with pmap as p:
    # #     p.compute()


