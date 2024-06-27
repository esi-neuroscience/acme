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
import h5py

# Add acme to Python search path
import os
import sys
acme_path = os.path.abspath(".." + os.sep + "..")
if acme_path not in sys.path:
    sys.path.insert(0, acme_path)

# Import package
from acme import ParallelMap, esi_cluster_setup



def f(x, y, z=3):
    return (x + y) * z

# import time

# def g(x, y, z=3):
#     fSize = np.dtype("float").itemsize
#     arrSize = 4
#     time.sleep(10)
#     arr = np.ones((int(arrSize * 1024**3 / fSize), ))
#     time.sleep(300)
#     return (sum(x) + y) * z * arr.max()

# def arr_test(x, y):
#     return x + y

# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    # Test stuff within here...
    client = esi_cluster_setup(partition="8GBXS", n_workers=1)
    # with ParallelMap(f, [elem * np.ones((3,)) for elem in [2, 4, 6, 8]], 4, result_shape=(None, np.inf), verbose=True, single_file=True) as pmap:
    #     pmap.compute()

    with ParallelMap(f, [elem * np.ones((3, 3)) for elem in [2, 4, 6, 8]], 4, result_shape=(np.inf, None, 3), verbose=True, single_file=True, logfile="asdf.txt") as pmap:
        pmap.compute(debug=True)

    h5f = h5py.File(pmap.results_container, "r")
    dset = h5f["result_0"]

    payloadDir = pmap.results_container.replace(".h5", "_payload")
    ff = h5py.File(f"{payloadDir}/f_0.h5", "r")
    subset = ff["result_0"]


#     with ParallelMap(arr_test, [np.ones((20,)), 2 * np.ones((20,)), 3 * np.ones((20,))], 4, result_shape=(None, 20), verbose=True, single_file=True) as pmap:
#         results = pmap.compute()

#     1/0

#     # pmap = ParallelMap(g, np.arange(100), 2)
#     # pmap.daemon.estimate_memuse()

#     # # with pmap as p:
#     # #     p.compute()


