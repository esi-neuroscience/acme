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

from scipy import signal
import tempfile
import h5py
# def lowpass(arr, b, a=None, padlen=200):
#     res = signal.filtfilt(b, a, arr.T, padlen=padlen).T
#     return res

def lowpass_h5dset(arr_like, b, a, channel_no, padlen=200):
    channel = arr_like[:, channel_no]
    res = signal.filtfilt(b, a, channel, padlen=padlen)
    return res

def lowpass_simple(h5name, channel_no):
    with h5py.File(h5name, "r") as h5f:
        channel = h5f["data"][:, channel_no]
        b = h5f["data"].attrs["b"]
        a = h5f["data"].attrs["a"]
    res = signal.filtfilt(b, a, channel, padlen=200)
    return res

# Prepare code to be executed using, e.g., iPython's `%run` magic command
if __name__ == "__main__":

    # Construct linear combination of low- and high-frequency sine waves
    # and use an IIR filter to reconstruct the low-frequency component
    nChannels = 32
    nTrials = 8
    fData = 2
    fNoise = 64
    fs = 1000
    t = np.linspace(-1, 1, fs)
    orig = np.sin(2 * np.pi * fData * t)
    sig = orig + np.sin(2 * np.pi * fNoise * t)
    cutoff = 50
    b, a = signal.butter(8, 2 * cutoff / fs)

    # Blow up the signal to have "channels" and "trials" and inflate the low-
    # frequency component accordingly for ad-hoc comparisons later
    sig = np.repeat(sig.reshape(-1, 1), axis=1, repeats=nChannels)
    sig = np.tile(sig, (nTrials, 1))
    orig = np.repeat(orig.reshape(-1, 1), axis=1, repeats=nChannels)
    orig = np.tile(orig, (nTrials, 1))

    with tempfile.TemporaryDirectory() as tdir:
        sigName = os.path.join(tdir, "sigdata.h5")
        origName = os.path.join(tdir, "origdata.h5")

        with h5py.File(sigName, "w") as sigFile:
            dset = sigFile.create_dataset("data", data=sig)
            dset.attrs["b"] = b
            dset.attrs["a"] = a

        with h5py.File(origName, "w") as origFile:
            origFile.create_dataset("data", data=sig)

        with ParallelMap(lowpass_simple, sigName, range(nChannels)) as pmap:
            pmap.compute()

        sigData = h5py.File(sigName, "r")["data"]
        with ParallelMap(lowpass_h5dset, sigData, b, a, range(nChannels), n_inputs=nChannels) as pmap:
            pmap.compute()

        # Close any open HDF5 files to not trigger any `WinError`s
        sigData.file.close()

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

