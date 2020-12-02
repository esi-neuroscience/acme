# -*- coding: utf-8 -*-
#
# Testing module for ACME's `ParallelMap` interface
#

# Builtin/3rd party package imports
import os
import h5py
import shutil
import pytest
import numpy as np
from scipy import signal

# Import main actor here
from acme import ParallelMap

# Functions that act as stand-ins for user-funcs
def simple_func(x, y, z=3):
    return (x + y) * z

def medium_func(x, y, z=3, w=np.ones((3, 3))):
    return (sum(x) + y) * z * w.max()

def hard_func(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return sum(x) + y,  z * w

def lowpass_hard(arr_like, b, a, channel_no, padlen=200):
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


# Main testing class
class TestParallelMap():

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

    # Blow up the signal to have "channels" and "trials": even/odd channels have
    # opposing periodicity; do the same to the low-freq component
    sig = np.repeat(sig.reshape(-1, 1), axis=1, repeats=nChannels)
    sig[:, ::2] *= -1
    sig = np.tile(sig, (nTrials, 1))
    orig = np.repeat(orig.reshape(-1, 1), axis=1, repeats=nChannels)
    orig[:, ::2] *= -1
    orig = np.tile(orig, (nTrials, 1))

    # Error tolerance for low-pass filtered results
    tol = 1e-3

    # Test setup of `ParallelMap` w/different functions args/kwargs
    def test_init(self):

        # Basic functionality w/simplest conceivable user-func
        pmap = ParallelMap(simple_func, [2, 4, 6, 8], 4)
        pmap = ParallelMap(simple_func, [2, 4, 6, 8], y=4)  # pos arg referenced via kwarg, cfg #2
        pmap = ParallelMap(simple_func, 0, 4, z=[3, 4, 5, 6])
        pmap = ParallelMap(simple_func, [2, 4, 6, 8], [2, 2], n_inputs=2)

        # User func has `np.ndarray` as keyword
        pmap = ParallelMap(medium_func, [2, 4, 6, 8], y=[2, 2], n_inputs=2)
        pmap = ParallelMap(medium_func, None, None, w=[np.ones((3, 3)), 2 * np.ones((3,3))])
        pmap = ParallelMap(medium_func, None, None, z=np.zeros((3,)))
        pmap = ParallelMap(medium_func, None, None, z=np.zeros((3, 1)))

        # Lots of ways for this to go wrong...
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], 2, w=np.ones((3,)))
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], y=22, w=np.ones((7, 1)))
        pmap = ParallelMap(hard_func, np.ones((3,)), 1, w=np.ones((7, 1)))
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], z=np.array([1, 2]), w=np.ones((8, 1)), n_inputs=2)
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], w=np.ones((8, 1)), n_inputs=4)

        # Ensure erroneous/ambiguous setups trigger the appropriate errors:
        # not enough positional args
        with pytest.raises(ValueError) as valerr:
            ParallelMap(simple_func, 4)
            assert "simple_func expects 2 positional arguments ('x', 'y'), found 1" in str(valerr.value)
        # invalid kwargs
        with pytest.raises(ValueError) as valerr:
            ParallelMap(simple_func, 4, 4, z=3, w=4)
            assert "simple_func accepts at maximum 1 keyword arguments ('z'), found 2" in str(valerr.value)
        # ill-posed parallelization: two candidate lists for input distribution
        with pytest.raises(ValueError) as valerr:
            ParallelMap(simple_func, [2, 4, 6, 8], [2, 2])
            assert "automatic input distribution failed: found 2 objects containing 2 to 4 elements" in str(valerr.value)
        # ill-posed parallelization: two candidate lists for input distribution (`x` and `w`)
        with pytest.raises(ValueError) as valerr:
            ParallelMap(medium_func, [1, 2, 3], None, w=[np.ones((3,3)), 2 * np.ones((3,3))])
            assert "automatic input distribution failed: found 2 objects containing 2 to 3 elements." in str(valerr.value)
        # invalid input spec
        with pytest.raises(ValueError) as valerr:
            ParallelMap(simple_func, [2, 4, 6, 8], [2, 2], n_inputs=3)
            assert "No object has required length of 3 matching `n_inputs`" in str(valerr.value)
        # invalid input spec: `w` expects a NumPy array, thus it is not considered for input distribution
        with pytest.raises(ValueError) as valerr:
            ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], w=np.ones((8, 1)), n_inputs=8)
            assert "No object has required length of 8 matching `n_inputs`" in str(valerr.value)

    # Functionality tests: perform channel-concurrent low-pass filtering
    def test_filter_example(self):

        # Create tmp directory and create data-containers
        tempDir = os.path.join(os.path.abspath(os.path.expanduser("~")), "acme_tmp")
        os.makedirs(tempDir, exist_ok=True)
        sigName = os.path.join(tempDir, "sigdata.h5")
        origName = os.path.join(tempDir, "origdata.h5")
        with h5py.File(sigName, "w") as sigFile:
            dset = sigFile.create_dataset("data", data=self.sig)
            dset.attrs["b"] = self.b
            dset.attrs["a"] = self.a
        with h5py.File(origName, "w") as origFile:
            origFile.create_dataset("data", data=self.orig)

        # Parallelize across channels, write results to disk
        with ParallelMap(lowpass_simple, sigName, range(self.nChannels)) as pmap:
            resOnDisk = pmap.compute()
        assert len(pmap.kwargv["outFile"]) == pmap.n_calls
        resFiles = [os.path.join(pmap.kwargv["outDir"][0], outFile) for outFile in pmap.kwargv["outFile"]]
        assert resOnDisk == resFiles
        assert all(os.path.isfile(fle) for fle in resOnDisk)

        # Compare compuated single-channel results to expected low-freq signal
        for chNo, h5name in enumerate(resOnDisk):
            with h5py.File(h5name, "r") as h5f:
                assert np.mean(np.abs(h5f["result"][()] - self.orig[:, chNo])) < self.tol

        # Same, but collect results in memory: ensure nothing freaky happens
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         write_worker_results=False) as pmap:
            resInMem = pmap.compute()
        for chNo in range(self.nChannels):
            assert np.mean(np.abs(resInMem[chNo][0] - self.orig[:, chNo])) < self.tol

        # use taskID in userfunc and store results somewhere else
        # test cleanup
        # ensure logfile is written (in combo w/verbose)
        # ensure SLURM options are respected (njobs, partition, mem_per_job)

        # with ParallelMap(lowpass_simple, sigName, range(nChannels), logfile=True) as pmap:
        #     pmap.compute()

        # Close any open HDF5 files to not trigger any `OSError`s and clean up the tmp dir
        # sigData.file.close()
        shutil.rmtree(tempDir, ignore_errors=True)

    # # test esi-cluster-setup called separately before pmap
    # def test_existing_cluster(self, testcluster):
    #     pass
    #     # # repeat selected test w/parallel processing engine
    #     # # ensure client stays alive
    #     # client = dd.Client(testcluster)
    #     # par_tests = ["test_relative_array_padding",
    #     #              "test_absolute_nextpow2_array_padding",
    #     #              "test_object_padding",
    #     #              "test_dataselection"]
    #     # for test in par_tests:
    #     #     getattr(self, test)()
    #     # client.close()
