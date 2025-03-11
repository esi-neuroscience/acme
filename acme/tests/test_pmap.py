#
# Testing module for ACME's `ParallelMap` interface
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import os
import sys
import platform
import pickle
import shutil
import inspect
import subprocess
import getpass
import time
import math
import itertools
import logging
import h5py
import pytest
import signal as sys_signal
import numpy as np
import dask.distributed as dd
from logging import handlers
from glob import glob
from scipy import signal
from numpy.typing import NDArray
from typing import Any, Optional, Union, Dict

# Import main actors here
from acme import ParallelMap, ACMEdaemon, cluster_cleanup, esi_cluster_setup, bic_cluster_setup
from conftest import skip_if_not_linux, useSLURM, onESI, onBIC, onx86, defaultQ

# Define custom types
realArrayLike = Union[float, NDArray[np.float64]]
realArray = NDArray[np.float64]

# Functions that act as stand-ins for user-funcs
def simple_func(
        x: float,
        y: float,
        z: float = 3) -> float:
    return (x + y) * z

def medium_func(
        x: realArray,
        y: realArrayLike,
        z: realArrayLike = 3,
        w: NDArray = np.ones((3, 3))) -> realArrayLike:
    return (sum(x) + y) * z * w.max()

def hard_func(
        x: realArray,
        y: realArrayLike,
        z: realArrayLike = 3,
        w: realArray = np.zeros((3, 1)),
        **kwargs: Optional[Any]) -> tuple[realArrayLike, realArray]:
    return sum(x) + y,  z * w

def lowpass_simple(
        h5name: str,
        channel_no: int) -> realArray:
    with h5py.File(h5name, "r") as h5f:
        channel = h5f["data"][:, channel_no]
        b = h5f["data"].attrs["b"]
        a = h5f["data"].attrs["a"]
    res = signal.filtfilt(b, a, channel, padlen=200)
    return res

def lowpass_medium(
        h5name: str,
        channel_no: int) -> tuple[realArray, int, float, float]:
    with h5py.File(h5name, "r") as h5f:
        channel = h5f["data"][:, channel_no]
        b = h5f["data"].attrs["b"]
        a = h5f["data"].attrs["a"]
    res = signal.filtfilt(b, a, channel, padlen=200)
    return res, channel_no, b, a

def lowpass_hard(
        arr_like: realArray,
        b: float,
        a: float,
        res_dir: str,
        res_base: str = "lowpass_hard_",
        dset_name: str = "custom_dset_name",
        padlen: int = 200,
        taskID: Optional[int] = None) -> None:
    channel = arr_like[:, taskID]
    res = signal.filtfilt(b, a, channel, padlen=padlen)
    h5name = os.path.join(res_dir, f"{res_base}{taskID}.h5")
    with h5py.File(h5name, "w") as h5f:
        h5f.create_dataset(dset_name, data=res)
    return

def pickle_func(
        arr: realArray,
        b: float,
        a: float,
        channel_no: int,
        sabotage_hdf5: bool = False) -> Union[Dict[str, float], realArray]:
    res = signal.filtfilt(b, a, arr[:, channel_no], padlen=200)
    if sabotage_hdf5:
        if channel_no % 2 == 0:
            return {"b" : b}
    return res

def memtest_func(
        x: float,
        y: float,
        z: float = 3,
        arrsize: float = 2,
        sleeper: int = 300) -> float:
    fSize = np.dtype("float").itemsize
    time.sleep(2)
    arr = np.ones((int(arrsize * 1024**3 / fSize), ))   # `arrsize` denotes array size in GB
    time.sleep(sleeper)
    return (x + y) * z * arr.max()


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

    # Helper method for allocating data containers
    def _prep_data(self, tmpName):

        # Create tmp directory and create data-containers
        tempDir = os.path.join(os.path.abspath(os.path.expanduser("~")), tmpName)
        if useSLURM and (onESI or onBIC):
            tempDir = f"/mnt/hpc/home/{getpass.getuser()}/{tmpName}"
        shutil.rmtree(tempDir, ignore_errors=True)
        os.makedirs(tempDir, exist_ok=True)
        sigName = os.path.join(tempDir, "sigdata.h5")
        origName = os.path.join(tempDir, "origdata.h5")
        with h5py.File(sigName, "w") as sigFile:
            dset = sigFile.create_dataset("data", data=self.sig)
            dset.attrs["b"] = self.b
            dset.attrs["a"] = self.a
        with h5py.File(origName, "w") as origFile:
            origFile.create_dataset("data", data=self.orig)

        return tempDir, sigName

    # Test setup of `ParallelMap` w/different functions args/kwargs
    def test_init(self, testclient=None):

        # Collected auto-generated output directories in list for later cleanup
        outDirs = []

        # Basic functionality w/simplest conceivable user-func
        pmap = ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap = ParallelMap(simple_func, [2, 4, 6, 8], y=4, partition=defaultQ, setup_interactive=False)  # pos arg referenced via kwarg, cfg #2
        outDirs.append(pmap.daemon.out_dir)
        pmap = ParallelMap(simple_func, 0, 4, z=[3, 4, 5, 6], partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap = ParallelMap(simple_func, [2, 4, 6, 8], [2, 2], n_inputs=2, partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)

        # User func has `np.ndarray` as keyword
        pmap = ParallelMap(medium_func, [2, 4, 6, 8], y=[2, 2], n_inputs=2, partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap = ParallelMap(medium_func, None, None, w=[np.ones((3, 3)), 2 * np.ones((3,3))], partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap = ParallelMap(medium_func, None, None, z=np.zeros((3,)), partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap = ParallelMap(medium_func, None, None, z=np.zeros((3, 1)), partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)

        # Lots of ways for this to go wrong...
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], 2, w=np.ones((3,)), partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], y=22, w=np.ones((7, 1)), partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap = ParallelMap(hard_func, np.ones((3,)), 1, w=np.ones((7, 1)), partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], z=np.array([1, 2]), w=np.ones((8, 1)), n_inputs=2, partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], w=np.ones((8, 1)), n_inputs=4, partition=defaultQ, setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)

        # Ensure erroneous/ambiguous setups trigger the appropriate errors:
        # not enough positional args
        with pytest.raises(ValueError) as valerr:
            ParallelMap(simple_func, 4, setup_interactive=False)
        assert "simple_func expects 2 positional arguments ('x', 'y'), found 1" in str(valerr.value)
        # invalid kwargs
        with pytest.raises(ValueError) as valerr:
            ParallelMap(simple_func, 4, 4, z=3, w=4, setup_interactive=False)
        assert "simple_func accepts at maximum 1 keyword arguments ('z'), found 2" in str(valerr.value)
        # ill-posed parallelization: two candidate lists for input distribution
        with pytest.raises(ValueError) as valerr:
            ParallelMap(simple_func, [2, 4, 6, 8], [2, 2], setup_interactive=False)
        assert "automatic input distribution failed: found 2 objects containing 2 to 4 elements" in str(valerr.value)
        # ill-posed parallelization: two candidate lists for input distribution (`x` and `w`)
        with pytest.raises(ValueError) as valerr:
            ParallelMap(medium_func, [1, 2, 3], None, w=[np.ones((3,3)), 2 * np.ones((3,3))], setup_interactive=False)
        assert "automatic input distribution failed: found 2 objects containing 2 to 3 elements." in str(valerr.value)
        # invalid input spec
        with pytest.raises(ValueError) as valerr:
            ParallelMap(simple_func, [2, 4, 6, 8], [2, 2], n_inputs=3, setup_interactive=False)
        assert "No object has required length of 3 matching `n_inputs`" in str(valerr.value)
        # invalid input spec: `w` expects a NumPy array, thus it is not considered for input distribution
        with pytest.raises(ValueError) as valerr:
            ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], w=np.ones((8, 1)), n_inputs=8, setup_interactive=False)
        assert "No object has required length of 8 matching `n_inputs`" in str(valerr.value)

        # Check if other parameters  are parsed correctly
        with pytest.raises(TypeError):
            ParallelMap("invalid")
        with pytest.raises(ValueError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, n_inputs="not-auto")
        with pytest.raises(ValueError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, n_inputs=-3)
        with pytest.raises(ValueError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, n_inputs=3.6)
        with pytest.raises(TypeError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, n_inputs={})
        with pytest.raises(TypeError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, write_worker_results="invalid")
        with pytest.raises(TypeError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, single_file="invalid")
        with pytest.raises(TypeError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, write_pickle="invalid")
        with pytest.raises(TypeError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, logfile=3)
        with pytest.raises(IOError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, logfile=os.path.dirname(os.path.realpath(__file__)))
        with pytest.raises(TypeError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, output_dir=2)
        with pytest.raises(OSError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, output_dir="/path/to/nowhere")
        with pytest.raises(TypeError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, stop_client=3)
        with pytest.raises(ValueError):
            ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, stop_client="not-auto")

        # Check parameters that are only parsed if a new client has been started
        if testclient is None:
            cluster_cleanup()
            if useSLURM:
                with pytest.raises(TypeError):
                    ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=3)
            with pytest.raises(ValueError):
                ParallelMap(simple_func, [2, 4, 6, 8], 4, partition=defaultQ, n_workers="invalid")

            # start a client for real
            cluster_cleanup()
            pmap = ParallelMap(simple_func,
                               [2, 4, 6, 8],
                               4,
                               partition=defaultQ,
                               n_workers=1,
                               setup_interactive=False)
            outDirs.append(pmap.daemon.out_dir)

            with pytest.raises(TypeError):
                pmap.daemon.compute(debug="invalid")

            # Kill our single worker and ensure ACME takes note
            client = pmap.daemon.client
            client.retire_workers(list(client.scheduler_info()['workers']), close_workers=True)
            with pytest.raises(RuntimeError) as rerr:
                pmap.daemon.compute()
            assert "no active workers found" in str(rerr.value)

            # Annihilate client slot and ensure pmap does not do anything funky
            pmap.daemon.client = None
            assert pmap.compute() is None

        # Finally, test ACMEdaemon only accepts `ParallelMap` objects
        with pytest.raises(TypeError) as tperr:
            ACMEdaemon("invalid")
        assert "`pmap` has to be a `ParallelMap` instance, not <class 'str'>" in str(tperr.value)

        # Clean up testing folder and any running clients
        if testclient is None:
            cluster_cleanup()
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)

        return testclient

    # test if examples shown on GitHub actually work
    def test_github_examples(self):

        # Collected auto-generated output directories in list for later cleanup
        outDirs = []

        def f(x, y, z=3):
            return (x + y) * z
        expected = list(map(f, [2, 4, 6, 8], [4, 4, 4, 4]))

        with ParallelMap(f, [2, 4, 6, 8], 4) as pmap:
            filenames = pmap.compute()
        outDirs.append(pmap.out_dir)

        out = np.zeros((4,))
        with h5py.File(pmap.results_container, "r") as h5f:
            for k, key in enumerate(h5f.keys()):
                out[k] = h5f[key]["result_0"][()]
        assert np.array_equal(out, expected)

        with ParallelMap(f, [2, 4, 6, 8], 4, result_shape=(None,)) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)
        with h5py.File(pmap.results_container, "r") as h5f:
            out == h5f["result_0"][()] # returns a NumPy array of shape (4,)
        assert np.array_equal(out, expected)

        mockName = f"mock_data_{platform.machine()}"
        tempDir = os.path.join(os.path.abspath(os.path.expanduser("~")), mockName)
        if useSLURM and (onESI or onBIC):
            tempDir = f"/mnt/hpc/home/{getpass.getuser()}/{mockName}"
        shutil.rmtree(tempDir, ignore_errors=True)
        os.makedirs(tempDir, exist_ok=True)
        outDirs.append(tempDir)

        # Assume only the channel count but not the number of samples is known
        nChannels = 10
        nSamples = 1234
        mock_data = np.random.rand(nChannels, nSamples)
        mock_file = os.path.join(tempDir, "mock_data.npy")
        np.save(mock_file, mock_data)

        def mock_processing(val):
            data = np.load(mock_file)
            return val * data

        with ParallelMap(mock_processing, [2, 4, 6, 8], result_shape=(None, nChannels, np.inf)) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        with h5py.File(pmap.results_container, "r") as h5f:
            mock_processed = h5f["result_0"][()] # returns a NumPy array of shape (4, nChannels, nSamples)
        for k, val in enumerate([2, 4, 6, 8]):
            assert np.array_equal(mock_processed[k], val * mock_data)

        with ParallelMap(f, [2, 4, 6, 8], 4, write_worker_results=False) as pmap:
            result = pmap.compute() # returns a 4-element list
        assert result == expected
        with ParallelMap(f, [2, 4, 6, 8], 4, write_worker_results=False, result_shape=(None,)) as pmap:
            result = pmap.compute() # returns a NumPy array of shape (4,)
        assert np.array_equal(out, expected)

        def f(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
            return (sum(x) + y) * z * w.max()
        expected = list(map(f, 2*[[2, 4, 6, 8]], [2, 2], np.array([1, 2]), 2*[np.ones((8, 1))]))

        pmap = ParallelMap(f, [2, 4, 6, 8], [2, 2], z=np.array([1, 2]), w=np.ones((8, 1)), n_inputs=2)
        with pmap as p:
            p.compute()
        outDirs.append(pmap.daemon.out_dir)
        out = []
        with h5py.File(pmap.daemon.results_container, "r") as h5f:
            for k, key in enumerate(h5f.keys()):
                out.append(h5f[key]["result_0"][()][0])
        assert np.array_equal(out, expected)

        cluster_cleanup()

        def f(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
            return (sum(x) + y) * z * w.max()

        def g(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
            return (max(x) + y) * z * w.sum()

        n_workers = 2
        x = [2, 4, 6, 8]
        y = np.random.rand(n_workers)
        z = range(n_workers)
        w = np.ones((8, 1))

        setup_kwargs = {"partition" : defaultQ,
                        "n_workers" : n_workers,
                        "interactive" : False}
        if onESI:
            client = esi_cluster_setup(**setup_kwargs)

        elif onBIC:
            client = bic_cluster_setup(**setup_kwargs)

        expected = list(map(f, n_workers*[x], y, list(z), n_workers*[w]))
        pmap = ParallelMap(f, x, y, z=z, w=w, n_inputs=n_workers)
        with pmap as p:
            p.compute()
        outDirs.append(pmap.daemon.out_dir)
        out = []
        with h5py.File(pmap.daemon.results_container, "r") as h5f:
            for k, key in enumerate(h5f.keys()):
                out.append(h5f[key]["result_0"][()][0])
        assert np.array_equal(out, expected)

        time.sleep(2.0)

        expected = list(map(g, n_workers*[x], y, list(z), n_workers*[w]))
        pmap = ParallelMap(g, x, y, z=z, w=w, n_inputs=n_workers)
        with pmap as p:
            p.compute()
        outDirs.append(pmap.daemon.out_dir)
        out = []
        with h5py.File(pmap.daemon.results_container, "r") as h5f:
            for k, key in enumerate(h5f.keys()):
                out.append(h5f[key]["result_0"][()][0])
        assert np.array_equal(out, expected)

        # Clean up testing folder and shut down allocated client
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)
        cluster_cleanup(client)

    # Functionality tests: perform channel-concurrent low-pass filtering
    def test_simple_filter(self, testclient=None):

        # Prepare data containers
        tempDir, sigName = self._prep_data(f"acme_tmp_{platform.machine()}")

        # Collect auto-generated output directories in list for later cleanup
        outDirs = []

        # Parallelize across channels, write results to disk
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            resOnDisk = pmap.compute()
        outDirs.append(pmap.out_dir)

        # Ensure each compute run generated a dedicated HDF5 file
        assert len(pmap.kwargv["outFile"]) == pmap.n_calls

        # Query auto-generated output directory
        outDirContents = glob(os.path.join(pmap.out_dir, "*"))
        payloadDir = pmap.results_container.replace(".h5", "_payload")
        assert pmap.results_container in outDirContents
        assert payloadDir in outDirContents
        resFiles = glob(os.path.join(payloadDir, "*.h5"))
        assert len(resFiles) == pmap.n_calls
        assert all(fle in resFiles for fle in resOnDisk)
        assert all(os.path.isfile(fle) for fle in resOnDisk)
        log = logging.getLogger("ACME")
        logFileList = [handler.target.baseFilename for handler in log.handlers if isinstance(handler, handlers.MemoryHandler)]
        assert len(logFileList) == 1
        assert logFileList[0] in outDirContents
        assert len(outDirContents) == 3 # results container, payload dir and log

        # Compare computed single-channel results to expected low-freq signal
        # and ensure collection container was assembled correctly
        with h5py.File(pmap.results_container, "r") as h5col:
            dset = "comp_{}/result_0"
            for chNo, h5name in enumerate(resOnDisk):
                with h5py.File(h5name, "r") as h5f:
                    assert np.mean(np.abs(h5f["result_0"][()] - self.orig[:, chNo])) < self.tol
                    assert np.array_equal(h5col[dset.format(chNo)][()], h5f["result_0"][()])

        # Remember results for later use
        colRes = str(pmap.results_container)
        colResPayload = str(payloadDir)

        # Same with `single_file`
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         partition=defaultQ,
                         setup_interactive=False,
                         single_file=True) as pmap:
            singleResOnDisk = pmap.compute()
        outDirs.append(pmap.out_dir)

        # Ensure only one file was generated
        assert len(singleResOnDisk) == 1
        outDirContents = glob(os.path.join(pmap.out_dir, "*.h5"))
        assert outDirContents == singleResOnDisk
        assert pmap.results_container in outDirContents[0]
        assert os.path.isfile(singleResOnDisk[0])

        # Compare results to container computed above
        with h5py.File(colRes, "r") as h5col:
            with h5py.File(pmap.results_container, "r") as h5single:
                dset = "comp_{}/result_0"
                for chNo in range(self.nChannels):
                    assert np.array_equal(h5single[dset.format(chNo)][()], h5col[dset.format(chNo)][()])

        # Now use non-standard output directory
        outDir = os.path.join(tempDir, "somewhere")
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         output_dir=outDir,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            resOnDisk = pmap.compute()

        # Query specified custom output directory
        assert pmap.out_dir == outDir
        outDirContents = glob(os.path.join(pmap.out_dir, "*"))
        payloadDir = pmap.results_container.replace(".h5", "_payload")
        assert pmap.results_container in outDirContents
        assert payloadDir in outDirContents
        resFiles = glob(os.path.join(payloadDir, "*.h5"))
        assert len(resFiles) == pmap.n_calls
        assert all(fle in resFiles for fle in resOnDisk)
        assert all(os.path.isfile(fle) for fle in resOnDisk)

        # A little overly paranoid, but compare results still...
        with h5py.File(colRes, "r") as h5col:
            with h5py.File(pmap.results_container, "r") as h5comp:
                dset = "comp_{}/result_0"
                for chNo in range(self.nChannels):
                    assert np.array_equal(h5comp[dset.format(chNo)][()], h5col[dset.format(chNo)][()])

        # Finally collect results in memory: ensure nothing freaky happens
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         write_worker_results=False,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            resInMem = pmap.compute()

        # Be double-paranoid: ensure on-disk and in-memory results match up
        with h5py.File(colRes, "r") as h5col:
            dset = "comp_{}/result_0"
            for chNo in range(self.nChannels):
                assert np.array_equal(h5col[dset.format(chNo)][()], resInMem[chNo])

        # Comparisons are over, now remove payload and ensure container is broken
        shutil.rmtree(colResPayload)
        with pytest.raises(KeyError) as keyerr:
            with h5py.File(colRes, "r") as h5col:
                chNo = np.random.choice(self.nChannels, size=1)[0]
                h5col[f"comp_{chNo}"]["result_0"]
        assert "unable to open external file" in str(keyerr.value)

        # Ensure `output_dir` is properly ignored if `write_worker_results` is `False`
        pmap = ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         output_dir=tempDir,
                         write_worker_results=False,
                         partition=defaultQ,
                         setup_interactive=False)
        assert pmap.daemon.out_dir is None
        assert pmap.daemon.collect_results is True

        # Simulate user-defined results-directory not auto-populated by ACME
        tempDir2 = os.path.join(os.path.abspath(os.path.expanduser("~")), "acme_tmp_lowpass_hard")
        if useSLURM and (onESI or onBIC):
            tempDir2 = f"/mnt/hpc/home/{getpass.getuser()}/acme_tmp_lowpass_hard_{platform.machine()}"
        shutil.rmtree(tempDir2, ignore_errors=True)
        os.makedirs(tempDir2, exist_ok=True)

        # Same task, different function: simulate user-defined saving scheme and "weird" inputs
        sigData = h5py.File(sigName, "r")["data"]
        res_base = "lowpass_hard_"
        dset_name = "custom_dset_name"
        with ParallelMap(lowpass_hard,
                         sigData,
                         self.b,
                         self.a,
                         res_dir=tempDir2,
                         res_base=res_base,
                         dset_name=dset_name,
                         padlen=[200] * self.nChannels,
                         n_inputs=self.nChannels,
                         write_worker_results=False,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        resFiles = glob(os.path.join(tempDir2, res_base + "*"))
        assert len(resFiles) == pmap.n_calls

        # Compare computed single-channel results to expected low-freq signal
        for chNo in range(self.nChannels):
            h5name = f"{res_base}{chNo}.h5"
            with h5py.File(os.path.join(tempDir2, h5name), "r") as h5f:
                assert np.mean(np.abs(h5f[dset_name][()] - self.orig[:, chNo])) < self.tol

        # Ensure log-file generation produces a non-empty log-file at the expected location
        # Bonus: leave computing client alive and vet default SLURM settings
        if testclient is None:
            cluster_cleanup(pmap.client)
        log = logging.getLogger("ACME")
        for handler in log.handlers:
            if isinstance(handler, logging.FileHandler):
                log.handlers.remove(handler)
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         logfile=True,
                         stop_client=False,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)
        logFileList = [handler.target.baseFilename for handler in log.handlers if isinstance(handler, handlers.MemoryHandler)]
        assert len(logFileList) == 1
        logFile = logFileList[0]
        # If running on the ESI cluster, account for /cs/home
        if onESI and useSLURM:
            assert "/cs/home/" in logFile
        else:
            assert os.path.dirname(os.path.realpath(__file__)) in logFile
        with open(logFile, "r", encoding="utf8") as fl:
            assert len(fl.readlines()) > 1

        # Ensure client has not been killed; perform post-hoc check of default SLURM settings
        assert dd.get_client()
        client = dd.get_client()
        if useSLURM is True and testclient is None:
            assert pmap.n_calls == pmap.n_workers
            assert len(client.cluster.workers) == pmap.n_workers
            partition = client.cluster.job_header.split("-p ")[1].split("\n")[0]
            if onx86:
                assert "8GB" in partition
                memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
                assert memory.size == 1
                assert math.ceil(memory[0] / 1000**3) == [int(s) for s in partition if s.isdigit()][0]

        # Wait a sec (literally) for dask to collect its bearings (after the
        # `get_client` above) before proceeding
        time.sleep(1.0)

        # Same, but use custom log-file
        customLog = os.path.join(tempDir, "acme_log.txt")
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         logfile=customLog,
                         verbose=True,
                         stop_client=testclient is None,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)
        assert os.path.isfile(customLog)
        with open(customLog, "r", encoding="utf8") as fl:
            assert len(fl.readlines()) > 1

        # Ensure only single log file `customLog` is used
        logFileList = [handler.target.baseFilename for handler in log.handlers if isinstance(handler, handlers.MemoryHandler)]
        assert len(logFileList) == 1
        assert logFileList[0] == customLog

        # Ensure client has been stopped
        if testclient is None:
            with pytest.raises(ValueError):
                dd.get_client()

        # Wait a sec (literally) to give dask enough time to close the client
        time.sleep(1.0)

        # Request a log-file but don't save results
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         logfile=True,
                         write_worker_results=False,
                         n_workers=1,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        assert pmap.out_dir is None
        log = logging.getLogger("ACME")
        logFileList = [handler.target.baseFilename for handler in log.handlers if isinstance(handler, handlers.MemoryHandler)]
        assert len(logFileList) == 1
        assert os.path.dirname(os.path.realpath(__file__)) in logFileList[0]
        os.unlink(logFileList[0])

        # Ensure ACME warns if arguments increase its "sanity" threshold
        # (lowered here to not overwhelm CI runners)
        mAS = ParallelMap._maxArgSize
        ParallelMap._maxArgSize = 1   # in MB
        pmap = ParallelMap(simple_func,
                           [np.ones((1000, 1000)), 4, 6, 8],
                           4,
                           partition=defaultQ,
                           n_workers=1,
                           logfile=True,
                           setup_interactive=False)
        log = logging.getLogger("ACME")
        memHandlers = [h for h in log.handlers if isinstance(h, handlers.MemoryHandler)]
        assert len(memHandlers) == 1
        memHandler = memHandlers[0]
        memHandler.flush()  # important: flush memory to write its contents to file
        assert memHandler.target is not None
        thisLogFile = memHandler.target.baseFilename
        with open(thisLogFile, "r", encoding="utf8") as fl:
            logTxt = fl.read()
        assert "exceeds recommended limit of 1 MB" in logTxt
        os.unlink(thisLogFile)

        # Same with kwargs
        pmap = ParallelMap(simple_func,
                           [2, 4, 6, 8],
                           4,
                           z=np.ones((1000, 1000)),
                           partition=defaultQ,
                           n_workers=1,
                           logfile=True,
                           setup_interactive=False)
        log = logging.getLogger("ACME")
        log = logging.getLogger("ACME")
        memHandlers = [h for h in log.handlers if isinstance(h, handlers.MemoryHandler)]
        assert len(memHandlers) == 1
        memHandler = memHandlers[0]
        memHandler.flush()  # important: flush memory to write its contents to file
        assert memHandler.target is not None
        thisLogFile = memHandler.target.baseFilename
        with open(thisLogFile, "r", encoding="utf8") as fl:
            logTxt = fl.read()
        assert "exceeds recommended limit of 1 MB" in logTxt
        os.unlink(thisLogFile)

        # Reset maxArgSize and continue
        ParallelMap._maxArgSize = mAS

        # Ensure warning is issued if single-file saving is requested but result writing is turned off
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         write_worker_results=False,
                         single_file=True,
                         partition=defaultQ,
                         logfile=True,
                         n_workers=1,
                         setup_interactive=False) as pmap:
            resInMem2 = pmap.compute()
        assert pmap.out_dir is None
        assert np.all(resInMem2) == np.all(resInMem)
        log = logging.getLogger("ACME")
        logFileList = [handler.target.baseFilename for handler in log.handlers if isinstance(handler, handlers.MemoryHandler)]
        assert len(logFileList) == 1
        with open(logFileList[0], "r", encoding="utf8") as fl:
            logTxt = fl.read()
        assert "Generating a single output file only possible if `write_worker_results` is `True`" in logTxt
        os.unlink(logFileList[0])

        if testclient is None:
            cluster_cleanup(pmap.client)

        # Underbook SLURM (more calls than workers)
        n_workers = int(self.nChannels / 2)
        mem_per_worker = "2GB"
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         partition=defaultQ,
                         n_workers=n_workers,
                         mem_per_worker=mem_per_worker,
                         stop_client=False,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        # Post-hoc check of client to ensure custom settings were respected
        client = pmap.client
        assert pmap.n_calls == self.nChannels
        if useSLURM is True and testclient is None:
            assert pmap.n_workers == n_workers
            assert len(client.cluster.workers) == pmap.n_workers
            actualPartition = client.cluster.job_header.split("-p ")[1].split("\n")[0]
            assert actualPartition == defaultQ
            memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
            assert memory.size == 1
            assert round(memory[0] / 1000**3) == int(mem_per_worker.replace("GB", ""))

        # Let `cluster_cleanup` murder the custom setup and ensure it did its job
        if testclient is None:
            cluster_cleanup(pmap.client)
            with pytest.raises(ValueError):
                dd.get_client()

        # Overbook SLURM (more workers than calls)
        n_workers = self.nChannels + 2
        mem_per_worker = "3000MB"
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         partition=defaultQ,
                         n_workers=n_workers,
                         mem_per_worker=mem_per_worker,
                         stop_client=False,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        # Post-hoc check of client to ensure custom settings were respected
        client = pmap.client
        assert pmap.n_calls == self.nChannels
        if useSLURM and testclient is None:
            assert pmap.n_workers == n_workers
            assert len(client.cluster.workers) == pmap.n_workers
            actualPartition = client.cluster.job_header.split("-p ")[1].split("\n")[0]
            assert actualPartition == defaultQ
            memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
            assert memory.size == 1
            assert round(memory[0] / 1000**3) * 1000 == int(mem_per_worker.replace("MB", ""))
        if testclient is None:
            cluster_cleanup(pmap.client)

        # Close any open HDF5 files to not trigger any `OSError`s, close running clusters
        # and clean up tmp dirs and created directories/log-files
        sigData.file.close()
        try:
            os.unlink(logFile)
        except PermissionError:
            pass
        shutil.rmtree(tempDir, ignore_errors=True)
        shutil.rmtree(tempDir2, ignore_errors=True)
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)

        # Wait a second (literally) so that no new parallel workers started by
        # `test_existing_cluster` erroneously use existing HDF files
        time.sleep(1.0)

        return testclient

    # More functionality tests: ensure user-funcs w/multiple outputs are processed correctly
    def test_medium_filter(self, testclient=None):

        # Prepare data containers
        _, sigName = self._prep_data("acme_tmp2")

        # Collect auto-generated output directories in list for later cleanup
        outDirs = []

        # Parallelize across channels, write results to disk
        with ParallelMap(lowpass_medium,
                         sigName,
                         range(self.nChannels),
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
             pmap.compute()
        outDirs.append(pmap.out_dir)

        # Ensure output container was created correctly
        payloadDir = pmap.results_container.replace(".h5", "_payload")
        assert len(glob(os.path.join(payloadDir, "*.h5"))) == pmap.n_calls

        # Compare computed results to expected values
        with h5py.File(pmap.results_container, "r") as h5col:
            dset = "comp_{}/result_{}"
            for chNo in range(self.nChannels):
                assert len(h5col[f"comp_{chNo}"].keys()) == 4
                assert np.mean(np.abs(h5col[dset.format(chNo, 0)][()] - self.orig[:, chNo])) < self.tol
                assert h5col[dset.format(chNo, 1)][()] == chNo
                assert np.array_equal(h5col[dset.format(chNo, 2)][()], self.b)
                assert np.array_equal(h5col[dset.format(chNo, 3)][()], self.a)

        # Remember results for later use
        colRes = str(pmap.results_container)

        # Same with `single_file`
        with ParallelMap(lowpass_medium,
                         sigName,
                         range(self.nChannels),
                         partition=defaultQ,
                         setup_interactive=False,
                         single_file=True) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        # Ensure only one (data) file was generated
        assert glob(os.path.join(pmap.out_dir, "*.h5")) == [pmap.results_container]

        # Compare single file to link collection computed above
        with h5py.File(colRes, "r") as h5col:
            with h5py.File(pmap.results_container, "r") as h5single:
                dset = "comp_{}/result_{}"
                for chNo in range(self.nChannels):
                    assert len(h5single[f"comp_{chNo}"].keys()) == 4
                    assert np.array_equal(h5single[dset.format(chNo, 0)][()], h5col[dset.format(chNo, 0)][()])
                    assert h5col[dset.format(chNo, 1)][()] == h5single[dset.format(chNo, 1)][()]
                    assert np.array_equal(h5col[dset.format(chNo, 2)][()], h5single[dset.format(chNo, 2)][()])
                    assert np.array_equal(h5col[dset.format(chNo, 3)][()], h5single[dset.format(chNo, 3)][()])

        # Same, but collect results in memory: ensure nothing freaky happens
        with ParallelMap(lowpass_medium,
                         sigName,
                         range(self.nChannels),
                         write_worker_results=False,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            resInMem = pmap.compute()

        # Be double-paranoid: ensure on-disk and in-memory results match up
        with h5py.File(colRes, "r") as h5col:
            dset = "comp_{}/result_{}"
            for chNo in range(self.nChannels):
                assert len(resInMem[chNo]) == 4
                assert np.array_equal(h5col[dset.format(chNo, 0)][()], resInMem[chNo][0])
                assert h5col[dset.format(chNo, 1)][()] == resInMem[chNo][1]
                assert np.array_equal(h5col[dset.format(chNo, 2)][()], resInMem[chNo][2])
                assert np.array_equal(h5col[dset.format(chNo, 3)][()], resInMem[chNo][3])

        # Clean up created results directories
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)

        # Wait a second (literally) so that no new parallel workers started by
        # `test_existing_cluster` erroneously use existing HDF files
        time.sleep(1.0)

        return testclient

    # Even more functionality tests: ensure output array stacking works
    def test_outshape(self, testclient=None):

        # Prepare data containers
        _, sigName = self._prep_data("acme_tmp_outshape")

        # Collect auto-generated output directories in list for later cleanup
        outDirs = []

        # Compute result length needed to determine final shape
        nSamples = self.fs * self.nTrials

        # Parallelize across channels, write results to disk
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         result_shape=(None, nSamples),
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            resOnDisk = pmap.compute()
        outDirs.append(pmap.out_dir)

        # Remember results for later use
        colRes = str(pmap.results_container)

        # Compare computed single-channel results to expected low-freq signal
        # and ensure collection container was assembled correctly
        with h5py.File(pmap.results_container, "r") as h5col:
            assert len(h5col.keys()) == 1
            assert h5col["result_0"].is_virtual
            for chNo, h5name in enumerate(resOnDisk):
                with h5py.File(h5name, "r") as h5f:
                    assert np.mean(np.abs(h5f["result_0"][()] - self.orig[:, chNo])) < self.tol
                    assert np.array_equal(h5col["result_0"][chNo, :], h5f["result_0"][()])

        # As above but don't specify `nSamples`
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         result_shape=(None, np.inf),
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        infRes = str(pmap.results_container)

        # Ensure not specifying `nSamples` did not change anything
        with h5py.File(colRes, "r") as h5col:
            with h5py.File(infRes, "r") as h5inf:
                assert np.array_equal(h5col["result_0"][()], h5inf["result_0"][()])

        # Same but don't use a virtual dataset and transpose the final array
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         result_shape=(nSamples, None),
                         single_file=True,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        singleRes = str(pmap.results_container)

        # Ensure only one (data) file was generated
        assert glob(os.path.join(pmap.out_dir, "*.h5")) == [pmap.results_container]

        # Compare single file to virtual dataset
        with h5py.File(colRes, "r") as h5col:
            with h5py.File(singleRes, "r") as h5single:
                assert len(h5single.keys()) == 1
                assert h5single["result_0"].is_virtual is False
                assert np.array_equal(h5single["result_0"][()].T, h5col["result_0"][()])

        # Use a resizable single hdf container (not specifying `nSamples`)
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         result_shape=(np.inf, None),
                         single_file=True,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        singleInfRes = str(pmap.results_container)

        # Ensure not specifying `nSamples` did not change anything
        with h5py.File(singleRes, "r") as h5single:
            with h5py.File(singleInfRes, "r") as h5singleinf:
                assert np.array_equal(h5single["result_0"][()], h5singleinf["result_0"][()])

        # More elaborate stacking/expendable dimension
        with ParallelMap(simple_func,
                         [elem * np.ones((3, 3)) for elem in [2, 4]],
                         4,
                         result_shape=(3, None, 3),
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        tripleDim = str(pmap.results_container)

        with ParallelMap(simple_func,
                         [elem * np.ones((3, 3)) for elem in [2, 4]],
                         4,
                         result_shape=(np.inf, None, 3),
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        tripleDimInf = str(pmap.results_container)

        with h5py.File(tripleDim, "r") as h5triple:
            with h5py.File(tripleDimInf, "r") as h5tripleinf:
                assert np.array_equal(h5triple["result_0"][()], h5tripleinf["result_0"][()])

        with ParallelMap(simple_func,
                         [elem * np.ones((3, 3)) for elem in [2, 4]],
                         4,
                         result_shape=(3, None, 3),
                         partition=defaultQ,
                         single_file=True,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        tripleDimSingle = str(pmap.results_container)

        with ParallelMap(simple_func,
                         [elem * np.ones((3, 3)) for elem in [2, 4]],
                         4,
                         result_shape=(np.inf, None, 3),
                         partition=defaultQ,
                         single_file=True,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        tripleDimSingleInf = str(pmap.results_container)

        with h5py.File(tripleDimSingle, "r") as h5triple:
            with h5py.File(tripleDimSingleInf, "r") as h5tripleinf:
                assert np.array_equal(h5triple["result_0"][()], h5tripleinf["result_0"][()])

        with h5py.File(tripleDim, "r") as h5triple:
            with h5py.File(tripleDimSingleInf, "r") as h5tripleinf:
                assert np.array_equal(h5triple["result_0"][()], h5tripleinf["result_0"][()])

        # Finally, ensure in-memory results-collection works as expected
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         result_shape=(None, nSamples),
                         write_worker_results=False,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            resInMem = pmap.compute()
        with h5py.File(colRes, "r") as h5col:
            assert np.array_equal(h5col["result_0"][()], resInMem)

        # Ensure dtype is respected
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         result_shape=(None, nSamples),
                         result_dtype="float16",
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)
        with h5py.File(pmap.results_container, "r") as h5f:
            assert h5f["result_0"].dtype.name == "float16"

        # Ensure invalid dtypes don't pass through
        with pytest.raises(TypeError) as tperr:
            with ParallelMap(lowpass_simple,
                             sigName,
                             range(self.nChannels),
                             result_shape=(None, nSamples),
                             result_dtype=np.ones((3,)),
                             partition=defaultQ,
                             setup_interactive=False) as pmap:
                pmap.compute()
        assert "`result_dtype` has to be a string" in str(tperr)
        with pytest.raises(TypeError) as tperr:
            with ParallelMap(lowpass_simple,
                             sigName,
                             range(self.nChannels),
                             result_shape=(None, nSamples),
                             result_dtype="invalid",
                             partition=defaultQ,
                             setup_interactive=False) as pmap:
                pmap.compute()
        assert "`result_dtype` has to be a valid NumPy datatype" in str(tperr)

        # Ensure borked shapes are caught
        with pytest.raises(TypeError) as tperr:
            with ParallelMap(lowpass_simple,
                             sigName,
                             range(self.nChannels),
                             result_shape=3,
                             partition=defaultQ,
                             setup_interactive=False) as pmap:
                pmap.compute()
        assert "`result_shape` has to be either `None` or tuple" in str(tperr)
        with pytest.raises(ValueError) as valerr:
            with ParallelMap(lowpass_simple,
                             sigName,
                             range(self.nChannels),
                             result_shape=(None, None, nSamples),
                             partition=defaultQ,
                             setup_interactive=False) as pmap:
                pmap.compute()
        assert "`result_shape` must contain exactly one `None`" in str(valerr)
        with pytest.raises(ValueError) as valerr:
            with ParallelMap(lowpass_simple,
                             sigName,
                             range(self.nChannels),
                             result_shape=(3, nSamples),
                             partition=defaultQ,
                             setup_interactive=False) as pmap:
                pmap.compute()
        assert "`result_shape` must contain exactly one `None`" in str(valerr)
        with pytest.raises(ValueError) as valerr:
            with ParallelMap(lowpass_simple,
                             sigName,
                             range(self.nChannels),
                             result_shape=("invalid", None, nSamples),
                             partition=defaultQ,
                             setup_interactive=False) as pmap:
                pmap.compute()
        assert "`result_shape` must only contain numerical values" in str(valerr)
        with pytest.raises(ValueError) as valerr:
            with ParallelMap(lowpass_simple,
                             sigName,
                             range(self.nChannels),
                             result_shape=(-3, None, nSamples),
                             partition=defaultQ,
                             setup_interactive=False) as pmap:
                pmap.compute()
        assert "`result_shape` must only contain non-negative integers" in str(valerr)
        with pytest.raises(ValueError) as valerr:
            with ParallelMap(lowpass_simple,
                             sigName,
                             range(self.nChannels),
                             result_shape=(np.pi, None, nSamples),
                             partition=defaultQ,
                             setup_interactive=False) as pmap:
                pmap.compute()
        assert "`result_shape` must only contain non-negative integers" in str(valerr)
        with pytest.raises(ValueError) as valerr:
            with ParallelMap(lowpass_simple,
                             sigName,
                             range(self.nChannels),
                             result_shape=(None, np.inf),
                             partition=defaultQ,
                             write_worker_results=False,
                             setup_interactive=False) as pmap:
                pmap.compute()
        assert "`np.inf` in `result_shape` is only valid if `write_worker_results` is `True`" in str(valerr)
        with pytest.raises(ValueError) as valerr:
            with ParallelMap(lowpass_simple,
                             sigName,
                             range(self.nChannels),
                             result_shape=(None, np.inf, np.inf),
                             partition=defaultQ,
                             setup_interactive=False) as pmap:
                pmap.compute()
        assert "cannot use more than one `np.inf` in `result_shape`" in str(valerr)

        # Emergency pickling
        with ParallelMap(pickle_func,
                         self.sig,
                         self.b,
                         self.a,
                         range(self.nChannels),
                         sabotage_hdf5=True,
                         n_inputs=self.nChannels,
                         result_shape=(nSamples, None),
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            mixedResults = pmap.compute()
        outDirs.append(pmap.out_dir)

        # Ensure pickles and hdf5's live together happily
        resultsContainer = os.path.basename(mixedResults[0])
        resultsContainer = os.path.join(os.path.dirname(mixedResults[0]),
                                        resultsContainer[:resultsContainer.rfind("_0")] + ".h5")
        payloadDir = resultsContainer.replace(".h5", "_payload")
        assert pmap.results_container is None
        assert not os.path.isfile(resultsContainer)
        assert not os.path.isdir(payloadDir)
        assert all(os.path.isfile(fle) for fle in mixedResults)
        assert len(mixedResults) == pmap.n_calls

        # Ensure deliberate pickling doesn't clash w/(erroneous) shape spec
        with ParallelMap(pickle_func,
                         self.sig,
                         self.b,
                         self.a,
                         range(self.nChannels),
                         sabotage_hdf5=False,
                         n_inputs=self.nChannels,
                         result_shape=(nSamples, None),
                         write_pickle=True,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pickles = pmap.compute()
        outDirs.append(pmap.out_dir)
        assert all(os.path.isfile(fle) for fle in pickles)
        assert len(pickles) == pmap.n_calls

        # Ensure multiple return values are handled correctly
        with ParallelMap(lowpass_medium,
                         sigName,
                         range(self.nChannels),
                         result_shape=(None, nSamples),
                         single_file=False,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        # Save results for later
        multiRet = str(pmap.results_container)

        # Compare computed results to stored "reference"
        with h5py.File(colRes, "r") as h5col:
            with h5py.File(pmap.results_container, "r") as h5f:
                assert len(h5f.keys()) == pmap.n_calls + 1
                assert h5f["result_0"].is_virtual is True
                assert np.array_equal(h5f["result_0"][()], h5col["result_0"][()])
                for k in range(pmap.n_calls):
                    assert len(h5f[f"comp_{k}"].keys()) == 3
                    assert h5f[f"comp_{k}/result_1"][()] == k
                    assert np.array_equal(h5f[f"comp_{k}/result_2"][()], self.b)
                    assert np.array_equal(h5f[f"comp_{k}/result_3"][()], self.a)

        # Same w/single output container
        with ParallelMap(lowpass_medium,
                         sigName,
                         range(self.nChannels),
                         result_shape=(None, nSamples),
                         single_file=True,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
             pmap.compute()
        outDirs.append(pmap.out_dir)

        # Compare results
        with h5py.File(multiRet, "r") as h5ref:
            with h5py.File(pmap.results_container, "r") as h5f:
                assert len(h5f.keys()) == pmap.n_calls + 1
                for k in range(pmap.n_calls):
                    for rk in range(1,4):
                        dset = f"comp_{k}/result_{rk}"
                        assert np.array_equal(h5f[dset], h5ref[dset])

        # Finally, ensure in-memory results-collection works w/multiple returns
        with ParallelMap(lowpass_medium,
                         sigName,
                         range(self.nChannels),
                         result_shape=(None, nSamples),
                         write_worker_results=False,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            resInMem = pmap.compute()
        with h5py.File(multiRet, "r") as h5ref:
            assert np.array_equal(h5ref["result_0"][()], resInMem[0])
            rCount = 1
            for k in range(pmap.n_calls):
                assert h5ref[f"comp_{k}/result_1"][()] == resInMem[rCount]
                rCount += 1
                assert np.array_equal(h5ref[f"comp_{k}/result_2"][()], resInMem[rCount])
                rCount += 1
                assert np.array_equal(h5ref[f"comp_{k}/result_3"][()], resInMem[rCount])
                rCount += 1

        # Clean up created results directories
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)

        # Wait a second (literally) so that no new parallel workers started by
        # `test_existing_cluster` erroneously use existing HDF files
        time.sleep(1.0)

        return testclient

    # Test if pickling/emergency pickling and I/O in general works as intended
    def test_pickling(self, testclient=None):

        # Collected auto-generated output directories in list for later cleanup
        outDirs = []

        # Execute `pickle_func` w/regular HDF5 saving
        with ParallelMap(pickle_func,
                         self.sig,
                         self.b,
                         self.a,
                         range(self.nChannels),
                         sabotage_hdf5=False,
                         n_inputs=self.nChannels,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            hdfResults = pmap.compute()
        colRes = str(pmap.results_container)
        outDirs.append(pmap.out_dir)

        # Execute `pickle_func` w/pickling
        with ParallelMap(pickle_func,
                         self.sig,
                         self.b,
                         self.a,
                         range(self.nChannels),
                         n_inputs=self.nChannels,
                         write_pickle=True,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            pklResults = pmap.compute()
        outDirs.append(pmap.out_dir)

        # Ensure HDF5 and pickle match up
        with h5py.File(colRes, "r") as h5col:
            dset = "comp_{}/result_0"
            for chNo in range(self.nChannels):
                with open(pklResults[chNo], "rb") as pkf:
                    pklRes = pickle.load(pkf)
                assert np.array_equal(pklRes, h5col[dset.format(chNo)][()])

        # Ensure single_file and pickling does not work
        with pytest.raises(ValueError) as valerr:
            with ParallelMap(pickle_func,
                            self.sig,
                            self.b,
                            self.a,
                            range(self.nChannels),
                            n_inputs=self.nChannels,
                            write_pickle=True,
                            single_file=True,
                            partition=defaultQ,
                            setup_interactive=False) as pmap:
                pmap.compute()
        assert "Pickling of results does not support single output file creation" in str(valerr.value)

        # Test emergency pickling
        with ParallelMap(pickle_func,
                         self.sig,
                         self.b,
                         self.a,
                         range(self.nChannels),
                         sabotage_hdf5=True,
                         n_inputs=self.nChannels,
                         partition=defaultQ,
                         setup_interactive=False) as pmap:
            mixedResults = pmap.compute()
        outDirs.append(pmap.out_dir)

        # Ensure warning is issued if pickling is requested but result writing is turned off
        with ParallelMap(simple_func,
                         [2, 4, 6, 8],
                         4,
                         partition=defaultQ,
                         write_worker_results=False,
                         write_pickle=True,
                         logfile=True,
                         n_workers=1,
                         setup_interactive=False) as pmap:
                results = pmap.compute()
        assert pmap.out_dir is None
        assert results == list(map(simple_func, [2, 4, 6, 8], [4, 4, 4, 4]))
        log = logging.getLogger("ACME")
        logFileList = [handler.target.baseFilename for handler in log.handlers if isinstance(handler, handlers.MemoryHandler)]
        assert len(logFileList) == 1
        with open(logFileList[0], "r", encoding="utf8") as fl:
            logTxt = fl.read()
        assert "Pickling of results only possible if `write_worker_results` is `True`" in logTxt

        # Collection container should have been auto-removed
        resultsContainer = os.path.basename(mixedResults[0])
        resultsContainer = os.path.join(os.path.dirname(mixedResults[0]),
                                        resultsContainer[:resultsContainer.rfind("_0")] + ".h5")
        payloadDir = resultsContainer.replace(".h5", "_payload")
        assert pmap.results_container is None
        assert not os.path.isfile(resultsContainer)
        assert not os.path.isdir(payloadDir)

        # Ensure non-compliant dicts were pickled, rest is in HDF5
        for chNo, fname in enumerate(mixedResults):
            if chNo % 2 == 0:
                assert fname.endswith(".pickle")
                with open(fname, "rb") as pkf:
                    assert np.array_equal(self.b, pickle.load(pkf)[0]["b"])
            else:
                assert fname.endswith(".h5")
                with h5py.File(fname, "r") as h5f:
                    with h5py.File(hdfResults[chNo], "r") as h5ref:
                        assert np.array_equal(h5f["result_0"][()], h5ref["result_0"][()])

        # Ensure emergency pickling and single file does not work
        with pytest.raises(RuntimeError):
            with ParallelMap(pickle_func,
                            self.sig,
                            self.b,
                            self.a,
                            range(self.nChannels),
                            sabotage_hdf5=True,
                            n_inputs=self.nChannels,
                            single_file=True,
                            partition=defaultQ,
                            setup_interactive=False) as pmap:
                pmap.compute()


        # Test write breakdown (both for HDF5 saving and pickling)
        pmap = ParallelMap(pickle_func,
                           self.sig,
                           self.b,
                           self.a,
                           range(self.nChannels),
                           sabotage_hdf5=True,
                           n_inputs=self.nChannels,
                           partition=defaultQ,
                           setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap.kwargv["outFile"][0] = "/path/to/nowhere"
        with pytest.raises(RuntimeError) as runerr:
            pmap.compute()
        assert "<ACMEdaemon> Parallel computation failed" in str(runerr.value)
        pmap = ParallelMap(pickle_func,
                           self.sig,
                           self.b,
                           self.a,
                           range(self.nChannels),
                           sabotage_hdf5=True,
                           n_inputs=self.nChannels,
                           write_pickle=True,
                           partition=defaultQ,
                           setup_interactive=False)
        outDirs.append(pmap.daemon.out_dir)
        pmap.kwargv["outFile"][0] = "/path/to/nowhere"
        with pytest.raises(RuntimeError) as runerr:
            pmap.compute()
        assert "<ACMEdaemon> Parallel computation failed" in str(runerr.value)

        # Clean up testing folder
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)

        return testclient

    # test if KeyboardInterrupts are handled correctly
    @skip_if_not_linux
    def test_cancel(self):

        # Setup temp-directory layout for subprocess-scripts and prepare interpreters
        tempDir = os.path.join(os.path.abspath(os.path.expanduser("~")), f"acme_tmp_{platform.machine()}")
        shutil.rmtree(tempDir, ignore_errors=True)
        os.makedirs(tempDir, exist_ok=True)
        pshells = [os.path.join(os.path.split(sys.executable)[0], pyExec) for pyExec in ["python", "ipython"]]

        # Prepare ad-hoc script for execution in new process
        scriptName = os.path.join(tempDir, "dummy.py")
        scriptContents = \
            "from acme import ParallelMap, cluster_cleanup\n" +\
            "import time\n" +\
            "def long_running(dummy):\n" +\
            "   time.sleep(10)\n" +\
            "   return\n" +\
            "if __name__ == '__main__':\n" +\
            "   cluster_cleanup() \n" +\
            f"   with ParallelMap(long_running, [None]*2, setup_interactive=False, partition='{defaultQ}', write_worker_results=False) as pmap: \n" +\
            "       pmap.compute()\n" +\
            "   print('ALL DONE')\n"
        with open(scriptName, "w", encoding="utf8") as f:
            f.write(scriptContents)

        # Execute the above script both in Python and iPython to ensure global functionality
        for pshell in pshells:

            # Launch new process in background (`stdbuf` prevents buffering of stdout)
            proc = subprocess.Popen("stdbuf -o0 " + pshell + " " + scriptName,
                                    shell=True, start_new_session=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

            # Wait for ACME to start up (as soon as logging info is shown, `pmap.compute()` is running)
            # However: don't wait indefinitely - if `pmap.compute` is not started within 30s, abort
            logStr = "Preparing 2 parallel calls"
            buffer = bytearray()
            timeout = 90
            t0 = time.time()
            for line in itertools.takewhile(lambda x: time.time() - t0 < timeout, iter(proc.stdout.readline, b"")):
                buffer.extend(line)
                if logStr in line.decode("utf8"):
                    break
            assert logStr in buffer.decode("utf8")

            # Wait a bit, then simulate CTRL+C in sub-process; make sure the above
            # impromptu script did not run to completion *but* the created client was
            # shut down with CTRL + C
            time.sleep(2)
            os.killpg(proc.pid, sys_signal.SIGINT)
            time.sleep(1)
            out = proc.stdout.read().decode()
            assert "ALL DONE" not in out
            assert "Successfully shut down" in out

        # Almost identical script, this time use an externally started client
        if onESI:
            scriptName = os.path.join(tempDir, "dummy2.py")
            scriptContents = \
                "from acme import ParallelMap, esi_cluster_setup\n" +\
                "import time\n" +\
                "def long_running(dummy):\n" +\
                "   time.sleep(10)\n" +\
                "   return\n" +\
                "if __name__ == '__main__':\n" +\
                f"   client = esi_cluster_setup(partition='{defaultQ}',n_workers=1, interactive=False)\n" +\
                "   with ParallelMap(long_running, [None]*2, setup_interactive=False, write_worker_results=False, verbose=True) as pmap: \n" +\
                "       pmap.compute()\n" +\
                "   print('ALL DONE')\n"
            with open(scriptName, "w", encoding="utf8") as f:
                f.write(scriptContents)

            # Test script functionality in both Python and iPython
            for pshell in pshells:
                proc = subprocess.Popen("stdbuf -o0 " + sys.executable + " " + scriptName,
                                        shell=True, start_new_session=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
                logStr = "This is ACME"
                buffer = bytearray()
                timeout = 60
                t0 = time.time()
                for line in itertools.takewhile(lambda x: time.time() - t0 < timeout, iter(proc.stdout.readline, b"")):
                    buffer.extend(line)
                    if logStr in line.decode("utf8"):
                        break
                assert logStr in buffer.decode("utf8")
                time.sleep(2)
                os.killpg(proc.pid, sys_signal.SIGINT)
                time.sleep(2)
                out = proc.stdout.read().decode()
                assert "ALL DONE" not in out

            # Ensure random exception does not immediately kill an active client
            scriptName = os.path.join(tempDir, "dummy3.py")
            scriptContents = \
                "from acme import esi_cluster_setup\n" +\
                "import time\n" +\
                "if __name__ == '__main__':\n" +\
                f"   esi_cluster_setup(partition='{defaultQ}',n_workers=1, interactive=False)\n" +\
                "   time.sleep(60)\n"
            with open(scriptName, "w", encoding="utf8") as f:
                f.write(scriptContents)
            proc = subprocess.Popen("stdbuf -o0 " + sys.executable + " " + scriptName,
                                    shell=True, start_new_session=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

            # Give the client time to start up, then send a floating-point exception
            # (equivalent to a `ZeroDivsionError` to the child process)
            time.sleep(5)
            assert proc.poll() is None
            proc.send_signal(sys_signal.SIGFPE)

            # Ensure the `ZeroDivsionError` did not kill the process. Then terminate it
            # and confirm that the floating-exception was propagated correctly
            assert proc.poll() is None
            proc.terminate()
            proc.wait()
            assert proc.returncode in [-sys_signal.SIGFPE.value, -sys_signal.SIGTERM.value]

        # Clean up tmp folder
        shutil.rmtree(tempDir, ignore_errors=True)

    # test dryrun keyword
    def test_dryrun(self, monkeypatch):

        # This call tests two things: first, the actual dryrun must work, second
        # using pytest's monkeypatch fixture user-input is simulated. If a user
        # decides to not move ahead after the dryrun the auto-generated output
        # directory must be cleaned up
        monkeypatch.setattr("builtins.input", lambda _ : "n")
        pmap = ParallelMap(simple_func, [2, 4, 6, 8], 4, setup_interactive=True, dryrun=True)
        time.sleep(1.0)

        # Ensure auto-generated output dir has been successfully removed
        outDir = pmap.daemon.out_dir
        assert os.path.exists(outDir) is False

        # Now go through with the dry-run
        monkeypatch.setattr("builtins.input", lambda _ : "y")
        with ParallelMap(simple_func,
                         [2, 4, 6, 8],
                         4,
                         setup_interactive=True,
                         dryrun=True) as pmap:
            pmap.compute()
        shutil.rmtree(pmap.out_dir, ignore_errors=True)

    def test_memest(self):

        # First kill any existing client (otherwise profiling is not happening)
        cluster_cleanup()

        # Create tmp directory for logfile
        tempDir = os.path.join(os.path.abspath(os.path.expanduser("~")), f"acme_tmp_{platform.machine()}")
        shutil.rmtree(tempDir, ignore_errors=True)
        os.makedirs(tempDir, exist_ok=True)
        customLog = os.path.join(tempDir, "mem_log.txt")
        outDirs = []

        # Set `arrsize` depending on available runner hardware and prepare expected
        # mem estimates accordingly
        arrsize = 2
        estMem = 3
        if not useSLURM and platform.machine() == "ppc64le":
            arrsize = 0.5
            estMem = 1

        # Prepare `ParallelMap` instance for 2 concurrent calls of `memtest_func`:
        # a 2GB array is allocated, set final sleep wait period to 2 seconds, so
        # total runtime of the function should be b/w 5-10 seconds (depending on how long
        # array allocation takes)
        pmap = ParallelMap(memtest_func,
                           np.arange(2),
                           2,
                           sleeper=2,
                           arrsize=arrsize,
                           logfile=customLog,
                           setup_interactive=False)

        # If executed locally, the above call did not invoke `estimate_memuse` since
        # there's no partition to choose
        if not useSLURM:

            # Fire off `estimate_memuse` manually
            tic = time.perf_counter()
            memEst = pmap.daemon.estimate_memuse()
            toc = time.perf_counter()

            # Ensure 2.12GB array allocation was profiled correctly (2.x GB are rounded
            # upwards towards the next integer, i.e., 2.13 GB -> estimate_memuse:3)
            if sys.platform != "darwin":
                assert memEst == "estimate_memuse:" + str(estMem)

            # Ensure profiler quit out early (since total runtime of `memtest_func` is below 30s)
            assert toc - tic < 30

        # Check that for 2 parallel calls only 2 workers were memory-profiled
        # (we need to flush `MemoryHandler` manually so that its contents
        # is actually written to `customLog`)
        log = logging.getLogger("ACME")
        for h in log.handlers:
            if hasattr(h, "flush"):
                h.flush()
        with open(customLog, "r", encoding="utf8") as f:
            logTxt = f.read()
        assert "Estimated memory consumption across 2 runs" in logTxt

        # If running on the ESI cluster, ensure the correct partition has been picked
        if useSLURM and (onESI or onBIC):
            assert "Picked partition 8GBS based on estimated memory consumption of 3 GB" in logTxt

        # Profiling completed full run of `memtest_func`: ensure any auto-created
        # output HDF5 files were removed
        outDirs.append(pmap.daemon.out_dir)
        assert len(os.listdir(os.path.dirname(pmap.kwargv['outFile'][0]))) == 0

        # Syncopy-related test: ensure memory profiling works also if ACME
        # does not handle result collection
        syncopylog = os.path.join(tempDir, "syncopylog.txt")
        pmap = ParallelMap(memtest_func,
                           np.arange(100),
                           2,
                           sleeper=300,
                           arrsize=arrsize,
                           logfile=syncopylog,
                           setup_timeout=10,
                           write_worker_results=False,
                           setup_interactive=False)
        with open(syncopylog, "r", encoding="utf8") as f:
            logTxt = f.read()
        assert "memEstRun" not in logTxt

        # Now prepare `ParallelMap` instance for 100 concurrent calls of `memtest_func`:
        # again a 2GB array is allocated, but set final sleep to 5 minutes, enforcing
        # abort of profiling runs (set `setup_interactive` low to not have 100
        # SLURM workers being actually started)
        del pmap
        cluster_cleanup()
        customLog2 = os.path.join(tempDir, "acme_log2.txt")
        pmap = ParallelMap(memtest_func,
                           np.arange(100),
                           2,
                           sleeper=300,
                           arrsize=arrsize,
                           logfile=customLog2,
                           setup_timeout=30,
                           setup_interactive=False)

        # Again, fire off `estimate_memuse` manually if tests are run locally
        if not useSLURM:
            tic = time.perf_counter()
            memEst = pmap.daemon.estimate_memuse()
            toc = time.perf_counter()

            # Ensure 2.12GB array allocation was profiled correctly
            if sys.platform != "darwin":
                assert memEst == "estimate_memuse:" + str(estMem)

            # Ensure profiler went the whole mile (since total runtime of `memtest_func` > 5 min)
            assert 140 < toc - tic < 200

        # Check that for max 5 workers were memory-profiled
        # (we need to flush `MemoryHandler` manually so that its contents
        # is actually written to `customLog`)
        log = logging.getLogger("ACME")
        for h in log.handlers:
            if hasattr(h, "flush"):
                h.flush()
        with open(customLog2, "r", encoding="utf8") as f:
            logTxt = f.read()
        assert "Estimated memory consumption across 5 runs" in logTxt

        # If running on the ESI cluster, ensure the correct partition has been picked (again)
        if useSLURM and (onESI or onBIC):
            assert "Picked partition 8GBS based on estimated memory consumption of 3 GB" in logTxt

        # Profiling should not have generated any output
        outDirs.append(pmap.daemon.out_dir)
        assert len(os.listdir(os.path.dirname(pmap.kwargv['outFile'][0]))) == 0

        # Prepare final "full" tests
        del pmap
        cluster_cleanup()
        customLog3 = os.path.join(tempDir, "acme_log3.txt")

        # Assert that `partition="auto"` has no effect in `LocalCluster` case
        if not useSLURM:

            with ParallelMap(memtest_func,
                             np.arange(2),
                             2,
                             sleeper=2,
                             arrsize=arrsize,
                             logfile=customLog3) as pmap:
                pmap.compute()
            with open(customLog3, "r", encoding="utf8") as f:
                logTxt = f.read()
            assert "Estimating memory consumption" not in logTxt
            outDirs.append(pmap.out_dir)

        else:

            # Check auto-partition selection on ESI/BIC clusters
            if onESI or onBIC:
                if onESI:
                    setup_func = esi_cluster_setup
                else:
                    setup_func = bic_cluster_setup

                # Simulate call of ParallelMap(partition="auto",...) but w/wrong mem_per_worker!
                with pytest.raises(IOError):
                    setup_func(partition="auto", mem_per_worker="invalid")

                # Simulate `ParallelMap(partition="auto",...)` call by invoking `esi_cluster_setup`
                # with `mem_per_worker='esstimate_memuse:XY'`
                memUse = "estimate_memuse:12"
                client = setup_func(partition="auto",
                                    mem_per_worker=memUse,
                                    n_workers=1,
                                    interactive=False)

                job_head = client.cluster.job_header.split("-p ")[1].split("\n")[0]

                # Ensure the right partition was picked (16GBXY, not 8GBXY on x86)
                if onESI:
                    if onx86:
                        assert "16GB" in job_head
                    else:
                        memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
                        assert memory.size == 1
                        assert round(memory[0] / 1000**3) == int(memUse.split(":")[1])
                else:
                    assert "16GB" in job_head

                cluster_cleanup(client)

                # Full run (finally) w/10 workers, 5 of em get mem-profiled
                with ParallelMap(memtest_func,
                                 np.arange(10),
                                 2,
                                 sleeper=35,
                                 arrsize=arrsize,
                                 partition="auto",
                                 logfile=customLog3,
                                 setup_interactive=False) as pmap:
                    pmap.compute()

                # Check correct partition and no. of workers profiled
                with open(customLog3, "r", encoding="utf8") as f:
                    logTxt = f.read()
                assert "Estimated memory consumption across 5 runs" in logTxt
                assert "Picked partition 8GBS" in logTxt
                outDirs.append(pmap.out_dir)

        # Clean up
        shutil.rmtree(tempDir, ignore_errors=True)
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)
        time.sleep(0.1)

    # test if deprecated keywords are mapped onto new names correctly
    def test_backcompat(self):

        # Prepare data containers
        _, sigName = self._prep_data("acme_tmp")

        # Collect auto-generated output directories in list for later cleanup
        outDirs = []

        # Invoke `ParallelMap` w/deprecated `n_jobs` kw
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         partition=defaultQ,
                         n_jobs=2,
                         logfile=True,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.out_dir)

        # Ensure a deprecation warning was issued
        log = logging.getLogger("ACME")
        logFileList = [handler.target.baseFilename for handler in log.handlers if isinstance(handler, handlers.MemoryHandler)]
        assert len(logFileList) == 1
        with open(logFileList[0], "r", encoding="utf8") as fl:
            logTxt = fl.read()
        assert "DEPRECATED" in logTxt

        if useSLURM:

            # Use the deprecated `n_jobs` and `mem_per_job` keywords
            n_workers = 2
            mem_per_worker = "2GB"
            with ParallelMap(lowpass_simple,
                            sigName,
                            range(self.nChannels),
                            partition=defaultQ,
                            n_jobs=n_workers,
                            mem_per_job=mem_per_worker,
                            stop_client=False,
                            setup_interactive=False) as pmap:
                pmap.compute()
            outDirs.append(pmap.out_dir)

            # Ensure the provided input was interpreted correctly
            client = dd.get_client()
            assert pmap.n_workers == n_workers
            assert len(client.cluster.workers) == pmap.n_workers
            memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
            assert memory.size == 1
            assert round(memory[0] / 1000**3) == int(mem_per_worker.replace("GB", ""))

        # Clean up
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)
        time.sleep(0.1)
        cluster_cleanup()

    # test esi-cluster-setup called separately before pmap
    def test_existing_cluster(self):

        # Do not execute on GitHub runner
        if os.environ.get("GITHUB_ACTIONS") or os.environ.get("INTOX"):
            return

        # Test custom SLURM cluster setup
        if useSLURM:

            # Supply extra args to start client for actual tests
            slurmOut = f"/mnt/hpc/home/{getpass.getuser()}/acme_out"
            inputArgs = {"partition" : defaultQ,
                         "n_workers" : 10,
                         "job_extra" : [f"--output={slurmOut}"],
                         "interactive" : False}
            if onESI:
                client = esi_cluster_setup(**inputArgs)
            elif onBIC:
                client = bic_cluster_setup(**inputArgs)
            else:
                return
            assert f"--output={slurmOut}" in client.cluster.job_header

        else:
            client = esi_cluster_setup(partition="auto", interactive=False)
        print("Allocated client")

        # Re-run tests with pre-allocated client (except for those in `skipTests`); ensure
        # client "survives" multiple independent test runs and is not accidentally closed
        skipTests = ["test_existing_cluster", "test_cancel", "test_dryrun",
                     "test_memest", "test_backcompat", "test_github_examples",
                     "_prep_data"]
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr not in skipTests)]
        for test in all_tests:
            print("Running test ", test)
            clnt = getattr(self, test)(testclient=client)
            assert clnt == client
        cluster_cleanup()
        if useSLURM:
            shutil.rmtree(slurmOut, ignore_errors=True)
