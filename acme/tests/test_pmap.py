# -*- coding: utf-8 -*-
#
# Testing module for ACME's `ParallelMap` interface
#

# Builtin/3rd party package imports
import os
import h5py
import shutil
import inspect
import pytest
import logging
import numpy as np
import dask.distributed as dd
from glob import glob
from scipy import signal

# Import main actors here
from acme import ParallelMap, cluster_cleanup, esi_cluster_setup
from acme.shared import is_slurm_node

# Functions that act as stand-ins for user-funcs
def simple_func(x, y, z=3):
    return (x + y) * z

def medium_func(x, y, z=3, w=np.ones((3, 3))):
    return (sum(x) + y) * z * w.max()

def hard_func(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return sum(x) + y,  z * w

def lowpass_simple(h5name, channel_no):
    with h5py.File(h5name, "r") as h5f:
        channel = h5f["data"][:, channel_no]
        b = h5f["data"].attrs["b"]
        a = h5f["data"].attrs["a"]
    res = signal.filtfilt(b, a, channel, padlen=200)
    return res

def lowpass_hard(arr_like, b, a, res_dir, res_base="lowpass_hard_", dset_name="custom_dset_name", padlen=200, taskID=None):
    channel = arr_like[:, taskID]
    res = signal.filtfilt(b, a, channel, padlen=padlen)
    h5name = os.path.join(res_dir, res_base +"{}.h5".format(taskID))
    with h5py.File(h5name, "w") as h5f:
        h5f.create_dataset(dset_name, data=res)
    return


# Perform SLURM-specific tests only on cluster nodes
useSLURM = is_slurm_node()

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

        # Collected auto-generated output directories in list for later cleanup
        outDirs = []

        # Basic functionality w/simplest conceivable user-func
        pmap = ParallelMap(simple_func, [2, 4, 6, 8], 4, setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap = ParallelMap(simple_func, [2, 4, 6, 8], y=4, setup_interactive=False)  # pos arg referenced via kwarg, cfg #2
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap = ParallelMap(simple_func, 0, 4, z=[3, 4, 5, 6], setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap = ParallelMap(simple_func, [2, 4, 6, 8], [2, 2], n_inputs=2, setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])

        # User func has `np.ndarray` as keyword
        pmap = ParallelMap(medium_func, [2, 4, 6, 8], y=[2, 2], n_inputs=2, setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap = ParallelMap(medium_func, None, None, w=[np.ones((3, 3)), 2 * np.ones((3,3))], setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap = ParallelMap(medium_func, None, None, z=np.zeros((3,)), setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap = ParallelMap(medium_func, None, None, z=np.zeros((3, 1)), setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])

        # Lots of ways for this to go wrong...
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], 2, w=np.ones((3,)), setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], y=22, w=np.ones((7, 1)), setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap = ParallelMap(hard_func, np.ones((3,)), 1, w=np.ones((7, 1)), setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], z=np.array([1, 2]), w=np.ones((8, 1)), n_inputs=2, setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap = ParallelMap(hard_func, [2, 4, 6, 8], [2, 2], w=np.ones((8, 1)), n_inputs=4, setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])

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

        # Clean up testing folder
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)

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

        # Collected auto-generated output directories in list for later cleanup
        outDirs = []

        # Parallelize across channels, write results to disk
        with ParallelMap(lowpass_simple, sigName, range(self.nChannels), setup_interactive=False) as pmap:
            resOnDisk = pmap.compute()
        outDirs.append(pmap.kwargv["outDir"][0])
        assert len(pmap.kwargv["outFile"]) == pmap.n_calls
        resFiles = [os.path.join(pmap.kwargv["outDir"][0], outFile) for outFile in pmap.kwargv["outFile"]]
        assert resOnDisk == resFiles
        assert all(os.path.isfile(fle) for fle in resOnDisk)

        # Compare compuated single-channel results to expected low-freq signal
        for chNo, h5name in enumerate(resOnDisk):
            with h5py.File(h5name, "r") as h5f:
                assert np.mean(np.abs(h5f["result_0"][()] - self.orig[:, chNo])) < self.tol

        # Same, but collect results in memory: ensure nothing freaky happens
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         write_worker_results=False,
                         setup_interactive=False) as pmap:
            resInMem = pmap.compute()
        for chNo in range(self.nChannels):
            assert np.mean(np.abs(resInMem[chNo][0] - self.orig[:, chNo])) < self.tol

        # Simulate user-defined results-directory
        tempDir2 = os.path.join(os.path.abspath(os.path.expanduser("~")), "acme_tmp_lowpass_hard")
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
                         setup_interactive=False) as pmap:
            pmap.compute()
        resFiles = glob(os.path.join(tempDir2, res_base + "*"))
        assert len(resFiles) == pmap.n_calls

        # Compare compuated single-channel results to expected low-freq signal
        for chNo in range(self.nChannels):
            h5name = res_base + "{}.h5".format(chNo)
            with h5py.File(os.path.join(tempDir2, h5name), "r") as h5f:
                assert np.mean(np.abs(h5f[dset_name][()] - self.orig[:, chNo])) < self.tol

        # Ensure log-file generation produces a non-empty log-file at the expected location
        # Bonus: leave computing client alive and vet default SLURM settings
        cluster_cleanup(pmap.client)
        for handler in pmap.log.handlers:
            if isinstance(handler, logging.FileHandler):
                pmap.log.handlers.remove(handler)
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         logfile=True,
                         stop_client=False,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.kwargv["outDir"][0])
        logFileList = [handler.baseFilename for handler in pmap.log.handlers if isinstance(handler, logging.FileHandler)]
        assert len(logFileList) == 1
        logFile = logFileList[0]
        assert os.path.dirname(os.path.realpath(__file__)) in logFile
        with open(logFile, "r") as fl:
            assert len(fl.readlines()) > 1

        # Ensure client has not been killed; perform post-hoc check of default SLURM settings
        assert dd.get_client()
        client = dd.get_client()
        if useSLURM:
            assert pmap.n_calls == pmap.n_jobs
            assert len(client.cluster.workers) == pmap.n_jobs
            partition = client.cluster.workers[0].job_header.split("-p ")[1].split("\n")[0]
            assert "8GB" in partition
            memStr = client.cluster.workers[0].worker_process_memory
            assert int(float(memStr.replace("GB", ""))) == [int(s) for s in partition if s.isdigit()][0]

        # Same, but use custom log-file
        for handler in pmap.log.handlers:
            if isinstance(handler, logging.FileHandler):
                pmap.log.handlers.remove(handler)
        customLog = os.path.join(tempDir, "acme_log.txt")
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         logfile=customLog,
                         verbose=True,
                         stop_client=True,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.kwargv["outDir"][0])
        assert os.path.isfile(customLog)
        with open(customLog, "r") as fl:
            assert len(fl.readlines()) > 1

        # Ensure client has been stopped
        with pytest.raises(ValueError):
            dd.get_client()

        # Underbook SLURM (more calls than jobs)
        partition = "8GBXS"
        n_jobs = int(self.nChannels / 2)
        mem_per_job = "2GB"
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         partition=partition,
                         n_jobs=n_jobs,
                         mem_per_job=mem_per_job,
                         stop_client=False,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.kwargv["outDir"][0])

        # Post-hoc check of client to ensure custom settings were respected
        client = pmap.client
        assert pmap.n_calls == self.nChannels
        if useSLURM:
            assert pmap.n_jobs == n_jobs
            assert len(client.cluster.workers) == pmap.n_jobs
            actualPartition = client.cluster.workers[0].job_header.split("-p ")[1].split("\n")[0]
            assert actualPartition == partition
            memStr = client.cluster.workers[0].worker_process_memory
            assert int(float(memStr.replace("GB", ""))) == int(mem_per_job.replace("GB", ""))

        # Let `cluster_cleanup` murder the custom setup and ensure it did its job
        cluster_cleanup(pmap.client)
        with pytest.raises(ValueError):
            dd.get_client()

        # Overbook SLURM (more jobs than calls)
        partition = "8GBXS"
        n_jobs = self.nChannels + 2
        mem_per_job = "3000MB"
        with ParallelMap(lowpass_simple,
                         sigName,
                         range(self.nChannels),
                         partition=partition,
                         n_jobs=n_jobs,
                         mem_per_job=mem_per_job,
                         stop_client=False,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.kwargv["outDir"][0])

        # Post-hoc check of client to ensure custom settings were respected
        client = pmap.client
        assert pmap.n_calls == self.nChannels
        if useSLURM:
            assert pmap.n_jobs == n_jobs
            assert len(client.cluster.workers) == pmap.n_jobs
            actualPartition = client.cluster.workers[0].job_header.split("-p ")[1].split("\n")[0]
            assert actualPartition == partition
            memStr = client.cluster.workers[0].worker_process_memory
            assert int(float(memStr.replace("GB", ""))) * 1000 == int(mem_per_job.replace("MB", ""))
        cluster_cleanup(pmap.client)

        # Close any open HDF5 files to not trigger any `OSError`s, close running clusters
        # and clean up tmp dirs and created directories/log-files
        sigData.file.close()
        os.unlink(logFile)
        shutil.rmtree(tempDir, ignore_errors=True)
        shutil.rmtree(tempDir2, ignore_errors=True)
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)

    # test esi-cluster-setup called separately before pmap
    def test_existing_cluster(self):

        # Re-run tests with pre-allocated client
        client = esi_cluster_setup(partition="8GBXS", n_jobs=12, interactive=False)
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr != "test_existing_cluster")]
        for test in all_tests:
            getattr(self, test)()
        client.close()
        client.cluster.close()
