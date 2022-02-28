# -*- coding: utf-8 -*-
#
# Testing module for ACME's `ParallelMap` interface
#

# Builtin/3rd party package imports
import os
import sys
import pickle
import shutil
import inspect
import subprocess
import getpass
import time
import itertools
import logging
import h5py
import pytest
import signal as sys_signal
import numpy as np
import dask.distributed as dd
from glob import glob
from scipy import signal

# Import main actors here
from acme import ParallelMap, cluster_cleanup, esi_cluster_setup
from acme.dask_helpers import customIOError
from acme.shared import is_slurm_node

# Construct decorators for skipping certain tests
skip_if_not_linux = pytest.mark.skipif(sys.platform != "linux", reason="Only works in Linux")

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

def pickle_func(arr, b, a, channel_no, sabotage_hdf5=False):
    res = signal.filtfilt(b, a, arr[:, channel_no], padlen=200)
    if sabotage_hdf5:
        if channel_no % 2 == 0:
            return {"b" : b}
    return res


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

        # Clean up testing folder and any running clients
        cluster_cleanup()
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)

    # Functionality tests: perform channel-concurrent low-pass filtering
    def test_filter_example(self):

        # If called by `test_existing_cluster` use pre-allocated client for all computations
        try:
            dd.get_client()
            existingClient = True
        except ValueError:
            existingClient = False

        # Create tmp directory and create data-containers
        tempDir = os.path.join(os.path.abspath(os.path.expanduser("~")), "acme_tmp")
        if useSLURM:
            tempDir = "/cs/home/{}/acme_tmp".format(getpass.getuser())
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

        # Compare computed single-channel results to expected low-freq signal
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
            assert np.mean(np.abs(resInMem[chNo] - self.orig[:, chNo])) < self.tol

        # Be double-paranoid: ensure on-disk and in-memory results match up
        for chNo, h5name in enumerate(resOnDisk):
            with h5py.File(h5name, "r") as h5f:
                assert np.array_equal(h5f["result_0"][()], resInMem[chNo])

        # Simulate user-defined results-directory
        tempDir2 = os.path.join(os.path.abspath(os.path.expanduser("~")), "acme_tmp_lowpass_hard")
        if useSLURM:
            tempDir2 = "/cs/home/{}/acme_tmp_lowpass_hard".format(getpass.getuser())
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

        # Compare computed single-channel results to expected low-freq signal
        for chNo in range(self.nChannels):
            h5name = res_base + "{}.h5".format(chNo)
            with h5py.File(os.path.join(tempDir2, h5name), "r") as h5f:
                assert np.mean(np.abs(h5f[dset_name][()] - self.orig[:, chNo])) < self.tol

        # Ensure log-file generation produces a non-empty log-file at the expected location
        # Bonus: leave computing client alive and vet default SLURM settings
        if not existingClient:
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
        if useSLURM and not existingClient:
            assert pmap.n_calls == pmap.n_jobs
            assert len(client.cluster.workers) == pmap.n_jobs
            partition = client.cluster.job_header.split("-p ")[1].split("\n")[0]
            assert "8GB" in partition
            memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
            assert memory.size == 1
            assert round(memory[0] / 1000**3) == [int(s) for s in partition if s.isdigit()][0]

        # Wait a sec (literally) for dask to collect its bearings (after the
        # `get_client` above) before proceeding
        time.sleep(1.0)

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
                         stop_client=not existingClient,
                         setup_interactive=False) as pmap:
            pmap.compute()
        outDirs.append(pmap.kwargv["outDir"][0])
        assert os.path.isfile(customLog)
        with open(customLog, "r") as fl:
            assert len(fl.readlines()) > 1

        # Ensure client has been stopped
        if not existingClient:
            with pytest.raises(ValueError):
                dd.get_client()

        # Wait a sec (literally) to give dask enough time to close the client
        time.sleep(1.0)

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
            actualPartition = client.cluster.job_header.split("-p ")[1].split("\n")[0]
            assert actualPartition == partition
            memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
            assert memory.size == 1
            assert round(memory[0] / 1000**3) == int(mem_per_job.replace("GB", ""))

        # Let `cluster_cleanup` murder the custom setup and ensure it did its job
        if not existingClient:
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
            actualPartition = client.cluster.job_header.split("-p ")[1].split("\n")[0]
            assert actualPartition == partition
            memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
            assert memory.size == 1
            assert round(memory[0] / 1000**3) * 1000 == int(mem_per_job.replace("MB", ""))
        if not existingClient:
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

        # Wait a second (literally) so that no new parallel jobs started by
        # `test_existing_cluster` erroneously use existing HDF files
        time.sleep(1.0)

    # Test if pickling/emergency pickling and I/O in general works as intended
    def test_pickling(self):

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
                         setup_interactive=False) as pmap:
            hdfResults = pmap.compute()
        outDirs.append(pmap.kwargv["outDir"][0])

        # Execute `pickle_func` w/pickling
        with ParallelMap(pickle_func,
                         self.sig,
                         self.b,
                         self.a,
                         range(self.nChannels),
                         n_inputs=self.nChannels,
                         write_pickle=True,
                         setup_interactive=False) as pmap:
            pklResults = pmap.compute()
        outDirs.append(pmap.kwargv["outDir"][0])

        # Ensure HDF5 and pickle match up
        for chNo, h5name in enumerate(hdfResults):
            with open(pklResults[chNo], "rb") as pkf:
                pklRes = pickle.load(pkf)
            with h5py.File(h5name, "r") as h5f:
                assert np.array_equal(pklRes, h5f["result_0"][()])

        # Test emergency pickling
        with ParallelMap(pickle_func,
                         self.sig,
                         self.b,
                         self.a,
                         range(self.nChannels),
                         sabotage_hdf5=True,
                         n_inputs=self.nChannels,
                         setup_interactive=False) as pmap:
            mixedResults = pmap.compute()
        outDirs.append(pmap.kwargv["outDir"][0])

        # Ensure non-compliant dicts were pickled, rest is in HDF5
        for chNo, fname in enumerate(mixedResults):
            if chNo % 2 == 0:
                assert fname.endswith(".pickle")
                with open(fname, "rb") as pkf:
                    assert np.array_equal(self.b, pickle.load(pkf)["b"])
            else:
                assert fname.endswith(".h5")
                with h5py.File(fname, "r") as h5f:
                    with h5py.File(hdfResults[chNo], "r") as h5ref:
                        assert np.array_equal(h5f["result_0"][()], h5ref["result_0"][()])

        # Test write breakdown (both for HDF5 saving and pickling)
        pmap = ParallelMap(pickle_func,
                           self.sig,
                           self.b,
                           self.a,
                           range(self.nChannels),
                           sabotage_hdf5=True,
                           n_inputs=self.nChannels,
                           setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap.kwargv["outDir"][0] = "/path/to/nowhere"
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
                           setup_interactive=False)
        outDirs.append(pmap.kwargv["outDir"][0])
        pmap.kwargv["outDir"][0] = "/path/to/nowhere"
        with pytest.raises(RuntimeError) as runerr:
            pmap.compute()
            assert "<ACMEdaemon> Parallel computation failed" in str(runerr.value)

        # Clean up testing folder
        for folder in outDirs:
            shutil.rmtree(folder, ignore_errors=True)

    # test if KeyboardInterrupts are handled correctly
    @skip_if_not_linux
    def test_cancel(self):

        # Setup temp-directory layout for subprocess-scripts and prepare interpreters
        tempDir = os.path.join(os.path.abspath(os.path.expanduser("~")), "acme_tmp")
        os.makedirs(tempDir, exist_ok=True)
        pshells = [os.path.join(os.path.split(sys.executable)[0], pyExec) for pyExec in ["python", "ipython"]]

        # Prepare ad-hoc script for execution in new process
        scriptName = os.path.join(tempDir, "dummy.py")
        scriptContents = \
            "from acme import ParallelMap\n" +\
            "import time\n" +\
            "def long_running(dummy):\n" +\
            "   time.sleep(10)\n" +\
            "   return\n" +\
            "if __name__ == '__main__':\n" +\
            "   with ParallelMap(long_running, [None]*2, setup_interactive=False, write_worker_results=False) as pmap: \n" +\
            "       pmap.compute()\n" +\
            "   print('ALL DONE')\n"
        with open(scriptName, "w") as f:
            f.write(scriptContents)

        # Execute the above script both in Python and iPython to ensure global functionality
        for pshell in pshells:

            # Launch new process in background (`stdbuf` prevents buffering of stdout)
            proc = subprocess.Popen("stdbuf -o0 " + pshell + " " + scriptName,
                                    shell=True, start_new_session=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

            # Wait for ACME to start up (as soon as logging info is shown, `pmap.compute()` is running)
            # However: don't wait indefinitely - if `pmap.compute` is not started within 30s, abort
            logStr = "<ParallelMap> INFO: Log information available at"
            buffer = bytearray()
            timeout = 30
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
            assert "INFO: <cluster_cleanup> Successfully shut down" in out

        # Almost identical script, this time use an externally started client
        scriptName = os.path.join(tempDir, "dummy2.py")
        scriptContents = \
            "from acme import ParallelMap, esi_cluster_setup\n" +\
            "import time\n" +\
            "def long_running(dummy):\n" +\
            "   time.sleep(10)\n" +\
            "   return\n" +\
            "if __name__ == '__main__':\n" +\
            "   client = esi_cluster_setup(partition='8GBDEV',n_jobs=1, interactive=False)\n" +\
            "   with ParallelMap(long_running, [None]*2, setup_interactive=False, write_worker_results=False) as pmap: \n" +\
            "       pmap.compute()\n" +\
            "   print('ALL DONE')\n"
        with open(scriptName, "w") as f:
            f.write(scriptContents)

        # Test script functionality in both Python and iPython
        for pshell in pshells:
            proc = subprocess.Popen("stdbuf -o0 " + sys.executable + " " + scriptName,
                                    shell=True, start_new_session=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            logStr = "<ParallelMap> INFO: Log information available at"
            buffer = bytearray()
            timeout = 30
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
            assert "<ParallelMap> INFO: <ACME> CTRL + C acknowledged, client and workers successfully killed" in out

        # Ensure random exception does not immediately kill an active client
        scriptName = os.path.join(tempDir, "dummy3.py")
        scriptContents = \
            "from acme import esi_cluster_setup\n" +\
            "import time\n" +\
            "if __name__ == '__main__':\n" +\
            "   esi_cluster_setup(partition='8GBDEV',n_jobs=1, interactive=False)\n" +\
            "   time.sleep(60)\n"
        with open(scriptName, "w") as f:
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

        # Ensure auto-generated output dir has been successfully removed
        outDir = pmap.kwargv["outDir"][0]
        assert os.path.exists(outDir) is False

    # test esi-cluster-setup called separately before pmap
    def test_existing_cluster(self):

        # Test custom SLURM cluster setup
        if useSLURM:

            # Ensure invalid partition/memory specifications are caught
            with pytest.raises(customIOError):
                esi_cluster_setup(partition="invalid", interactive=False)
            cluster_cleanup()
            with pytest.raises(ValueError):
                esi_cluster_setup(mem_per_job="invalidGB", interactive=False)
            cluster_cleanup()
            with pytest.raises(ValueError):
                esi_cluster_setup(mem_per_job="-20MB", interactive=False)
            cluster_cleanup()

            # Over-allocation of memory should default to partition max
            client = esi_cluster_setup(partition="8GBDEV", n_jobs=1, mem_per_job="9000MB", interactive=False)
            memory = np.unique([w["memory_limit"] for w in client.cluster.scheduler_info["workers"].values()])
            assert memory.size == 1
            assert np.round(memory / 1000**3)[0] == 8
            cluster_cleanup(client)

            # Test if invalid extra args are caught
            slurmOut = "/cs/home/{}/acme_out".format(getpass.getuser())
            with pytest.raises(TypeError):
                esi_cluster_setup(job_extra="--output={}".format(slurmOut), interactive=False)
            cluster_cleanup()
            with pytest.raises(ValueError):
                esi_cluster_setup(job_extra=["output={}".format(slurmOut)], interactive=False)
            cluster_cleanup()
            with pytest.raises(ValueError):
                esi_cluster_setup(job_extra=["--output=/path/to/nowhere"], interactive=False)
            cluster_cleanup()

            # Supply extra args to start client for actual tests
            client = esi_cluster_setup(partition="8GBXS", job_extra=["--output={}".format(slurmOut)], interactive=False)
            assert "--output={}".format(slurmOut) in client.cluster.job_header

        else:
            client = esi_cluster_setup(n_jobs=6, interactive=False)

        # Re-run tests with pre-allocated client (except for `test_cancel` and `test_dryrun`)
        skipTests = ["test_existing_cluster", "test_cancel", "test_dryrun"]
        all_tests = [attr for attr in self.__dir__()
                     if (inspect.ismethod(getattr(self, attr)) and attr not in skipTests)]
        for test in all_tests:
            getattr(self, test)()
        client.close()
        client.cluster.close()
        if useSLURM:
            shutil.rmtree(slurmOut, ignore_errors=True)
