# -*- coding: utf-8 -*-
#
# Computational scaffolding for user-interface
#

# Builtin/3rd party package imports
import time
import socket
import getpass
import datetime
import inspect
import numbers
import collections
import os
import sys
import glob
import shutil
import pickle
import logging
import functools
import multiprocessing
import psutil
import tqdm
import h5py
import dask
import dask.distributed as dd
import dask_jobqueue as dj
import numpy as np

# Local imports
from . import __path__
from .dask_helpers import (esi_cluster_setup, local_cluster_setup,
                           slurm_cluster_setup, cluster_cleanup)
from .shared import user_yesno, is_esi_node
from . import shared as acs
isSpyModule = False
if "syncopy" in sys.modules:
    isSpyModule = True

__all__ = ["ACMEdaemon"]


# Main manager for parallel execution of user-defined functions
class ACMEdaemon(object):

    __slots__ = "func", "argv", "kwargv", "n_calls", "n_jobs", "acme_func", \
        "task_ids", "collect_results", "client", "stop_client", "has_slurm", "log"

    # Prepend every stdout/stderr message with the name of this class
    msgName = "<ACMEdaemon>"

    # format string for tqdm progress bars
    tqdmFormat = "{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

    # time (in seconds) b/w querying state of futures ('pending' -> 'finished')
    sleepTime = 0.1

    def __init__(
        self,
        pmap=None,
        func=None,
        argv=None,
        kwargv=None,
        n_calls=None,
        n_jobs="auto",
        write_worker_results=True,
        write_pickle=False,
        dryrun=False,
        partition="auto",
        mem_per_job="auto",
        setup_timeout=60,
        setup_interactive=True,
        stop_client="auto",
        verbose=None,
        logfile=None):
        """
        Manager class for performing concurrent user function calls

        Parameters
        ----------
        pmap : :class:~`acme.ParallelMap` context manager or None
            If `pmap` is not `None`, `:class:~`acme.ACMEDaemon` assumes
            that that the provided :class:~`acme.ParallelMap` instance has already
            been properly set up to process `func` (all input arguments parsed and
            properly formatted). All other input arguments of `:class:~`acme.ACMEDaemon`
            are extracted from the provided :class:~`acme.ParallelMap` instance.
            If `pmap` is `None`, `:class:~`acme.ACMEDaemon` runs in "stand-alone"
            mode: all remaining arguments have to be manually supplied (in the
            correct format)
        func : callable
            User-defined function to be executed concurrently. See :class:~`acme.ParallelMap`
            for details.
        argv : list of lists
            Positional arguments of `func`: all elements of have to be list-like
            with lengths `n_calls` or 1
        kwargv : list of dicts
            Keyword arguments of `func`: all values of have to be list-like with
            lengths `n_calls` or 1
        n_calls : int
            Number of concurrent calls of `func` to perform. If `pmap` is not `None`,
            then ``n_calls = pmap.n_inputs``
        n_jobs : int or "auto"
            Number of SLURM jobs (=workers) to spawn. See :class:~`acme.ParallelMap`
            for details.
        write_worker_results : bool
            If `True`, the return value(s) of `func` is/are saved on disk. See
            :class:~`acme.ParallelMap` for details.
        write_pickle : bool
            If `True`, the return value(s) of `func` is/are pickled to disk. See
            :class:~`acme.ParallelMap` for details.
        dryrun : bool
            If `True`, a dry-run of calling `func` is performed using a single
            `args`, `kwargs` tuple. See :class:~`acme.ParallelMap` for details.
        partition : str
            Name of SLURM partition to use. See :class:~`acme.ParallelMap` for details.
        mem_per_job : str
            Memory booking for each SLURM worker. See :class:~`acme.ParallelMap` for details.
        setup_timeout : int
            Timeout period (in seconds) for SLURM workers to come online. See
            :class:~`acme.ParallelMap` for details.
        setup_interactive : bool
            If `True`, user input is queried in case not enough SLURM workers could
            be started within `setup_timeout` seconds. See :class:~`acme.ParallelMap`
            for details.
        stop_client : bool or "auto"
            If `"auto"`, automatically started distributed computing clients
            are shut down at the end of computation, while user-provided clients
            are left untouched. See :class:~`acme.ParallelMap` for details.
        verbose : None or bool
            If `None` (default), general run-time information as well as warnings
            and errors are shown. See :class:~`acme.ParallelMap` for details.
        logfile : None or bool or str
            If `None` (default) or `False`, all run-time information as well as errors and
            warnings are printed to the command line only. See :class:~`acme.ParallelMap`
            for details.

        Returns
        -------
        results : list
            If `write_worker_results` is `True`, `results` is a list of HDF5 file-names
            containing computed results. If `write_worker_results` is `False`,
            results is a list comprising the actual return values of `func`.
            If `:class:~`acme.ACMEDaemon` was instantiated by :class:~`acme.ParallelMap`,
            results are propagated back to :class:~`acme.ParallelMap`.

        See also
        --------
        ParallelMap : Context manager and main user interface
        """

        # The only error checking happening in `__init__`
        if pmap is not None:
            if pmap.__class__.__name__ != "ParallelMap":
                msg = "{} `pmap` has to be a `ParallelMap` instance, not {}"
                raise TypeError(msg.format(self.msgName, str(pmap)))

        # Input pre-processed by a `ParallelMap` object takes precedence over keyword args
        self.initialize(getattr(pmap, "func", func),
                        getattr(pmap, "argv", argv),
                        getattr(pmap, "kwargv", kwargv),
                        getattr(pmap, "n_inputs", n_calls))

        # If `log` is `None`, `prepare_log` has not been called yet
        if getattr(pmap, "log", None) is None:
            self.log = acs.prepare_log(func, caller=self.msgName, logfile=logfile,
                                       verbose=verbose)
        else:
            self.log = pmap.log

        # Set up output handler
        self.prepare_output(write_worker_results, write_pickle)

        # If requested, perform single-worker dry-run (and quit if desired)
        if dryrun:
            goOn = self.perform_dryrun(setup_interactive)
            if not goOn:
                return

        # Either use existing dask client or start a fresh instance
        self.prepare_client(n_jobs=n_jobs,
                            partition=partition,
                            mem_per_job=mem_per_job,
                            setup_timeout=setup_timeout,
                            setup_interactive=setup_interactive,
                            stop_client=stop_client)

    def initialize(self, func, argv, kwargv, n_calls):
        """
        Parse (provided) inputs: make sure positional and keyword args are
        properly formatted for concurrently calling `func`
        """

        # Ensure `func` is callable
        if not callable(func):
            msg = "{} first input has to be a callable function, not {}"
            raise TypeError(msg.format(self.msgName, str(type(func))))

        # Next, vet `n_calls` which is needed to validate `argv` and `kwargv`
        try:
            acs._scalar_parser(n_calls, varname="n_calls", ntype="int_like", lims=[1, np.inf])
        except Exception as exc:
            raise exc

        # Ensure all elements of `argv` are list-like with lengths `n_calls` or 1
        msg = "{} `argv` has to be a list with list-like elements of length 1 or {}"
        if not isinstance(argv, (list, tuple)):
            raise TypeError(msg.format(self.msgName, n_calls))
        try:
            validArgv = all(len(arg) == n_calls or len(arg) == 1 for arg in argv)
        except TypeError:
            raise TypeError(msg.format(self.msgName, n_calls))
        if not validArgv:
            raise ValueError(msg.format(self.msgName, n_calls))

        # Ensure all values of `kwargv` are list-like with lengths `n_calls` or 1
        msg = "{} `kwargv` has to be a dictionary with list-like elements of length {}"
        try:
            validKwargv = all(len(value) == n_calls or len(value) == 1 for value in kwargv.values())
        except TypeError:
            raise TypeError(msg.format(self.msgName, n_calls))
        if not validKwargv:
            raise ValueError(msg.format(self.msgName, n_calls))

        # Basal sanity checks have passed, keep the provided input signature
        self.func = func
        self.argv = argv
        self.kwargv = kwargv
        self.n_calls = n_calls

        # Define list of taskIDs for distribution across workers
        self.task_ids = list(range(n_calls))

        # Finally, determine if the code is executed on a SLURM-enabled node
        self.has_slurm = acs.is_slurm_node()

    def prepare_output(self, write_worker_results, write_pickle):
        """
        If `write_*` is `True` set up directories for saving output HDF5 containers
        (or pickle files). Warn if results are to be collected in memory
        """

        # Basal sanity check for Boolean flags
        if not isinstance(write_worker_results, bool):
            msg = "{} `write_worker_results` has to be `True` or `False`, not {}"
            raise TypeError(msg.format(self.msgName, str(write_worker_results)))
        if not isinstance(write_pickle, bool):
            msg = "{} `write_pickle` has to be `True` or `False`, not {}"
            raise TypeError(msg.format(self.msgName, str(write_pickle)))
        if not write_worker_results and write_pickle:
            self.log.warning("Pickling of results only possible if `write_worker_results` is `True`. ")

        # If automatic saving of results is requested, make necessary preparations
        if write_worker_results:

            # On the ESI cluster, save results on HPC mount, otherwise use location of `func`
            if self.has_slurm:
                outDir = "/cs/home/{usr:s}/".format(usr=getpass.getuser())
            else:
                outDir = os.path.dirname(os.path.abspath(inspect.getfile(self.func)))
            outDir = os.path.join(outDir, "ACME_{date:s}")
            outDir = outDir.format(date=datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            try:
                os.makedirs(outDir)
            except Exception as exc:
                msg = "{} automatic creation of output folder {} failed. Original error message below:\n{}"
                raise OSError(msg.format(self.msgName, outDir, str(exc)))

            # Prepare `outDir` for distribution across workers via `kwargv` and
            # re-define or allocate key "taskID" to track concurrent processing results
            self.kwargv["outDir"] = [outDir] * self.n_calls
            self.kwargv["taskID"] = self.task_ids
            self.collect_results = False
            fExt = "h5"
            if write_pickle:
                fExt = "pickle"
            self.kwargv["outFile"] = ["{}_{}.{}".format(self.func.__name__, taskID, fExt) for taskID in self.task_ids]

            # Include logger name in keywords so that workers can use it
            self.kwargv["logName"] = [self.log.name] * self.n_calls

            # Wrap the user-provided func and distribute it across workers
            self.kwargv["userFunc"] = [self.func] * self.n_calls
            self.acme_func = self.func_wrapper

        else:

            # If `taskID` is not an explicit kw-arg of `func` and `func` does not
            # accept "anonymous" `**kwargs`, don't save anything but return stuff
            if self.kwargv.get("taskID") is None:
                if not isSpyModule:
                    msg = "`write_worker_results` is `False` and `taskID` is not a keyword argument of {}. " +\
                        "Results will be collected in memory by caller - this might be slow and can lead " +\
                        "to excessive memory consumption. "
                    self.log.warning(msg.format(self.func.__name__))
                self.collect_results = True
            else:
                self.kwargv["taskID"] = self.task_ids
                self.collect_results = False

            # The "raw" user-provided function is used in the computation
            self.acme_func = self.func

    def perform_dryrun(self, setup_interactive):
        """
        Execute user function with one prepared randomly picked args, kwargs combo
        """

        # Let helper randomly pick a single scheduled job and prepare corresponding args + kwargs
        [dryRunIdx], [dryRunArgs], [dryRunKwargs] = self._dryrun_setup(n_runs=1)

        # Create log entry
        msg = "Performing a single dry-run of {fname:s} simulating randomly " +\
            "picked worker #{wrknum:d} with automatically distributed arguments"
        self.log.info(msg.format(fname=self.func.__name__, wrknum=dryRunIdx))

        # Use resident memory size (in MB) to estimate job's memory footprint and measure elapsed time
        mem0 = psutil.Process().memory_info().rss / 1024 ** 2
        tic = time.perf_counter()
        self.acme_func(*dryRunArgs, **dryRunKwargs)
        toc = time.perf_counter()
        mem1 = psutil.Process().memory_info().rss / 1024 ** 2

        # Remove any generated output files
        if "outDir" in self.kwargv.keys():
            fname = os.path.join(self.kwargv["outDir"][dryRunIdx], self.kwargv["outFile"][dryRunIdx])
            os.unlink(fname)

        # Compute elapsed time and memory usage
        elapsedTime = toc - tic
        memUsage = mem1 - mem0

        # Prepare info message
        memUnit = "MB"
        if memUsage > 1000:
            memUsage /= 1024
            memUnit = "GB"
        msg = "Dry-run completed. Elapsed time is {runtime:f} seconds, " +\
            "estimated memory consumption was {memused:3.2f} {memunit:s}."
        self.log.info(msg.format(runtime=elapsedTime, memused=memUsage, memunit=memUnit))

        # If the worker setup is supposed to be interactive, ask for confirmation
        # here as well; if execution is terminated, remove auto-generated output directory
        goOn = True
        if setup_interactive:
            msg = "Do you want to continue executing {fname:s} with the provided arguments?"
            if not user_yesno(msg.format(fname=self.func.__name__), default="yes"):
                if "outDir" in self.kwargv.keys():
                    shutil.rmtree(self.kwargv["outDir"][dryRunIdx], ignore_errors=True)
                goOn = False

        return goOn

    def _dryrun_setup(self, n_runs=None):
        """
        Pick scheduled job(s) at random and extract corresponding (already prepared!) args + kwargs
        """

        # If not provided, attempt to infer a "sane" default for the number of jobs to pick
        if n_runs is None:
            n_runs = min(self.n_calls, max(5, min(1, int(0.05 * self.n_calls))))

        # Randomly pick `n_runs` jobs and extract positional and keyword args
        dryRunIdx = np.random.choice(self.n_calls, size=n_runs, replace=False)
        dryRunArgs = []
        dryRunKwargs = []
        for idx in dryRunIdx:
            dryRunArgs.append([arg[idx] if len(arg) > 1 else arg[0] for arg in self.argv])
            dryRunKwargs.append([{key:value[idx] if len(value) > 1 else value[0] \
                for key, value in self.kwargv.items()}][0])
        return dryRunIdx, dryRunArgs, dryRunKwargs

    def prepare_client(
        self,
        n_jobs="auto",
        partition="auto",
        mem_per_job="auto",
        setup_timeout=180,
        setup_interactive=True,
        stop_client="auto"):
        """
        Setup or fetch dask distributed processing client. Depending on available
        hardware, either start a local multi-processing client or launch a
        worker cluster via SLURM.

        Also ensure that ad-hoc clients created here are stopped and worker jobs
        are properly released at the end of computation. However, ensure any client
        not created by `prepare_client` is **not** automatically cleaned up.
        """

        # Modify automatic setting of `stop_client` if requested
        msg = "{} `stop_client` has to be 'auto' or Boolean, not {}"
        if isinstance(stop_client, str):
            if stop_client != "auto":
                raise ValueError(msg.format(self.msgName, stop_client))
        elif isinstance(stop_client, bool):
            self.stop_client = stop_client
        else:
            raise TypeError(msg.format(self.msgName, stop_client))

        # Check if a dask client is already running
        try:
            self.client = dd.get_client()
            if stop_client == "auto":
                self.stop_client = False
            self.n_jobs = len(self.client.cluster.workers)
            msg = "Attaching to global parallel computing client {}"
            self.log.info(msg.format(str(self.client)))
            return
        except ValueError:
            if stop_client == "auto":
                self.stop_client = True

        # If things are running locally, simply fire up a dask-distributed client,
        # otherwise go through the motions of preparing a full cluster job swarm
        if not self.has_slurm:
            self.client = local_cluster_setup(interactive=False)

        else:

            # If `partition` is "auto", use `estimate_memuse` to heuristically determine
            # average memory consumption of jobs
            if not isinstance(partition, str):
                msg = "{} `partition` has to be 'auto' or a valid SLURM partition name, not {}"
                raise TypeError(msg.format(self.msgName, str(partition)))
            if partition == "auto":
                if is_esi_node():
                    msg = "Automatic SLURM partition selection is experimental"
                    self.log.warning(msg)
                    mem_per_job = self.estimate_memuse()
                else:
                    err = "Automatic SLURM partition selection currently only available " +\
                        "on the ESI HPC cluster. "
                    self.log.error(err)

            # If `n_jobs` is `"auto`, set `n_jobs = n_calls` (default)
            msg = "{} `n_jobs` has to be 'auto' or an integer >= 2, not {}"
            if isinstance(n_jobs, str):
                if n_jobs != "auto":
                    raise ValueError(msg.format(self.msgName, n_jobs))
                n_jobs = self.n_calls

            # All set, remaining input processing is done by respective `*_cluster_setup` routines
            if is_esi_node():
                self.client = esi_cluster_setup(partition=partition, n_jobs=n_jobs,
                                                mem_per_job=mem_per_job, timeout=setup_timeout,
                                                interactive=setup_interactive, start_client=True)

            # Unknown cluster node, use vanilla config
            else:
                wrng = "Cluster node {} not recognized. Falling back to vanilla " +\
                    "SLURM setup allocating one worker and one core per job"
                self.log.warning(wrng.format(socket.getfqdn()))
                workers_per_job = 1
                n_cores = 1
                self.client = slurm_cluster_setup(partition=partition,
                                                  n_cores=n_cores,
                                                  n_jobs=n_jobs,
                                                  workers_per_job=workers_per_job,
                                                  mem_per_job=mem_per_job,
                                                  n_jobs_startup=100,
                                                  timeout=setup_timeout,
                                                  interactive=setup_interactive,
                                                  interactive_wait=120,
                                                  start_client=True,
                                                  job_extra=[],
                                                  invalid_partitions=[])

            # If startup is aborted by user, get outta here
            if self.client is None:
                msg = "{} Could not start distributed computing client. "
                raise ConnectionAbortedError(msg.format(self.msgName))

        # Set `n_jobs` to no. of active workers in the initialized cluster
        self.n_jobs = len(self.client.cluster.workers)

    def estimate_memuse(self):
        """
        A brute-force guessing approach to determine memory consumption of provided
        workload
        """

        # Let helper randomly pick some jobs and prepare corresponding args + kwargs
        dryRunIdx, dryRunArgs, dryRunKwargs = self._dryrun_setup()

        # Set run-time for each job (in seconds) and allocate arrays for memory
        # consumption estimates (in GB)
        runTime = 30
        memPerSec = np.zeros((runTime,))
        memPerJob = np.zeros((len(dryRunIdx),))

        # Check if auto-generated output files have to be removed
        rmOutDir = False
        if "outDir" in self.kwargv.keys():
            rmOutDir = True

        # Adequately warn about this heuristic gymnastics...
        msg = "Estimating memory consumption of {fname:s} by running {numwrks:d} " +\
            "random workers for {rtime:d} seconds..."
        self.log.info(msg.format(fname=self.func.__name__, numwrks=len(dryRunIdx), rtime=runTime))
        wmsg = "Launching worker #{wrknum:d}"

        for i, idx in enumerate(dryRunIdx):

            # Set up dedicated process to execute user-provided function w/allocated args/kwargs
            proc = multiprocessing.Process(target=self.acme_func,
                                           args=dryRunArgs[i],
                                           kwargs=dryRunKwargs[i])

            # Run user-func for `runTime` seconds, get memory footprint every second
            proc.start()
            with tqdm.tqdm(desc=wmsg.format(wrknum=idx),
                           total=runTime,
                           bar_format=self.tqdmFormat,
                           position=0) as pbar:
                for k in range(runTime):
                    memPerSec[k] = psutil.Process(proc.pid).memory_info().rss / 1024 ** 3
                    time.sleep(1)
                    pbar.update(1)
            proc.kill()

            # Compute peak memory consumption across `runTime` seconds
            memPerJob[i] = memPerSec.max()

            # Remove auto-generated files (if any were created within `runTime` seconds)
            if rmOutDir:
                fname = os.path.join(self.kwargv["outDir"][idx], self.kwargv["outFile"][idx])
                if os.path.isfile(fname):
                    os.unlink(fname)

        # Compute aggregate average memory consumption across all runs
        memUsage = memPerJob.mean()

        # Communicate results
        msg = "Estimated memory consumption across {numwrks:d} runs is {memuse:3.2f} GB "
        self.log.info(msg.format(numwrks=len(dryRunIdx), memuse=memUsage))

        return "estimate_memuse:" + str(max(1, int(round(memUsage))))

    def compute(self, debug=False):
        """
        Perform the actual parallel execution of `func`

        If `debug` is `True`, use a single-threaded dask scheduler that does
        not actually process anythin concurrently but uses the dask framework
        in a sequential setup.
        """

        # If `prepare_client` has not been called yet, don't attempt to compute anything
        if not hasattr(self, "client"):
            return

        # Ensure `debug` is a simple Boolean flag
        if not isinstance(debug, bool):
            msg = "{} `debug` has to be `True` or `False`, not {}"
            raise TypeError(msg.format(self.msgName, str(debug)))

        # If `client` attribute is not set, the daemon is being re-used: prepare
        # everything for re-entry
        if self.client is None:
            write_worker_results = self.acme_func == self.func_wrapper
            if write_worker_results:
                write_pickle = self.kwargv["outFile"][0].endswith(".pickle")
            else:
                write_pickle = False
            self.prepare_output(write_worker_results=write_worker_results,
                                write_pickle=write_pickle)
            self.prepare_client(n_jobs=self.n_jobs, stop_client=self.stop_client)

        # Check if the underlying parallel computing cluster hosts actually usable workers
        if len([w["memory_limit"] for w in self.client.cluster.scheduler_info["workers"].values()]) == 0:
            msg = "{} no active workers found in distributed computing cluster {} " +\
                "Consider running \n" +\
                "\timport dask.distributed as dd; dd.get_client().restart()\n" +\
                "If this fails to make workers come online, please use\n" +\
                "\timport acme; acme.cluster_cleanup()\n" +\
                "to shut down any defunct distributed computing clients"
            raise RuntimeError(msg.format(self.msgName, self.client))

        # Dask does not correctly forward the `sys.path` from the parent process
        # to its workers. Fix this.
        def init_acme(dask_worker, syspath):
            sys.path = list(syspath)
        self.client.register_worker_callbacks(setup=functools.partial(init_acme, syspath=sys.path))

        # Format positional arguments for worker-distribution: broadcast all
        # inputs that are used by all workers and create a list of references
        # to this (single!) future on the cluster for submission
        for ak, arg in enumerate(self.argv):
            if len(arg) == 1:
                ftArg = self.client.scatter(arg, broadcast=True)
                if isinstance(ftArg, collections.abc.Sized):
                    ftArg = ftArg[0]
                self.argv[ak] = [ftArg] * self.n_calls

        # Same as above but for keyword-arguments
        for name, value in self.kwargv.items():
            if len(value) == 1:
                ftVal = self.client.scatter(value, broadcast=True)[0]
                self.kwargv[name] = [ftVal] * self.n_calls

        # Re-format keyword arguments to be usable with single-to-many arg submission.
        # Idea: with `self.n_calls = 3` and ``self.kwargv = {'a': [5, 5, 5], 'b': [6, 6, 6]}``
        # then ``kwargList = [{'a': 5, 'b': 6}, {'a': 5, 'b': 6}, {'a': 5, 'b': 6}]``
        kwargList = []
        kwargKeys = self.kwargv.keys()
        kwargVals = list(self.kwargv.values())
        for nc in range(self.n_calls):
            kwDict = {}
            for kc, key in enumerate(kwargKeys):
                kwDict[key] = kwargVals[kc][nc]
            kwargList.append(kwDict)

        # In case a debugging run is performed, use the single-threaded scheduler and return
        if debug:
            with dask.config.set(scheduler='single-threaded'):
                values = self.client.gather([self.client.submit(self.acme_func, *args, **kwargs) \
                    for args, kwargs in zip(zip(*self.argv), kwargList)])
                return values

        # Depending on the used dask cluster object, point to respective log info
        if isinstance(self.client.cluster, dj.SLURMCluster):
            logFiles = self.client.cluster.job_header.split("--output=")[1].replace("%j", "{}")
            logDir = os.path.split(logFiles)[0]
        else:
            logFiles = []
            logDir = os.path.dirname(self.client.cluster.dashboard_link) + "/info/main/workers.html"
        msg = "Preparing {} parallel calls of `{}` using {} workers"
        self.log.info(msg.format(self.n_calls, self.func.__name__, self.n_jobs))
        msg = "Log information available at {}"
        self.log.info(msg.format(logDir))

        # Submit `self.n_calls` function calls to the cluster
        futures = [self.client.submit(self.acme_func, *args, **kwargs) \
            for args, kwargs in zip(zip(*self.argv), kwargList)]

        # Set up progress bar: the while loop ensures all futures are executed
        totalTasks = len(futures)
        pbar = tqdm.tqdm(total=totalTasks, bar_format=self.tqdmFormat, position=0, leave=True)
        cnt = 0
        while any(f.status == "pending" for f in futures):
            time.sleep(self.sleepTime)
            new = max(0, sum([f.status == "finished" for f in futures]) - cnt)
            cnt += new
            pbar.update(new)
        pbar.close()

        # Avoid race condition: give futures time to perform switch from 'pending'
        # to 'finished' so that `finishedTasks` is computed correctly
        time.sleep(self.sleepTime)

        # If number of 'finished' tasks is less than expected, go into
        # problem analysis mode: all futures that erred hav an `.exception`
        # method which can be used to track down the worker it was executed by
        # Once we know the worker, we can point to the right log file. If
        # futures were cancelled (by the user or the SLURM controller),
        # `.exception` is `None` and we can't reliably track down the
        # respective executing worker
        finishedTasks = sum([f.status == "finished" for f in futures])
        if finishedTasks < totalTasks:
            schedulerLog = list(self.client.cluster.get_logs(cluster=False, scheduler=True, workers=False).values())[0]
            erredFutures = [f for f in futures if f.status == "error"]
            msg = "{} Parallel computation failed: {}/{} tasks failed or stalled.\n"
            msg = msg.format(self.msgName, totalTasks - finishedTasks, totalTasks)
            msg += "Concurrent computing scheduler log below: \n\n"
            msg += schedulerLog + "\n"

            # If we're working w/`SLURMCluster`, perform the Herculean task of
            # tracking down which dask worker was executed by which SLURM job...
            if self.client.cluster.__class__.__name__ == "SLURMCluster":
                try:
                    erredJobs = [f.exception().last_worker.identity()["id"] for f in erredFutures]
                except AttributeError:
                    erredJobs = []
                erredJobs = list(set(erredJobs))
                validIDs = [job for job in erredJobs if job in self.client.cluster.workers.keys()]
                erredJobIDs = [self.client.cluster.workers[job].job_id for job in validIDs]
                errFiles = glob.glob(logDir + os.sep + "*.err")
                if len(erredFutures) > 0 or len(errFiles) > 0:
                    msg += "Please consult the following SLURM log files for details:\n"
                    if len(erredJobIDs) > 0:
                        msg += "".join(logFiles.format(id) + "\n" for id in erredJobIDs)
                    else:
                        msg += "".join(logDir)
                    msg += "".join(errfile + "\n" for errfile in errFiles)
                else:
                    msg += "Please check SLURM logs in {}".format(logDir)

            # In case of a `LocalCluster`, syphon worker logs
            else:
                msg += "\nParallel worker logs below: \n"
                workerLogs = self.client.get_worker_logs().values()
                for wLog in workerLogs:
                    if "Failed" in wLog:
                        msg += wLog

            # Finally, raise an error and get outta here
            raise RuntimeError(msg)

        # If wanted (not recommended) collect computed results in local memory
        if self.collect_results:
            if not isSpyModule:
                self.log.info("Gathering results in local memory")
            values = self.client.gather(futures)
        else:
            values = None

        # Assemble final triumphant output message and get out
        msg = "SUCCESS! Finished parallel computation. "
        if "outDir" in self.kwargv.keys():
            dirname = self.kwargv["outDir"][0]
            msgRes = "Results have been saved to {}".format(dirname)
            msg += msgRes
            # Determine filepaths of results files (query disk to catch emergency pickles)
            if values is None:
                values = []
                for fname in self.kwargv["outFile"]:
                    h5Name = os.path.join(dirname, fname)
                    pklName = h5Name.rstrip(".h5") + ".pickle"
                    if os.path.isfile(h5Name):
                        values.append(h5Name)
                    elif os.path.isfile(pklName):
                        values.append(pklName)
                    else:
                        values.append("Missing {}".format(fname.rstrip(".h5")))
        self.log.info(msg)

        # Either return collected by-worker results or the filepaths of results
        return values

    def cleanup(self):
        """
        Shut down any ad-hoc distributed computing clients created by `prepare_client`
        """

        # If `prepare_client` has not been launched yet, just get outta here
        if not hasattr(self, "client"):
            return
        if self.stop_client and self.client is not None:
            cluster_cleanup(self.client)
            self.client = None

    @staticmethod
    def func_wrapper(*args, **kwargs):
        """
        If the output of `func` is saved to disk, wrap `func` with this static
        method to take care of filling up HDF5/pickle files

        If writing to HDF5 fails, use an "emergency-pickling" mechanism to try
        to save the output of `func` using pickle instead
        """

        # Extract everything from `kwargs` appended by `ACMEdaemon`
        func = kwargs.pop("userFunc")
        outDir = kwargs.pop("outDir")
        taskID = kwargs.pop("taskID")
        fname = kwargs.pop("outFile")
        logName = kwargs.pop("logName")
        log = logging.getLogger(logName)

        # Call user-provided function
        result = func(*args, **kwargs)

        # Save results: either (try to) use HDF5 or pickle stuff
        if fname.endswith(".h5"):
            try:
                h5name = os.path.join(outDir, fname)
                with h5py.File(h5name, "w") as h5f:
                    if isinstance(result, (list, tuple)):
                        if not all(isinstance(value, (numbers.Number, str)) for value in result):
                            for rk, res in enumerate(result):
                                h5f.create_dataset("result_{}".format(rk), data=res)
                        else:
                            h5f.create_dataset("result_0", data=result)
                    else:
                        h5f.create_dataset("result_0", data=result)
            except TypeError as exc:
                if "has no native HDF5 equivalent" in str(exc) or "One of data, shape or dtype must be specified" in str(exc):
                    try:
                        os.unlink(h5name)
                        pname = fname.rstrip(".h5") + ".pickle"
                        with open(os.path.join(outDir, pname), "wb") as pkf:
                            pickle.dump(result, pkf)
                        msg = "Could not write %s results have been pickled instead: %s. Return values are most likely " +\
                            "not suitable for storage in HDF5 containers. Original error message: %s"
                        log.warning(msg, fname, pname, str(exc))
                    except pickle.PicklingError as pexc:
                        err = "Unable to write %s, successive attempts to pickle results failed too: %s"
                        log.error(err, fname, str(pexc))
                else:
                    err = "Could not access %s. Original error message: %s"
                    log.error(err, h5name, str(exc))
                    raise exc
        else:
            try:
                with open(os.path.join(outDir, fname), "wb") as pkf:
                    pickle.dump(result, pkf)
            except pickle.PicklingError as pexc:
                err = "Could not pickle results to file %s. Original error message: %s"
                log.error(err, fname, str(pexc))
                raise pexc
