# -*- coding: utf-8 -*-
#
# Computational scaffolding for user-interface
#

# Builtin/3rd party package imports
import time
import getpass
import datetime
import inspect
import numbers
import os
import sys
import glob
import pickle
import logging
import functools
import tqdm
import h5py
import dask
import dask.distributed as dd
import dask_jobqueue as dj
import numpy as np

# Local imports
from . import __path__
from .dask_helpers import esi_cluster_setup, cluster_cleanup, _tag_client
from . import shared as acs
isSpyModule = False
if "syncopy" in sys.modules:
    isSpyModule = True

__all__ = ["ACMEdaemon"]


# Main context manager for parallel execution of user-defined functions
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
        partition="auto",
        mem_per_job="auto",
        setup_timeout=60,
        setup_interactive=True,
        stop_client="auto",
        verbose=None,
        logfile=None):
        """
        Coming soon...
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

        # Either use existing dask client or start a fresh instance
        self.prepare_client(n_jobs=n_jobs,
                            partition=partition,
                            mem_per_job=mem_per_job,
                            setup_timeout=setup_timeout,
                            setup_interactive=setup_interactive,
                            stop_client=stop_client)

    def initialize(self, func, argv, kwargv, n_calls):

        # Ensure `func` is callable
        if not callable(func):
            msg = "{} first input has to be a callable function, not {}"
            raise TypeError(msg.format(self.msgName, str(type(func))))

        # Next, vet `n_calls` which is needed to validate `argv` and `kwargv`
        try:
            acs._scalar_parser(n_calls, varname="n_calls", ntype="int_like", lims=[2, np.inf])
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

            # On the ESI cluster, save results on HPX, otherwise use location of `func`
            if self.has_slurm:
                outDir = "/mnt/hpx/home/{usr:s}/".format(usr=getpass.getuser())
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

    def prepare_client(
        self,
        n_jobs="auto",
        partition="auto",
        mem_per_job="auto",
        setup_timeout=180,
        setup_interactive=True,
        stop_client="auto"):

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
            self.client = esi_cluster_setup(interactive=False)

        else:

            # If `partition` is "auto", use `select_queue` to heuristically determine
            # the "best" SLURM queue
            if not isinstance(partition, str):
                msg = "{} `partition` has to be 'auto' or a valid SLURM partition name, not {}"
                raise TypeError(msg.format(self.msgName, str(partition)))
            if partition == "auto":
                msg = "Automatic SLURM queueing selection not implemented yet. " +\
                    "Falling back on default '8GBXS' partition. "
                self.log.warning(msg)
                partition = "8GBXS"
                # self.select_queue()

            # Either use `n_jobs = n_calls` (default) or parse provided value
            msg = "{} `n_jobs` has to be 'auto' or an integer >= 2, not {}"
            if isinstance(n_jobs, str):
                if n_jobs != "auto":
                    raise ValueError(msg.format(self.msgName, n_jobs))
                n_jobs = self.n_calls
            else:
                try:
                    acs._scalar_parser(n_jobs, varname="n_jobs", ntype="int_like", lims=[2, np.inf])
                except Exception as exc:
                    raise exc

            # All set, remaining input processing is done by `esi_cluster_setup`
            self.client = esi_cluster_setup(partition=partition, n_jobs=n_jobs,
                                            mem_per_job=mem_per_job, timeout=setup_timeout,
                                            interactive=setup_interactive, start_client=True)

            # If startup is aborted by user, get outta here
            if self.client is None:
                msg = "{} Could not start distributed computing client. "
                raise ConnectionAbortedError(msg.format(self.msgName))

        # Set `n_jobs` to no. of active workers in the initialized cluster
        self.n_jobs = len(self.client.cluster.workers)

    def select_queue(self):

        # FIXME: Very much WIP - everyting below is just a scratchpad, nothing final yet
        nSamples = min(self.n_calls, max(5, min(1, int(0.05*self.n_calls))))
        dryRunInputs = np.random.choice(self.n_calls, size=nSamples, replace=False).tolist()

        dryRun0 = dryRunInputs.pop()
        args = [arg[dryRun0] for arg in self.argv]
        kwargs = [{key:value[dryRun0] for key, value in self.kwargv.items()}][0]

        # use multi-processing module to launch `func` in background; terminate after
        # 60sec, get memory consumption, do this for all `dryRunInputs` -> pick
        # "shortest" queue w/appropriate memory (e.g., 16GBS, not 16GBXL)
        tic = time.perf_counter()
        self.func(*args, **kwargs)
        toc = time.perf_counter()

        # argBag = [[arg[k] for arg in self.argv] for k in range(self.n_calls)]
        # kwargBag = [{key:value[k] for key, value in self.kwargv.items()} for k in range(self.n_calls)]

    def compute(self, debug=False):

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
        if len(self.client.cluster.workers) == 0:
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
                ftArg = self.client.scatter(arg, broadcast=True)[0]
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
        if self.stop_client and self.client is not None:
            cluster_cleanup(self.client)
            self.client = None

    @staticmethod
    def func_wrapper(*args, **kwargs):

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
