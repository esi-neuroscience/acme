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
import h5py
import glob
import tqdm
import functools
import dask.distributed as dd
import dask_jobqueue as dj
import dask.bag as db
import numpy as np

# Local imports
from . import __path__
from .dask_helpers import esi_cluster_setup, cluster_cleanup
from . import shared as acs

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
        self.prepare_output(write_worker_results)

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

        # Ensure all elements of `argv` are list-like with length `n_calls`
        msg = "{} `argv` has to be a list with list-like elements of length {}"
        if not isinstance(argv, (list, tuple)):
            raise TypeError(msg.format(self.msgName, n_calls))
        try:
            validArgv = all(len(arg) == n_calls for arg in argv)
        except TypeError:
            raise TypeError(msg.format(self.msgName, n_calls))
        if not validArgv:
            raise ValueError(msg.format(self.msgName, n_calls))

        # Ensure all keys of `kwargv` have length `n_calls`
        msg = "{} `kwargv` has to be a dictionary with list-like elements of length {}"
        try:
            validKwargv = all(len(value) == n_calls for value in kwargv.values())
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

    def prepare_output(self, write_worker_results):

        # Basal sanity check for Boolean flag
        if not isinstance(write_worker_results, bool):
            msg = "{} `write_worker_results` has to be `True` or `False`, not {}"
            raise TypeError(msg.format(self.msgName, str(write_worker_results)))

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
            self.kwargv["outFile"] = ["{}_{}.h5".format(self.func.__name__, taskID) for taskID in self.task_ids]
            self.kwargv["taskID"] = self.task_ids
            self.collect_results = False

            # Wrap the user-provided func and distribute it across workers
            self.kwargv["userFunc"] = [self.func] * self.n_calls
            self.acme_func = self.func_wrapper

        else:

            # If `taskID` is not an explicit kw-arg of `func` and `func` does not
            # accept "anonymous" `**kwargs`, don't save anything but return stuff
            if self.kwargv.get("taskID") is None \
                and "kwargs" not in inspect.signature(self.func).parameters.keys():
                msg = "`write_worker_results` is `False` and `taskID` is not a keyword argument of {}." +\
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
                    "Falling back on default '8GBS' partition. "
                self.log.warning(msg)
                partition = "8GBS"
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
            self.prepare_output(write_worker_results=self.acme_func == self.func_wrapper)
            self.prepare_client(n_jobs=self.n_jobs, stop_client=self.stop_client)

        # Check if the underlying parallel computing cluster hosts actually usable workers
        if not len(self.client.cluster.workers):
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

        # Convert positional/keyword arg lists to dask bags
        firstArg = db.from_sequence(self.argv[0], npartitions=self.n_calls)
        otherArgs = [db.from_sequence(arg, npartitions=self.n_calls) for arg in self.argv[1:] if len(self.argv) > 1]
        kwargBags = {key:db.from_sequence(value, npartitions=self.n_calls) for key, value in self.kwargv.items()}

        # Now, start to actually do something: map pos./kw. args onto (wrapped) user function
        results = firstArg.map(self.acme_func, *otherArgs, **kwargBags)

        # In case a debugging run is performed, use the single-threaded scheduler and return
        if debug:
            values = results.compute(scheduler="single-threaded")
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

        # Persist task graph to cluster and keep track of futures
        futures = self.client.futures_of(results.persist())

        # Set up progress bar: the while loop ensures all futures are executed
        totalTasks = len(futures)
        pbar = tqdm.tqdm(total=totalTasks, bar_format=self.tqdmFormat)
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
                erredJobIDs = [self.client.cluster.workers[job].job_id for job in erredJobs]
                errFiles = glob.glob(logDir + os.sep + "*.err")
                if len(erredFutures) or len(errFiles):
                    msg += "Please consult the following SLURM log files for details:\n"
                    msg += "".join(logFiles.format(id) + "\n" for id in erredJobIDs)
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
            # Determine filepaths of results files
            if values is None:
                values = [os.path.join(dirname, x) for x in self.kwargv["outFile"]]
        self.log.info(msg)

        # Either return collected by-worker results or the filepaths of results
        return values

    def cleanup(self):
        if self.stop_client and self.client is not None:
            cluster_cleanup(self.client)
            self.client = None

    @staticmethod
    def func_wrapper(*args, **kwargs):
        func = kwargs.pop("userFunc")
        outDir = kwargs.pop("outDir")
        taskID = kwargs.pop("taskID")
        fname = kwargs.pop("outFile")
        result = func(*args, **kwargs)
        if outDir is not None:
            with h5py.File(os.path.join(outDir, fname), "w") as h5f:
                if isinstance(result, (list, tuple)):
                    if not all(isinstance(value, (numbers.Number, str)) for value in result):
                        for rk, res in enumerate(result):
                            h5f.create_dataset("result_{}".format(rk), data=res)
                else:
                    h5f.create_dataset("result_0", data=result)
