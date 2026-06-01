#
# Computational scaffolding for user-interface
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
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
import functools
import pickle
import logging
import multiprocessing
import psutil
import tqdm
import h5py
import dask
import dask.distributed as dd
import numpy as np
from dask_jobqueue import SLURMCluster
from typing import TYPE_CHECKING, Optional, Any, Union, List
from numpy.typing import ArrayLike

# Local imports
from . import __path__
from .dask_helpers import (
    esi_cluster_setup,
    bic_cluster_setup,
    local_cluster_setup,
    slurm_cluster_setup,
    cluster_cleanup,
    count_online_workers,
)
from .shared import user_yesno, is_esi_node, is_slurm_node, is_bic_node
from .logger import prepare_log
from .validators import validate_boolean, validate_pmap
from .config import ACMEConfig
from .memory_profiler import MemoryProfiler
from .argument_processor import ArgumentProcessor
from .results.output_setup import OutputDirectoryManager, HDF5ContainerFactory
from .results.result_handler import ResultStorageManager
from .results.post_processor import ResultPostProcessor

isSpyModule = False
if "syncopy" in sys.modules:  # pragma: no cover
    isSpyModule = True
if TYPE_CHECKING:  # pragma: no cover
    from frontend import ParallelMap

__all__: List["str"] = ["ACMEdaemon"]

# Fetch logger
log = logging.getLogger("ACME")


# Main manager for parallel execution of user-defined functions
class ACMEdaemon(object):

    # Restrict valid class attributes
    __slots__ = ("results_container", "config", "processor", "profiler")

    def __init__(
        self,
        pmap: "ParallelMap",
        n_workers: Union[int, str] = "auto",
        write_worker_results: bool = True,
        output_dir: Optional[str] = None,
        result_shape: Optional[tuple[Optional[int], ...]] = None,
        result_dtype: str = "float",
        single_file: bool = False,
        write_pickle: bool = False,
        dryrun: bool = False,
        partition: str = "auto",
        mem_per_worker: str = "auto",
        setup_timeout: int = 60,
        setup_interactive: bool = True,
        stop_client: Union[bool, str] = "auto",
        verbose: Optional[bool] = None,
        logfile: Optional[Union[bool, str]] = None,
    ) -> None:
        """
        Manager class for performing concurrent user function calls

        Parameters
        ----------
        pmap : :class:`~acme.ParallelMap` context manager
            By default, `:class:~`acme.ACMEDaemon` assumes  that that
            the provided :class:`~acme.ParallelMap` instance has already
            been properly set up to process `func` (all input arguments parsed and
            properly formatted). All other input arguments of `:class:~`acme.ACMEDaemon`
            are extracted from the provided :class:`~acme.ParallelMap` instance.
        n_workers : int or "auto"
            Number of SLURM workers (=jobs) to spawn. See :class:`~acme.ParallelMap`
            for details.
        write_worker_results : bool
            If `True`, the return value(s) of `func` is/are saved on disk. See
            :class:`~acme.ParallelMap` for details.
        output_dir : str or None
            If provided, auto-generated results are stored in the given path. See
            :class:`~acme.ParallelMap` for details.
        result_shape : tuple or None
            If provided, results are slotted into a dataset/array with layout `result_shape`. See
            :class:`~acme.ParallelMap` for details.
        result_dtype : str
            Determines numerical datatype of dataset laid out by `result_shape`.
            See :class:`~acme.ParallelMap` for details.
        single_file : bool
            If `True`, parallel workers write to the same results container. See
            :class:`~acme.ParallelMap` for details.
        write_pickle : bool
            If `True`, the return value(s) of `func` is/are pickled to disk. See
            :class:`~acme.ParallelMap` for details.
        dryrun : bool
            If `True`, a dry-run of calling `func` is performed using a single
            `args`, `kwargs` tuple. See :class:`~acme.ParallelMap` for details.
        partition : str
            Name of SLURM partition to use. See :class:`~acme.ParallelMap` for details.
        mem_per_worker : str
            Memory booking for each SLURM worker. See :class:`~acme.ParallelMap` for details.
        setup_timeout : int
            Timeout period (in seconds) for SLURM workers to come online. See
            :class:`~acme.ParallelMap` for details.
        setup_interactive : bool
            If `True`, user input is queried in case not enough SLURM workers could
            be started within `setup_timeout` seconds. See :class:`~acme.ParallelMap`
            for details.
        stop_client : bool or "auto"
            If `"auto"`, automatically started distributed computing clients
            are shut down at the end of computation, while user-provided clients
            are left untouched. See :class:`~acme.ParallelMap` for details.
        verbose : None or bool
            If `None` (default), general run-time information as well as warnings
            and errors are shown. See :class:`~acme.ParallelMap` for details.
        logfile : None or bool or str
            If `None` (default) or `True`, and `write_worker_results` is
            `True`, all run-time information as well as errors and
            warnings are tracked in a log-file. See :class:`~acme.ParallelMap`
            for details.

        Returns
        -------
        results : list
            If `write_worker_results` is `True`, `results` is a list of HDF5 file-names
            containing computed results. If `write_worker_results` is `False`,
            results is a list comprising the actual return values of `func`.
            If `:class:~`acme.ACMEDaemon` was instantiated by :class:`~acme.ParallelMap`,
            results are propagated back to :class:`~acme.ParallelMap`.

        See also
        --------
        ParallelMap : Context manager and main user interface
        """

        # First and foremost: ensure we got something useful to work with
        validate_pmap(pmap)

        # Create configuration
        self.config = ACMEConfig(
            func=pmap.func,
            argv=pmap.argv,
            kwargv=pmap.kwargv,
            n_calls=pmap.n_inputs,
            n_workers=n_workers,
            write_worker_results=write_worker_results,
            output_dir=output_dir,
            result_shape=result_shape,
            result_dtype=result_dtype,
            single_file=single_file,
            write_pickle=write_pickle,
            dryrun=dryrun,
            partition=partition,
            mem_per_worker=mem_per_worker,
            setup_timeout=setup_timeout,
            setup_interactive=setup_interactive,
            stop_client=stop_client,
            verbose=verbose,
            logfile=logfile,
        )
        self.config.validate()

        # Set up output handler
        self.pre_process()

        # Set up argument processing helper class
        self.processor = ArgumentProcessor(
            self.config.argv, self.config.kwargv, self.config.n_calls
        )

        # Set up memory profiling helper class
        self.profiler = MemoryProfiler(
            self.processor,
            self.config.acme_func,
            self.config.func.__name__,
            self.config.tqdmFormat,
        )

        # If requested, perform single-worker dry-run (and quit if desired)
        if dryrun:
            goOn = self.profiler.perform_dryrun(
                output_dir=self.config.output_dir,
                setup_interactive=self.config.setup_interactive,
            )
            if not goOn:
                log.debug("Quitting after dryrun")
                return
            log.debug("Continuing after dryrun")

        # Either use existing dask client or start a fresh instance
        self.prepare_client()

    def pre_process(self) -> None:
        """
        If `write_*` is `True` set up directories for saving output HDF5 containers
        (or pickle files). Warn if results are to be collected in memory
        """

        # If automatic saving of results is requested, make necessary preparations
        if self.config.write_worker_results:
            self.setup_output()
        else:

            # If `taskID` is not an explicit kw-arg of `func` and `func` does not
            # accept "anonymous" `**kwargs`, don't save anything but return stuff
            log.debug("Automatic output processing disabled.")
            if self.config.kwargv.get("taskID") is None:
                if not isSpyModule:
                    msg = (
                        "`write_worker_results` is `False` and `taskID` is not a keyword argument of %s. "
                        + "Results will be collected in memory by caller - this might be slow and can lead "
                        + "to excessive memory consumption. "
                    )
                    log.warning(msg, self.config.func.__name__)
                self.config.collect_results = True  # type: ignore
            else:
                self.config.kwargv["taskID"] = self.config.task_ids
                self.config.collect_results = False  # type: ignore
                msg = (
                    "Not collecting results in memory, leaving output "
                    + "processing to user-provided function"
                )
                log.debug(msg)

            # The "raw" user-provided function is used in the computation
            self.config.acme_func = self.config.func
            log.debug("Not wrapping user-provided function but invoking it directly")

        # If progress tracking in a log-file was requested, set it up now
        prepare_log(
            logname="ACME", logfile=self.config.logfile, verbose=self.config.verbose
        )
        log.debug("Set up logfile=%s", str(self.config.logfile))

        return

    def setup_output(self) -> None:
        """
        Local helper for creating output directories and preparing containers
        """
        from .results.output_setup import OutputDirectoryManager, HDF5ContainerFactory

        # Use output setup manager for directory creation
        
        # Create output directory structure
        if not self.config.single_file and not self.config.write_pickle:
            log.debug("Preparing payload directory for HDF5 containers")
            payloadName = f"{self.config.func.__name__}_payload"
            outputDir = HDF5ContainerFactory.create_payload_directory(self.config.output_dir, self.config.func.__name__)
        else:
            msg = (
                "Either single-file output or pickling was requested. "
                + "Not creating payload directory"
            )
            log.debug(msg)
            outputDir = self.config.output_dir
            try:
                os.makedirs(outputDir)
                log.debug("Created %s", outputDir)
            except Exception as exc:
                err = "automatic creation of output folder %s failed: %s"
                log.error(err, outputDir, str(exc))
                raise OSError(err % (outputDir, str(exc)))

        # Re-define or allocate key "taskID" to track concurrent processing results
        self.config.kwargv["taskID"] = self.config.task_ids
        self.config.collect_results = False

        # Set up correct file-extension for output files; in case of HDF5
        # containers, prepare "main" file for collecting/symlinking worker results
        if self.config.write_pickle:
            fExt = "pickle"
            log.debug("Pickling was requested")
        else:
            fExt = "h5"
            self.config.results_container = os.path.join(self.config.output_dir, f"{self.config.func.__name__}.h5")  # type: ignore
            log.debug("Using HDF5 storage %s", self.config.results_container)

        # Use HDF5 container factory for container creation
        container_factory = HDF5ContainerFactory()

        # By default, `results_container` is a collection of links that point to
        # worker-generated HDF5 containers; if `single_file` is `True`, then
        # `results_container` is a "real" container with actual dataset(s)
        if self.config.single_file:
            self.config.kwargv["singleFile"] = [True]
            self.config.kwargv["outFile"] = [self.config.results_container]
            log.debug("Saving results in single HDF5 container")

            # If no output shape provided, prepare groups for storing datasets;
            # otherwise allocate a single dataset w/specified dimension
            if self.config.result_shape is None:
                container_factory.create_single_file_container(
                    self.config.results_container,
                    self.config.task_ids,
                    None,
                    self.config.result_dtype
                )
            else:
                container_factory.create_single_file_container(
                    self.config.results_container,
                    self.config.task_ids,
                    self.config.result_shape,
                    self.config.result_dtype
                )

        else:
            self.config.kwargv["outFile"] = [
                os.path.join(outputDir, f"{self.config.func.__name__}_{taskID}.{fExt}")
                for taskID in self.config.task_ids
            ]
            if not self.config.write_pickle:

                # If no output shape provided, generate links to external datasets;
                # otherwise allocate a virtual dataset w/specified dimension
                if self.config.result_shape is None:
                    container_factory.create_virtual_dataset_container(
                        self.config.results_container,
                        self.config.task_ids,
                        self.config.kwargv["outFile"],
                        None,
                        0,  # default stacking dim
                        self.config.result_dtype,
                        outputDir
                    )
                else:
                    container_factory.create_virtual_dataset_container(
                        self.config.results_container,
                        self.config.task_ids,
                        self.config.kwargv["outFile"],
                        self.config.result_shape,
                        self.config.stacking_dim,  # type: ignore
                        self.config.result_dtype,
                        outputDir
                    )

        # Include logger name in keywords so that workers can use it
        self.config.kwargv["logName"] = [log.name]

        # Wrap the user-provided func and distribute it across workers
        self.config.kwargv["userFunc"] = [self.config.func]
        self.config.acme_func = self.func_wrapper  # type: ignore
        log.debug("Wrapping user-provided function inside func_wrapper")

        # Finally, attach verbosity flag to enable logging inside wrapper
        self.config.kwargv["logLevel"] = [log.level]

        return

    def prepare_client(self) -> None:
        """
        Setup or fetch dask distributed processing client. Depending on available
        hardware, either start a local multi-processing client or launch a
        worker cluster via SLURM.

        Also ensure that ad-hoc clients created here are stopped and worker jobs
        are properly released at the end of computation. However, ensure any client
        not created by `prepare_client` is **not** automatically cleaned up.
        """

        # Check if a dask client is already running
        try:
            self.config.client = dd.get_client()  # type: ignore
            log.debug("Detected running client %s", str(self.config.client))
            if self.config.stop_client == "auto":
                self.config.stop_client = False
                msg = (
                    "Changing `stop_client` from `'auto'` to `False` "
                    + "to not terminate external client"
                )
                log.debug(msg)
            self.config.n_workers = count_online_workers(self.config.client.cluster)  # type: ignore
            log.debug("Found %d alive workers in the client", self.config.n_workers)
            msg = "Attaching to parallel computing client %s"
            log.info(msg % (str(self.config.client)))
            return
        except ValueError:
            msg = "No running client detected, preparing to start a new one"
            log.debug(msg)
            if self.config.stop_client == "auto":
                self.config.stop_client = True  # type: ignore
                msg = (
                    "Changing `stop_client` from `'auto'` to `True` "
                    + "to clean up client started by `ParallelMap`"
                )
                log.debug(msg)

        # If things are running locally, simply fire up a dask-distributed client,
        # otherwise go through the motions of preparing a full worker cluster
        if not self.config.has_slurm:  # pragma: no cover

            log.debug("SLURM not found, Calling `local_cluster_setup`")
            self.config.client = local_cluster_setup(n_workers=self.config.n_workers, interactive=False)  # type: ignore

        else:

            # If `partition` is "auto", attempt to heuristically determine average
            # memory consumption of jobs
            if partition == "auto":
                mem_per_worker = self.profiler.estimate_memory(self.config.output_dir)

            # All set, remaining input processing is done by respective `*_cluster_setup` routines
            if is_esi_node():
                msg = "Running on ESI compute node, Calling `esi_cluster_setup`"
                log.debug(msg)
                self.config.client = esi_cluster_setup(
                    partition=partition,
                    n_workers=n_workers,  # type: ignore
                    mem_per_worker=mem_per_worker,
                    timeout=setup_timeout,
                    interactive=setup_interactive,
                    start_client=True,
                )

            # All set, remaining input processing is done by respective `*_cluster_setup` routines
            elif is_bic_node():
                msg = "Running on CoBIC compute node, Calling `bic_cluster_setup`"
                log.debug(msg)
                self.config.client = bic_cluster_setup(
                    partition=partition,
                    n_workers=n_workers,  # type: ignore
                    mem_per_worker=mem_per_worker,
                    timeout=setup_timeout,
                    interactive=setup_interactive,
                    start_client=True,
                )

            # Unknown cluster node, use vanilla config
            else:  # pragma: no cover
                wrng = (
                    "Cluster node %s not recognized. Falling back to vanilla "
                    + "SLURM setup allocating one worker and one core per worker"
                )
                log.warning(wrng % (socket.getfqdn()))
                processes_per_worker = 1
                n_cores = 1
                self.config.client = slurm_cluster_setup(
                    partition=partition,  # type: ignore
                    n_cores=n_cores,
                    n_workers=n_workers,  # type: ignore
                    processes_per_worker=processes_per_worker,
                    mem_per_worker=mem_per_worker,
                    n_workers_startup=1,
                    timeout=setup_timeout,
                    interactive=setup_interactive,
                    interactive_wait=120,
                    start_client=True,
                    job_extra=[],
                    invalid_partitions=[],
                )

            # If startup is aborted by user, get outta here
            if self.config.client is None:  # pragma: no cover
                err = "Could not start distributed computing client. "
                log.error(err)
                raise ConnectionAbortedError(err)

        # Set `n_workers` to no. of active workers in the initialized cluster
        self.config.n_workers = len(self.config.client.cluster.workers)  # type: ignore
        log.debug(
            "Setting `n_workers = %d` based on active workers in %s",
            self.config.n_workers,
            str(self.config.client),
        )

        # If single output file saving was chosen, initialize distributed
        # lock for shared writing to container
        if self.config.kwargv.get("singleFile") is not None:
            msg = "Initializing distributed lock for writing to single shared results container"
            log.debug(msg)
            dd.lock.Lock(name=os.path.basename(self.config.results_container))  # type: ignore

        return

    def compute(self, debug: bool = False) -> Union[List, None]:
        """
        Perform the actual parallel execution of `func`

        If `debug` is `True`, use a single-threaded dask scheduler that does
        not actually process anything concurrently but uses the dask framework
        in a sequential setup.
        """

        validate_boolean(debug, name="debug")

        # If `prepare_client` has not been called yet, don't attempt to compute anything
        if self.config.client is None:
            log.debug("No parallel computing client allocated, exiting")
            return None

        # Check if the underlying parallel computing cluster hosts actually usable workers
        if count_online_workers(self.config.client.cluster) == 0:
            err = (
                "no active workers found in distributed computing client %s "
                + "Consider running \n"
                + "\timport dask.distributed as dd; dd.get_client().restart()\n"
                + "If this fails to make workers come online, please use\n"
                + "\timport acme; acme.cluster_cleanup()\n"
                + "to shut down any defunct distributed computing clients"
            )
            log.error(err, str(self.config.client))
            raise RuntimeError(err % (str(self.config.client)))
        log.debug(
            "Found %d workers in client %s",
            count_online_workers(self.config.client.cluster),
            str(self.config.client),
        )

        # Dask does not correctly forward the `sys.path` from the parent process
        # to its workers. Fix this.
        def init_acme(dask_worker, syspath):
            sys.path = list(syspath)

        self.config.client.register_worker_callbacks(
            setup=functools.partial(init_acme, syspath=sys.path)
        )
        log.debug("Registered worker callback to forward `sys.path`")

        # Broadcast arguments and format keyword arguments
        self.config.argv, self.config.kwargv = self.processor.broadcast_arguments(
            self.config.client
        )
        kwargList = self.processor.format_kwarg_list()

        # In case a debugging run is performed, use the single-threaded scheduler and return
        if debug:
            log.warning("Running in debug mode")
            with dask.config.set(scheduler="single-threaded"):
                log.debug("Using single-threaded scheduler to evaluate function")
                values = self.config.client.gather(
                    [
                        self.config.client.submit(
                            self.config.acme_func, *args, **kwargs
                        )
                        for args, kwargs in zip(zip(*self.config.argv), kwargList)
                    ]
                )
                return values

        # Depending on the used dask cluster object, point to respective log info
        if isinstance(self.config.client.cluster, SLURMCluster):
            logFiles = self.config.client.cluster.job_header.split("--output=")[
                1
            ].replace("%j", "{}")
            logDir = os.path.split(logFiles)[0]
        else:  # pragma: no cover
            logFiles = []
            logDir = (
                os.path.dirname(self.config.client.cluster.dashboard_link)
                + "/info/main/workers.html"
            )
        msg = "Preparing %d parallel calls of `%s` using %d workers"
        log.info(
            msg
            % (self.config.n_calls, self.config.func.__name__, self.config.n_workers)
        )
        msg = "Log information available at %s"
        log.debug(msg % (logDir))

        # Submit `self.config.n_calls` function calls to the cluster
        log.debug(
            "Submitting %d function calls to client %s",
            self.config.n_calls,
            str(self.config.client),
        )
        futures = [
            self.config.client.submit(self.config.acme_func, *args, **kwargs)
            for args, kwargs in zip(zip(*self.config.argv), kwargList)
        ]

        # Set up progress bar: the while loop ensures all futures are executed
        totalTasks = len(futures)
        pbar = tqdm.tqdm(
            total=totalTasks, bar_format=self.config.tqdmFormat, position=0, leave=True
        )
        cnt = 0
        while any(f.status == "pending" for f in futures):
            time.sleep(self.config.sleepTime)
            new = max(0, sum([f.status == "finished" for f in futures]) - cnt)
            cnt += new
            pbar.update(new)
        pbar.close()

        # Avoid race condition: give futures time to perform switch from 'pending'
        # to 'finished' so that `finishedTasks` is computed correctly
        log.debug("Waiting %f seconds for futures", self.config.sleepTime)
        time.sleep(self.config.sleepTime)

        # If number of 'finished' tasks is less than expected, go into
        # problem analysis mode: all futures that erred hav an `.exception`
        # method which can be used to track down the worker it was executed by
        # Once we know the worker, we can point to the right log file. If
        # futures were cancelled (by the user or the SLURM controller),
        # `.exception` is `None` and we can't reliably track down the
        # respective executing worker
        finishedTasks = sum([f.status == "finished" for f in futures])
        if finishedTasks < totalTasks:
            schedulerLog = list(
                self.config.client.cluster.get_logs(
                    cluster=False, scheduler=True, workers=False
                ).values()
            )[0]
            erredFutures = [f for f in futures if f.status == "error"]
            msg = "Parallel computation failed: %d/%d tasks failed or stalled. "
            msg = msg % (totalTasks - finishedTasks, totalTasks)
            msg += "Concurrent computing scheduler log info: "
            msg += schedulerLog + "\n"

            # If we're working w/`SLURMCluster`, perform the Herculean task of
            # tracking down which dask worker was executed by which SLURM job...
            if self.config.client.cluster.__class__.__name__ == "SLURMCluster":
                try:
                    erredJobs = [
                        f.exception().last_worker.identity()["id"] for f in erredFutures
                    ]
                except AttributeError:
                    erredJobs = []
                erredJobs = list(set(erredJobs))
                validIDs = [
                    job
                    for job in erredJobs
                    if job in self.config.client.cluster.workers.keys()
                ]
                erredJobIDs = [
                    self.config.client.cluster.workers[job].job_id for job in validIDs
                ]
                errFiles = glob.glob(logDir + os.sep + "*.err")
                if len(erredFutures) > 0 or len(errFiles) > 0:
                    msg += "Please consult the following SLURM log files for details:\n"
                    if len(erredJobIDs) > 0:
                        msg += "".join(logFiles.format(id) + "\n" for id in erredJobIDs)
                    else:
                        msg += "".join(logDir)
                    msg += "".join(errfile + "\n" for errfile in errFiles)
                else:
                    msg += "Please check SLURM logs in %s" % (logDir)

            # In case of a `LocalCluster`, syphon worker logs
            else:  # pragma: no cover
                msg += "Parallel worker log details: \n"
                workerLogs = self.config.client.get_worker_logs().values()
                for wLog in workerLogs:
                    if "Failed" in wLog:
                        msg += wLog

            # Finally, raise an error and get outta here
            log.error(msg)
            raise RuntimeError(msg)

        # Postprocessing of results
        values = self.post_process(futures)

        # Either return collected by-worker results or the filepaths of results
        return values

    def post_process(self, futures: dd.Future) -> Union[List, None]:
        """
        Local helper to post-process results on disk/in-memory

        The return `values` is either
        `None` : if neither in-memory results collection or auto-writing was requested
        list of file-names: if `write_worker_results` is `True`
        list of objects: if in-memory results collection was requested
        """
        from .results.post_processor import ResultPostProcessor

        # Use result post-processor for handling results
        post_processor = ResultPostProcessor(self.config.client, self.config.results_container)
        
        # Process futures using the post-processor
        result = post_processor.process_futures(
            futures,
            self.config.collect_results,
            self.config.result_shape,
            self.config.stacking_dim,
            self.config.result_dtype,
            self.config.acme_func,
            self.config.func,
            self.config.kwargv
        )

        # Finally, establish shortcut to `results_container` (if present) for easier access
        self.results_container = self.config.results_container

        return result

    def cleanup(self) -> None:
        """
        Shut down any ad-hoc distributed computing clients created by `prepare_client`
        """

        # If `prepare_client` has not been launched yet, just get outta here
        if self.config.client is None:
            log.debug("Helper `prepare_client` not yet launched, exiting")
            return
        if self.config.stop_client and self.config.client is not None:
            log.debug(
                "Found client %s, calling `cluster_cleanup`", str(self.config.client)
            )
            cluster_cleanup(self.config.client)
            self.config.client = None
            return
        log.debug("Either `stop_client = False` or no client found, returning")

        return

    @staticmethod
    def func_wrapper(*args: Any, **kwargs: Optional[Any]) -> None:  # pragma: no cover
        """
        If the output of `func` is saved to disk, wrap `func` with this static
        method to take care of filling up HDF5/pickle files

        If writing to HDF5 fails, use an "emergency-pickling" mechanism to try
        to save the output of `func` using pickle instead
        """
        from .results.result_handler import ResultStorageManager

        # Extract everything from `kwargs` appended by `ACMEdaemon`
        func = kwargs.pop("userFunc")
        taskID = kwargs.pop("taskID")
        fname = kwargs.pop("outFile")
        logName = kwargs.pop("logName")
        logLevel = kwargs.pop("logLevel")
        singleFile = kwargs.pop("singleFile", False)
        stackingDim = kwargs.pop("stackingDim", None)
        memEstRun = kwargs.pop("memEstRun", False)

        # Set up logger
        log = logging.getLogger(logName)
        log.setLevel(logLevel)  # type: ignore
        for h in log.handlers:
            h.setLevel(logLevel)  # type: ignore

        # Call user-provided function
        result = func(*args, **kwargs)  # type: ignore

        # For memory estimation runs, don't start saving stuff
        if memEstRun:
            return

        # Use result storage manager for handling result storage
        try:
            # Determine storage strategy based on file extension
            write_pickle = fname.endswith(".pickle")
            write_worker_results = True  # Since we're in func_wrapper
            
            # Create appropriate result handler
            result_handler = ResultStorageManager.create_handler(
                write_pickle=write_pickle,
                write_worker_results=write_worker_results,
                single_file=singleFile,
                result_shape=None,  # Will be determined by result
                result_dtype="float",  # Default dtype
                outfile_pattern=fname
            )
            
            # Write result using the handler
            result_handler.write_result(
                result,
                taskID,
                outFile=fname,
                singleFile=singleFile,
                stackingDim=stackingDim,
                logName=logName
            )
            
        except Exception as exc:
            log.error("Failed to write result using result handler: %s", str(exc))
            # Fallback to original implementation for compatibility
            if fname.endswith(".h5"):  # type: ignore
                self._legacy_hdf5_write(result, taskID, fname, singleFile, stackingDim, log)
            else:
                self._legacy_pickle_write(result, fname, log)

    def _legacy_hdf5_write(self, result: Any, taskID: int, fname: str, singleFile: bool, stackingDim: Optional[int], log: logging.Logger) -> None:
        """Legacy HDF5 writing method for backward compatibility"""
        grpName = ""
        if singleFile:
            lock = dd.lock.Lock(name=os.path.basename(fname))  # type: ignore
            lock.acquire()
            grpName = f"comp_{taskID}/"

        if not isinstance(result, (list, tuple)):
            result = [result]

        try:
            with h5py.File(fname, "a") as h5f:
                if stackingDim is None:
                    if not all(
                        isinstance(value, (numbers.Number, str)) for value in result
                    ):
                        for rk, res in enumerate(result):
                            h5f.create_dataset(f"{grpName}result_{rk}", data=res)
                            log.debug(
                                "Created new dataset `result_%d` in %s", rk, fname
                            )
                    else:
                        h5f.create_dataset(grpName + "result_0", data=result)
                        log.debug("Created new dataset `result_0` in %s", fname)
                else:
                    if singleFile:
                        dset = h5f["result_0"]
                        idx = [slice(None)] * len(dset.shape)  # type: ignore
                        idx[stackingDim] = taskID  # type: ignore
                        if None in dset.maxshape:
                            if len(result[0].shape) < len(idx):
                                lenDim = list(
                                    set(result[0].shape).difference(dset.maxshape)
                                )
                                if len(lenDim) == 0:
                                    lenDim = result[0].shape[0]
                                else:
                                    lenDim = lenDim[0]
                                actShape = tuple(
                                    spec if spec is not None else lenDim
                                    for spec in dset.maxshape
                                )
                            else:
                                actShape = list(result[0].shape)  # type: ignore
                                actShape[stackingDim] = dset.maxshape[stackingDim]  # type: ignore
                                actShape = tuple(actShape)
                            dset.resize(actShape)
                        dset[tuple(idx)] = result[0]
                        log.debug(
                            "Wrote to pre-allocated dataset `result_0` in %s", fname
                        )
                        for rk, res in enumerate(result[1:]):
                            h5f.create_dataset(
                                f"{grpName}result_{rk + 1}", data=res
                            )
                            log.debug(
                                "Created new dataset `result_%d` in %s",
                                rk + 1,
                                fname,
                            )
                    else:
                        for rk, res in enumerate(result):
                            h5f.create_dataset(f"{grpName}result_{rk}", data=res)
                            log.debug(
                                "Created new dataset `result_%d` in %s", rk, fname
                            )

            if singleFile:
                lock.release()

        except TypeError as exc:
            if (
                "has no native HDF5 equivalent" in str(exc)
                or "One of data, shape or dtype must be specified" in str(exc)
            ) and not singleFile:
                try:
                    os.unlink(fname)  # type: ignore
                    pname = fname.rstrip(".h5") + ".pickle"  # type: ignore
                    with open(os.path.join(pname), "wb") as pkf:
                        pickle.dump(result, pkf)
                    msg = (
                        "Could not write %s results have been pickled instead: %s. Return values are most likely "
                        + "not suitable for storage in HDF5 containers. Original error message: %s"
                    )
                    log.warning(msg, fname, pname, str(exc))
                except pickle.PicklingError as pexc:
                    err = "Unable to write %s, successive attempts to pickle results failed too: %s"
                    log.error(err, fname, str(pexc))
            else:
                if singleFile:
                    err = "Could not write to %s. File potentially corrupted. Original error message: %s"
                    lock.release()
                else:
                    err = "Could not access %s. Original error message: %s"
                log.error(err, fname, str(exc))
                raise exc

        except Exception as exc:
            if singleFile:
                lock.release()
            log.error(str(exc))
            raise exc

    def _legacy_pickle_write(self, result: Any, fname: str, log: logging.Logger) -> None:
        """Legacy pickle writing method for backward compatibility"""
        try:
            with open(os.path.join(fname), "wb") as pkf:  # type: ignore
                pickle.dump(result, pkf)
                log.debug("Pickled to %s", fname)
        except pickle.PicklingError as pexc:
            err = "Could not pickle results to file %s. Original error message: %s"
            log.error(err, fname, str(pexc))
            raise pexc

        return
