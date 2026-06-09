#
# Client orchestration utilities for ACME
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import socket
import os
import sys
import time
import glob
import functools
import logging
import tqdm
import dask
import dask.distributed as dd
from dask_jobqueue import SLURMCluster
from typing import Union, List

# Local imports
from .dask_helpers import (
    esi_cluster_setup,
    bic_cluster_setup,
    local_cluster_setup,
    slurm_cluster_setup,
    cluster_cleanup,
    count_online_workers,
)
from .shared import is_esi_node, is_bic_node
from .validators import validate_boolean
from .argument_processor import ArgumentProcessor
from .memory_profiler import MemoryProfiler
from .config import ACMEConfig

# Fetch logger
log = logging.getLogger("ACME")


class ClientOrchestrator:
    """
    Unified manager for dask client lifecycle and computation orchestration.

    Consolidates client management and execution coordination into a single
    coherent interface, building on successful patterns from Phase 1-3.

    This class handles:
    - Client lifecycle management (creation, validation, cleanup)
    - Cluster setup and configuration
    - Computation orchestration (task submission, monitoring, error handling)
    - Worker callback management
    - Progress monitoring coordination
    """

    def __init__(
        self,
        config: ACMEConfig,
        processor: ArgumentProcessor,
        profiler: MemoryProfiler,
    ):
        """
        Initialize orchestrator with configuration and helper objects.

        Parameters
        ----------
        config : ACMEConfig
            Centralized configuration object with all execution parameters
        processor : ArgumentProcessor
            Argument broadcasting and formatting helper
        profiler : MemoryProfiler
            Memory estimation for auto-partition selection
        """
        self.config = config
        self.processor = processor
        self.profiler = profiler
        self._distributed_locks = {}
        self._initialized = False

    def prepare_client(self) -> None:
        """
        Setup or fetch dask distributed processing client.

        Orchestrates client creation/attachment based on:
        - Existing client detection and configuration
        - SLURM vs local environment detection
        - Memory profiling for auto-partition selection
        - Worker validation and counting
        - Distributed locking initialization for single-file output

        Also ensures that ad-hoc clients created here are stopped and worker jobs
        are properly released at the end of computation. However, ensures any client
        not created by `prepare_client` is **not** automatically cleaned up.
        """
        # Check if a dask client is already running
        try:
            self.config.client = dd.get_client()  # type: ignore
            log.debug("Detected running client %s", str(self.config.client))
            self._use_existing_client(self.config.client)
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
            self._create_local_client()
        else:
            self._create_slurm_client()

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

    def execute_computation(self, debug: bool = False) -> Union[List, None]:
        """
        Execute parallel computation with full orchestration.

        Parameters
        ----------
        debug : bool
            If True, use a single-threaded dask scheduler that does
            not actually process anything concurrently but uses the dask framework
            in a sequential setup.

        Returns
        -------
        results : list or None
            Computation results or filepaths. Returns None if prepare_client
            has not been called yet.
        """
        # If `prepare_client` has not been called yet, don't attempt to compute anything
        if self.config.client is None:
            log.debug("No parallel computing client allocated, exiting")
            return None

        # Ensure validity of debug flag
        validate_boolean(debug, name="debug")

        # Validate client has active workers
        self._validate_client()

        # Setup worker callbacks for sys.path forwarding
        self._setup_worker_callbacks()

        # Broadcast arguments and format keyword arguments
        self.config.argv, self.config.kwargv = self.processor.broadcast_arguments(
            self.config.client
        )
        kwargList = self.processor.format_kwarg_list()

        # In case a debugging run is performed, use the single-threaded scheduler and return
        if debug:
            return self._execute_debug_mode(kwargList)

        # Determine log information location
        logDir = self._determine_log_directory()

        # Log execution information
        msg = "Preparing %d parallel calls of `%s` using %d workers"
        log.info(
            msg
            % (self.config.n_calls, self.config.func.__name__, self.config.n_workers)
        )
        msg = "Log information available at %s"
        log.debug(msg % logDir)

        # Submit function calls to the cluster
        futures = self._submit_tasks(kwargList)

        # Monitor task execution with progress bar
        self._monitor_progress(futures)

        # Ensure futures have completed
        time.sleep(self.config.sleepTime)

        # Check completion and handle errors
        self._check_completion(futures, logDir)

        return futures

    def cleanup(self) -> None:
        """
        Cleanup client resources and distributed locks.

        Shut down any ad-hoc distributed computing clients created by `prepare_client`.
        If `prepare_client` has not been launched yet, simply return.
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

    # --- Private Methods ---

    def _use_existing_client(self, client: dd.Client) -> None:
        """Configure and validate existing client attachment."""
        log.debug("Detected running client %s", str(client))
        if self.config.stop_client == "auto":
            self.config.stop_client = False
            msg = (
                "Changing `stop_client` from `'auto'` to `False` "
                + "to not terminate external client"
            )
            log.debug(msg)

        self.config.n_workers = count_online_workers(client.cluster)  # type: ignore
        log.debug("Found %d alive workers in the client", self.config.n_workers)
        msg = "Attaching to parallel computing client %s"
        log.info(msg % (str(client)))

    def _create_local_client(self) -> None:
        """Create local multiprocessing client."""
        log.debug("SLURM not found, calling `local_cluster_setup`")
        self.config.client = local_cluster_setup(n_workers=self.config.n_workers, interactive=False)  # type: ignore

    def _create_slurm_client(self) -> None:
        """Create SLURM-based cluster client with cluster detection."""
        # If `partition` is "auto", attempt to heuristically determine average
        # memory consumption of jobs
        if self.config.partition == "auto":
            mem_per_worker = self.profiler.estimate_memory(self.config.output_dir)

        # Use appropriate cluster setup based on environment
        if is_esi_node():
            self._setup_esi_cluster()
        elif is_bic_node():
            self._setup_bic_cluster()
        else:
            self._setup_generic_slurm_cluster()

        # If startup is aborted by user, get outta here
        if self.config.client is None:  # pragma: no cover
            err = "Could not start distributed computing client. "
            log.error(err)
            raise ConnectionAbortedError(err)

    def _setup_esi_cluster(self) -> None:
        """Setup ESI-specific cluster configuration."""
        msg = "Running on ESI compute node, calling `esi_cluster_setup`"
        log.debug(msg)
        self.config.client = esi_cluster_setup(
            partition=self.config.partition,
            n_workers=self.config.n_workers,  # type: ignore
            mem_per_worker=self.config.mem_per_worker,
            timeout=self.config.setup_timeout,
            interactive=self.config.setup_interactive,
            start_client=True,
        )

    def _setup_bic_cluster(self) -> None:
        """Setup CoBIC-specific cluster configuration."""
        msg = "Running on CoBIC compute node, calling `bic_cluster_setup`"
        log.debug(msg)
        self.config.client = bic_cluster_setup(
            partition=self.config.partition,
            n_workers=self.config.n_workers,  # type: ignore
            mem_per_worker=self.config.mem_per_worker,
            timeout=self.config.setup_timeout,
            interactive=self.config.setup_interactive,
            start_client=True,
        )

    def _setup_generic_slurm_cluster(self) -> None:
        """Setup generic SLURM cluster configuration."""
        wrng = (
            "Cluster node %s not recognized. Falling back to vanilla "
            + "SLURM setup allocating one worker and one core per worker"
        )
        log.warning(wrng % (socket.getfqdn()))
        processes_per_worker = 1
        n_cores = 1
        self.config.client = slurm_cluster_setup(
            partition=self.config.partition,  # type: ignore
            n_cores=n_cores,
            n_workers=self.config.n_workers,  # type: ignore
            processes_per_worker=processes_per_worker,
            mem_per_worker=self.config.mem_per_worker,
            n_workers_startup=1,
            timeout=self.config.setup_timeout,
            interactive=self.config.setup_interactive,
            interactive_wait=120,
            start_client=True,
            job_extra=[],
            invalid_partitions=[],
        )

    def _validate_client(self) -> None:
        """Ensure client has active workers."""
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

    def _setup_worker_callbacks(self) -> None:
        """Register worker callbacks for sys.path forwarding."""

        # Dask does not correctly forward the `sys.path` from the parent process
        # to its workers. Fix this.
        def init_acme(dask_worker, syspath):
            sys.path = list(syspath)

        self.config.client.register_worker_callbacks(
            setup=functools.partial(init_acme, syspath=sys.path)
        )
        log.debug("Registered worker callback to forward `sys.path`")

    def _execute_debug_mode(self, kwargList: List) -> List:
        """Execute computation in debug mode with single-threaded scheduler."""
        log.warning("Running in debug mode")
        with dask.config.set(scheduler="single-threaded"):
            log.debug("Using single-threaded scheduler to evaluate function")
            values = self.config.client.gather(
                [
                    self.config.client.submit(self.config.acme_func, *args, **kwargs)
                    for args, kwargs in zip(zip(*self.config.argv), kwargList)
                ]
            )
            return values

    def _submit_tasks(self, kwargList: List) -> List[dd.Future]:
        """Submit parallel tasks to cluster and return futures."""
        log.debug(
            "Submitting %d function calls to client %s",
            self.config.n_calls,
            str(self.config.client),
        )
        futures = [
            self.config.client.submit(self.config.acme_func, *args, **kwargs)
            for args, kwargs in zip(zip(*self.config.argv), kwargList)
        ]
        return futures

    def _determine_log_directory(self) -> str:
        """Determine log directory based on cluster type."""
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
        return logDir

    def _monitor_progress(self, futures: List[dd.Future]) -> None:
        """Monitor task execution with progress bar."""
        # Set up progress bar: the while loop ensures all futures are executed
        totalTasks = len(futures)
        pbar = tqdm.tqdm(
            total=totalTasks,
            bar_format=self.config.tqdmFormat,
            position=0,
            leave=True,
        )
        cnt = 0
        while any(f.status == "pending" for f in futures):
            time.sleep(self.config.sleepTime)
            new = max(0, sum([f.status == "finished" for f in futures]) - cnt)
            cnt += new
            pbar.update(new)
        pbar.close()

    def _check_completion(self, futures: List[dd.Future], logDir: str) -> None:
        """
        Verify all tasks completed successfully with error analysis.

        If number of 'finished' tasks is less than expected, go into
        problem analysis mode: all futures that erred have an `.exception`
        method which can be used to track down the worker it was executed by.
        """
        finishedTasks = sum([f.status == "finished" for f in futures])
        totalTasks = len(futures)

        if finishedTasks < totalTasks:
            # Get scheduler log for error analysis
            schedulerLog = list(
                self.config.client.cluster.get_logs(
                    cluster=False, scheduler=True, workers=False
                ).values()
            )[0]
            erredFutures = [f for f in futures if f.status == "error"]

            # Build error message
            msg = "Parallel computation failed: %d/%d tasks failed or stalled. "
            msg = msg % (totalTasks - finishedTasks, totalTasks)
            msg += "Concurrent computing scheduler log info: "
            msg += schedulerLog + "\n"

            # Analyze errors based on cluster type
            if self.config.client.cluster.__class__.__name__ == "SLURMCluster":
                logFiles, logDirectory = self._get_slurm_log_info()
                msg += self._analyze_slurm_errors(
                    erredFutures, schedulerLog, logFiles, logDirectory
                )
            else:
                msg += self._analyze_local_errors(erredFutures)

            # Finally, raise an error and get outta here
            log.error(msg)
            raise RuntimeError(msg)

    def _get_slurm_log_info(self) -> tuple:
        """Get SLURM log file information."""
        logFiles = self.config.client.cluster.job_header.split("--output=")[1].replace(
            "%j", "{}"
        )
        logDir = os.path.split(logFiles)[0]
        return logFiles, logDir

    def _analyze_slurm_errors(
        self,
        erredFutures: List[dd.Future],
        schedulerLog: str,
        logFiles: str,
        logDir: str,
    ) -> str:
        """
        Analyze SLURM-specific errors and generate detailed error messages.

        Performs the Herculean task of tracking down which dask worker was
        executed by which SLURM job...
        """
        msg = ""
        try:
            erredJobs = [
                f.exception().last_worker.identity()["id"] for f in erredFutures
            ]
        except AttributeError:
            erredJobs = []
        erredJobs = list(set(erredJobs))
        validIDs = [
            job for job in erredJobs if job in self.config.client.cluster.workers.keys()
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

        return msg

    def _analyze_local_errors(self, erredFutures: List[dd.Future]) -> str:
        """Analyze local cluster-specific errors."""
        msg = "Parallel worker log details: \n"
        workerLogs = self.config.client.get_worker_logs().values()
        for wLog in workerLogs:
            if "Failed" in wLog:
                msg += wLog
        return msg
