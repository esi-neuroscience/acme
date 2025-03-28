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
from .dask_helpers import (esi_cluster_setup, bic_cluster_setup, local_cluster_setup,
                           slurm_cluster_setup, cluster_cleanup, count_online_workers)
from .shared import user_yesno, is_esi_node, is_slurm_node, is_bic_node
from .logger import prepare_log
isSpyModule = False
if "syncopy" in sys.modules:            # pragma: no cover
    isSpyModule = True
if TYPE_CHECKING:                       # pragma: no cover
    from frontend import ParallelMap

__all__: List["str"] = ["ACMEdaemon"]

# Fetch logger
log = logging.getLogger("ACME")


# Main manager for parallel execution of user-defined functions
class ACMEdaemon(object):

    # Restrict valid class attributes
    __slots__ = "func", "argv", "kwargv", "n_calls", "n_workers", "acme_func", \
        "task_ids", "out_dir", "collect_results", "results_container", "result_shape", \
        "result_dtype", "stacking_dim", "client", "stop_client", "has_slurm"

    # Prepend every stdout/stderr message with the name of this class
    objName = "<ACMEdaemon>"

    # format string for tqdm progress bars
    tqdmFormat = "{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

    # time (in seconds) b/w querying state of futures ('pending' -> 'finished')
    sleepTime = 0.1

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
        logfile: Optional[Union[bool, str]] = None) -> None:
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

        # The only error checking happening in `__init__`
        try:
            pClassName = pmap.__class__.__name__
        except:
            pClassName = "notParallelMap"
        if pClassName != "ParallelMap":
            msg = "%s `pmap` has to be a `ParallelMap` instance, not %s"
            raise TypeError(msg%(self.objName, str(type(pmap))))

        # Allocate first batch of slots
        self.func = pmap.func
        self.argv = pmap.argv
        self.kwargv = pmap.kwargv
        self.n_calls = pmap.n_inputs
        self.n_workers = None
        self.acme_func = None
        self.out_dir = None
        self.collect_results = None
        self.results_container = None
        self.result_shape = None
        self.result_dtype = None
        self.stacking_dim = None
        self.client = None
        self.stop_client = None

        # Define list of taskIDs for distribution across workers
        self.task_ids = list(range(self.n_calls))
        log.debug("Allocated `taskID` list: %s", str(self.task_ids))

        # Finally, determine if the code is executed on a SLURM-enabled node
        self.has_slurm = is_slurm_node()
        log.debug("Set `has_slurm = %s`", str(self.has_slurm))

        # Set up output handler
        self.pre_process(verbose,
                         logfile,
                         write_worker_results,
                         output_dir,
                         result_shape,
                         result_dtype,
                         single_file,
                         write_pickle)

        # If requested, perform single-worker dry-run (and quit if desired)
        if dryrun:
            goOn = self.perform_dryrun(setup_interactive)
            if not goOn:
                log.debug("Quitting after dryrun")
                return
            log.debug("Continuing after dryrun")

        # Either use existing dask client or start a fresh instance
        self.prepare_client(n_workers=n_workers,
                            partition=partition,
                            mem_per_worker=mem_per_worker,
                            setup_timeout=setup_timeout,
                            setup_interactive=setup_interactive,
                            stop_client=stop_client)

    def pre_process(
            self,
            verbose: Union[bool, None],
            logfile: Union[bool, str, None],
            write_worker_results: bool,
            output_dir: Union[str, None],
            result_shape: Union[tuple[Optional[int], ...], None],
            result_dtype: str,
            single_file: bool,
            write_pickle: bool) -> None:
        """
        If `write_*` is `True` set up directories for saving output HDF5 containers
        (or pickle files). Warn if results are to be collected in memory
        """

        # Basal sanity check for Boolean flags
        if not isinstance(write_worker_results, bool):
            msg = "%s `write_worker_results` has to be `True` or `False`, not %s"
            raise TypeError(msg%(self.objName, str(write_worker_results)))
        log.debug("Found `write_worker_results = %s`", str(write_worker_results))
        if not isinstance(single_file, bool):
            msg = "%s `single_file` has to be `True` or `False`, not %s"
            raise TypeError(msg%(self.objName, str(single_file)))
        log.debug("Found `single_file = %s`", str(single_file))
        if not isinstance(write_pickle, bool):
            msg = "%s `write_pickle` has to be `True` or `False`, not %s"
            raise TypeError(msg%(self.objName, str(write_pickle)))
        log.debug("Found `write_pickle = %s`", str(write_pickle))

        # Check compatibility of provided optional args
        if not write_worker_results and write_pickle:
            log.warning("Pickling of results only possible if `write_worker_results` is `True`. ")
        if not write_worker_results and output_dir:
            log.warning("Output directory specification has no effect if `write_worker_results` is `False`.")
        if write_pickle and result_shape:
            log.warning("Pickling of results does not support output array shape specification. ")
        if not write_worker_results and single_file:
            log.warning("Generating a single output file only possible if `write_worker_results` is `True`. ")
        if write_pickle and single_file:
            msg = "%s Pickling of results does not support single output file creation. "
            raise ValueError(msg%self.objName)

        # Check validity of output shape/dtype specifications
        if result_shape is not None:
            if not isinstance(result_shape, (list, tuple)):
                msg = "%s `result_shape` has to be either `None` or tuple, not %s"
                raise TypeError(msg%(self.objName, str(type(result_shape))))

            if not isinstance(result_dtype, str):
                msg = "%s `result_dtype` has to be a string, not %s"
                raise TypeError(msg%(self.objName, str(type(result_shape))))

            if sum(spec is None for spec in result_shape) != 1:
                msg = "%s `result_shape` must contain exactly one `None` entry"
                raise ValueError(msg%self.objName)

            rShape = list(result_shape)
            self.stacking_dim = result_shape.index(None)                # type: ignore
            rShape[self.stacking_dim] = self.n_calls                    # type: ignore

            if not write_worker_results and any(np.isinf(spec) for spec in rShape): # type: ignore
                msg = "%s using `np.inf` in `result_shape` is only valid if `write_worker_results` is `True`"
                raise ValueError(msg%self.objName)
            if rShape.count(np.inf) > 1:                                        # type: ignore
                msg = "%s cannot use more than one `np.inf` in `result_shape`"
                raise ValueError(msg%self.objName)
            if not all(isinstance(spec, numbers.Number) for spec in rShape):
                msg = "%s `result_shape` must only contain numerical values"
                raise ValueError(msg%self.objName)
            if any(spec < 0 or int(spec) != spec or np.isnan(spec) for spec in rShape if not np.isinf(spec)):   # type: ignore
                msg = "%s `result_shape` must only contain non-negative integers"
                raise ValueError(msg%self.objName)

            self.result_shape = tuple(rShape)                                   # type: ignore
            msg = "Found `result_shape = %s`. Set stacking dimension to %d"
            log.debug(msg, str(result_shape), self.stacking_dim)

            try:
                self.result_dtype = np.dtype(result_dtype)              # type: ignore
            except Exception as exc:
                msg = "%s `result_dtype` has to be a valid NumPy datatype specification. "
                msg += "Original error message below:\n%s"
                raise TypeError(msg%(self.objName, str(exc)))
            log.debug("Set `result_dtype = %s`", self.result_dtype)

            if write_worker_results:
                self.kwargv["stackingDim"] = [self.stacking_dim]

        else:
            log.debug("Found `result_shape = %s`", str(result_shape))
            log.debug("Found `result_dtype = %s`", str(result_dtype))

        # If automatic saving of results is requested, make necessary preparations
        if write_worker_results:
            self.setup_output(output_dir, result_shape, single_file, write_pickle)
        else:

            # If `taskID` is not an explicit kw-arg of `func` and `func` does not
            # accept "anonymous" `**kwargs`, don't save anything but return stuff
            log.debug("Automatic output processing disabled.")
            if self.kwargv.get("taskID") is None:
                if not isSpyModule:
                    msg = "`write_worker_results` is `False` and `taskID` is not a keyword argument of %s. " +\
                        "Results will be collected in memory by caller - this might be slow and can lead " +\
                        "to excessive memory consumption. "
                    log.warning(msg, self.func.__name__)
                self.collect_results = True                             # type: ignore
            else:
                self.kwargv["taskID"] = self.task_ids
                self.collect_results = False                            # type: ignore
                msg = "Not collecting results in memory, leaving output " +\
                    "processing to user-provided function"
                log.debug(msg)

            # The "raw" user-provided function is used in the computation
            self.acme_func = self.func
            log.debug("Not wrapping user-provided function but invoking it directly")

        # Unless specifically disabled by the user, enable progress-tracking
        # in a log-file if results are auto-generated
        if logfile is None and write_worker_results is True:
            logfile = True

        # Either parse provided `logfile` or set up an auto-generated file;
        # After this test, `logfile` is either a filename or `None`
        msg = "%s `logfile` has to be `None`, `True`, `False` or a valid file-name, not %s"
        if logfile is None or isinstance(logfile, bool):
            if logfile is True:
                if write_worker_results:
                    logfile = self.out_dir
                else:
                    logfile = os.path.dirname(os.path.abspath(inspect.getfile(self.func)))
                logfile = os.path.join(
                    logfile,                                                    # type: ignore
                    f"ACME_{self.func.__name__}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log")   # type: ignore
            else:
                logfile = None
        elif isinstance(logfile, str):
            if os.path.isdir(logfile):
                raise IOError(msg%(self.objName, "a directory"))
            logfile = os.path.abspath(os.path.expanduser(logfile))
        else:
            raise TypeError(msg%(self.objName, str(type(logfile))))

        # If progress tracking in a log-file was requested, set it up now
        prepare_log(logname="ACME", logfile=logfile, verbose=verbose)
        log.debug("Set up logfile=%s", str(logfile))

        return

    def setup_output(
            self,
            output_dir: Union[str, None],
            result_shape: Union[tuple[Optional[int], ...], None],
            single_file: bool,
            write_pickle: bool) -> None:
        """
        Local helper for creating output directories and preparing containers
        """

        # Check validity of output dir specification
        if not isinstance(output_dir, (type(None), str)):
            msg = "%s `output_dir` has to be either `None` or str, not %s"
            raise TypeError(msg%(self.objName, str(type(output_dir))))
        log.debug("Found `output_dir = %s`", str(output_dir))

        # If provided, standardize output dir spec, otherwise use default locations
        if output_dir is not None:
            outDir = os.path.abspath(os.path.expanduser(output_dir))

        else:
            # On the ESI cluster, save results on HPC mount, otherwise use location of `func`
            if is_esi_node() or is_bic_node():
                outDir = f"/mnt/hpc/home/{getpass.getuser()}/"
            else:                                                       # pragma: no cover
                outDir = os.path.dirname(os.path.abspath(inspect.getfile(self.func)))
            outDir = os.path.join(outDir, f"ACME_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}")
        log.debug("Using output directory %s", outDir)

        # Unless specifically denied by the user, each worker stores results
        # separately with a common container file pointing to the individual
        # by-worker files residing in a "payload" directory
        self.out_dir = str(outDir)                                      # type: ignore
        if not single_file and not write_pickle:
            log.debug("Preparing payload directory for HDF5 containers")
            payloadName = f"{self.func.__name__}_payload"
            outputDir = os.path.join(self.out_dir, payloadName)         # type: ignore
        else:
            msg = "Either single-file output or pickling was requested. " +\
                "Not creating payload directory"
            log.debug(msg)
            outputDir = self.out_dir
        try:
            os.makedirs(outputDir)
            log.debug("Created %s", outputDir)
        except Exception as exc:
            msg = "%s automatic creation of output folder %s failed: %s"
            raise OSError(msg%(self.objName, outputDir, str(exc)))

        # Re-define or allocate key "taskID" to track concurrent processing results
        self.kwargv["taskID"] = self.task_ids
        self.collect_results = False                                    # type: ignore

        # Set up correct file-extension for output files; in case of HDF5
        # containers, prepare "main" file for collecting/symlinking worker results
        if write_pickle:
            fExt = "pickle"
            log.debug("Pickling was requested")
        else:
            fExt = "h5"
            self.results_container = os.path.join(self.out_dir, f"{self.func.__name__}.h5")     # type: ignore
            log.debug("Using HDF5 storage %s", self.results_container)

        # By default, `results_container` is a collection of links that point to
        # worker-generated HDF5 containers; if `single_file` is `True`, then
        # `results_container` is a "real" container with actual dataset(s)
        if single_file:
            self.kwargv["singleFile"] = [True]
            self.kwargv["outFile"] = [self.results_container]
            log.debug("Saving results in single HDF5 container")

            # If no output shape provided, prepare groups for storing datasets;
            # otherwise allocate a single dataset w/specified dimension
            if result_shape is None:
                msg = "Created group comp_%d in single shared results container"
                with h5py.File(self.results_container, "w") as h5f:
                    for i in self.task_ids:
                        h5f.create_group(f"comp_{i}")
                        log.debug(msg, i)
            else:
                if np.inf in self.result_shape:                                 # type: ignore
                    actShape = tuple(spec if spec is not np.inf else 1 for spec in self.result_shape)    # type: ignore
                    maxShape = tuple(spec if spec is not np.inf else None for spec in self.result_shape) # type: ignore
                else:
                    actShape = self.result_shape                                # type: ignore
                    maxShape = None
                msg = "Created unique dataset 'result_0' with shape %s " +\
                    "in single shared results container"
                with h5py.File(self.results_container, "w") as h5f:
                    h5f.create_dataset("result_0",
                                       shape=actShape,
                                       maxshape=maxShape,
                                       dtype=self.result_dtype)
                    log.debug(msg, str(self.result_shape))

        else:
            self.kwargv["outFile"] = [os.path.join(outputDir,
                                                   f"{self.func.__name__}_{taskID}.{fExt}")
                                                   for taskID in self.task_ids]
            if not write_pickle:

                # If no output shape provided, generate links to external datasets;
                # otherwise allocate a virtual dataset w/specified dimension
                if result_shape is None:

                    msg = "Created external link comp_%d pointing to " +\
                        "%s in results container"
                    with h5py.File(self.results_container, "w") as h5f:
                        for i, fname in enumerate(self.kwargv["outFile"]):
                            relPath = os.path.join(payloadName, os.path.basename(fname))
                            h5f[f"comp_{i}"] = h5py.ExternalLink(relPath, "/")
                            log.debug(msg, i, relPath)
                else:

                    VSourceShape = [spec if spec is not np.inf else None for spec in self.result_shape] # type: ignore
                    VSourceShape.pop(self.stacking_dim)                         # type: ignore
                    VSourceShape = tuple(VSourceShape)

                    # Account for resizable datasets
                    if None in VSourceShape:
                        resActShape = tuple(spec if spec is not np.inf else 1 for spec in self.result_shape)    # type: ignore
                        resMaxShape = tuple(spec if spec is not np.inf else None for spec in self.result_shape) # type: ignore
                        vsActShape = tuple(spec if spec is not None else 1 for spec in VSourceShape)
                        vsMaxShape = VSourceShape
                    else:
                        resActShape = self.result_shape                         # type: ignore
                        resMaxShape = None
                        vsActShape = VSourceShape
                        vsMaxShape = None
                    layout = h5py.VirtualLayout(shape=resActShape,
                                                dtype=self.result_dtype,
                                                maxshape=resMaxShape)   # type: ignore
                    idx = [slice(None) if spec is not np.inf else slice(h5py.h5s.UNLIMITED) for spec in self.result_shape] # type: ignore
                    jdx = list(idx)
                    jdx.pop(self.stacking_dim)                                  # type: ignore

                    msg = "Created virtual dataset result_0' with shape " +\
                        "%s in results container"
                    for i, fname in enumerate(self.kwargv["outFile"]):
                        idx[self.stacking_dim] = i                      # type: ignore
                        relPath = os.path.join(payloadName, os.path.basename(fname))
                        vsource = h5py.VirtualSource(fname, "result_0", shape=vsActShape, maxshape=vsMaxShape)
                        layout[tuple(idx)] = vsource[tuple(jdx)]
                    with h5py.File(self.results_container, "w", libver="latest") as h5f:
                        h5f.create_virtual_dataset("result_0", layout)
                        log.debug(msg, self.result_shape)

        # Include logger name in keywords so that workers can use it
        self.kwargv["logName"] = [log.name]

        # Wrap the user-provided func and distribute it across workers
        self.kwargv["userFunc"] = [self.func]
        self.acme_func = self.func_wrapper                              # type: ignore
        log.debug("Wrapping user-provided function inside func_wrapper")

        # Finally, attach verbosity flag to enable logging inside wrapper
        self.kwargv["logLevel"] = [log.level]

        return

    def perform_dryrun(
            self,
            setup_interactive: bool) -> bool:
        """
        Execute user function with one prepared randomly picked args, kwargs combo
        """

        # Let helper randomly pick a single scheduled job and prepare corresponding args + kwargs
        [dryRunIdx], [dryRunArgs], [dryRunKwargs] = self._dryrun_setup(n_runs=1)    # type: ignore

        # Create log entry
        msg = "Performing a single dry-run of %s simulating randomly " +\
            "picked worker #%d with automatically distributed arguments"
        log.info(msg, self.func.__name__, dryRunIdx)

        # Use resident memory size (in MB) to estimate job's memory footprint and measure elapsed time
        mem0 = psutil.Process().memory_info().rss / 1024 ** 2
        log.debug("Initial memory consumption estimate: %3.f MB", mem0)
        log.debug("Starting dryrun")
        tic = time.perf_counter()
        self.acme_func(*dryRunArgs, **dryRunKwargs)                     # type: ignore
        toc = time.perf_counter()
        log.debug("Finished dryrun")
        mem1 = psutil.Process().memory_info().rss / 1024 ** 2
        log.debug("Memory consumption estimate after dryrun: %3.f MB", mem1)

        # Remove any generated output files
        if self.out_dir is not None:
            log.debug("Removing %s generated during dryrun", self.kwargv["outFile"][dryRunIdx])
            os.unlink(self.kwargv["outFile"][dryRunIdx])

        # Compute elapsed time and memory usage
        elapsedTime = toc - tic
        memUsage = mem1 - mem0

        # Prepare info message
        memUnit = "MB"
        if memUsage > 1000:
            memUsage /= 1024
            memUnit = "GB"
        msg = "Dry-run completed. Elapsed time is %f seconds, " +\
            "estimated memory consumption was %3.2f %s."
        log.info(msg, elapsedTime, memUsage, memUnit)

        # If the worker setup is supposed to be interactive, ask for confirmation
        # here as well; if execution is terminated, remove auto-generated output directory
        goOn = True
        if setup_interactive:
            msg = f"Do you want to continue executing {self.func.__name__} with the provided arguments?"
            if not user_yesno(msg, default="yes"):
                if self.out_dir is not None:
                    shutil.rmtree(self.out_dir, ignore_errors=True)
                goOn = False
        return goOn

    def _dryrun_setup(
            self,
            n_runs: Optional[int] = None) -> tuple[ArrayLike, List, List]:
        """
        Pick scheduled job(s) at random and extract corresponding (already prepared!) args + kwargs
        """

        # If not provided, attempt to infer a "sane" default for the number of jobs to pick
        if n_runs is None:
            n_runs = min(self.n_calls, max(5, min(1, int(0.05 * self.n_calls))))
        log.debug("Picking %d jobs at random", n_runs)

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
            n_workers: Union[int, str] = "auto",
            partition: str = "auto",
            mem_per_worker: str = "auto",
            setup_timeout: int = 60,
            setup_interactive: bool = True,
            stop_client: Union[bool, str] = "auto") -> None:
        """
        Setup or fetch dask distributed processing client. Depending on available
        hardware, either start a local multi-processing client or launch a
        worker cluster via SLURM.

        Also ensure that ad-hoc clients created here are stopped and worker jobs
        are properly released at the end of computation. However, ensure any client
        not created by `prepare_client` is **not** automatically cleaned up.
        """

        # Modify automatic setting of `stop_client` if requested
        msg = "%s `stop_client` has to be 'auto' or Boolean, not %s"
        if isinstance(stop_client, str):
            if stop_client != "auto":
                raise ValueError(msg%(self.objName, stop_client))
        elif isinstance(stop_client, bool):
            self.stop_client = stop_client                              # type: ignore
        else:
            raise TypeError(msg%(self.objName, str(type(stop_client))))
        log.debug("Using `stop_client = %s`", str(stop_client))

        # Check if a dask client is already running
        try:
            self.client = dd.get_client()                               # type: ignore
            log.debug("Detected running client %s", str(self.client))
            if stop_client == "auto":
                self.stop_client = False                                # type: ignore
                msg = "Changing `stop_client` from `'auto'` to `False` " +\
                    "to not terminate external client"
                log.debug(msg)
            self.n_workers = count_online_workers(self.client.cluster)  # type: ignore
            log.debug("Found %d alive workers in the client", self.n_workers)
            msg = "Attaching to parallel computing client %s"
            log.info(msg%(str(self.client)))
            return
        except ValueError:
            msg = "No running client detected, preparing to start a new one"
            log.debug(msg)
            if stop_client == "auto":
                self.stop_client = True                                 # type: ignore
                msg = "Changing `stop_client` from `'auto'` to `True` " +\
                    "to clean up client started by `ParallelMap`"
                log.debug(msg)

            # If `n_workers` is `"auto`, set `n_workers = n_calls` (default)
            msg = "%s `n_workers` has to be 'auto' or an integer >= 1, not %s"
            if isinstance(n_workers, str):
                if n_workers != "auto":
                    raise ValueError(msg%(self.objName, n_workers))
                if self.has_slurm:
                    n_workers = self.n_calls
                else:
                    n_workers = None                                            # type: ignore
                log.debug("Changing `n_workers` from `'auto'` to %s", str(n_workers))
            log.debug("Using provided `n_workers = %d` to start client", n_workers)

        # If things are running locally, simply fire up a dask-distributed client,
        # otherwise go through the motions of preparing a full worker cluster
        if not self.has_slurm:                                          # pragma: no cover

            log.debug("SLURM not found, Calling `local_cluster_setup`")
            self.client = local_cluster_setup(n_workers=n_workers, interactive=False)        # type: ignore

        else:

            # If `partition` is "auto", use `estimate_memuse` to heuristically determine
            # average memory consumption of jobs
            log.debug("SLURM available parsing settings")
            if not isinstance(partition, str):
                msg = "%s `partition` has to be 'auto' or a valid SLURM partition name, not %s"
                raise TypeError(msg%(self.objName, str(type(partition))))
            if partition == "auto":
                if is_esi_node() or is_bic_node():
                    msg = "Automatic SLURM partition selection is experimental"
                    log.warning(msg)
                    mem_per_worker = self.estimate_memuse()
                else:                                                   # pragma: no cover
                    err = "Automatic SLURM partition selection currently only available " +\
                        "on ESI/CoBIC HPC clusters "
                    log.error(err)

            # All set, remaining input processing is done by respective `*_cluster_setup` routines
            if is_esi_node():
                msg = "Running on ESI compute node, Calling `esi_cluster_setup`"
                log.debug(msg)
                self.client = esi_cluster_setup(partition=partition, n_workers=n_workers,               # type: ignore
                                                mem_per_worker=mem_per_worker, timeout=setup_timeout,
                                                interactive=setup_interactive, start_client=True)

            # All set, remaining input processing is done by respective `*_cluster_setup` routines
            elif is_bic_node():
                msg = "Running on CoBIC compute node, Calling `bic_cluster_setup`"
                log.debug(msg)
                self.client = bic_cluster_setup(partition=partition, n_workers=n_workers,               # type: ignore
                                                mem_per_worker=mem_per_worker, timeout=setup_timeout,
                                                interactive=setup_interactive, start_client=True)

            # Unknown cluster node, use vanilla config
            else:                                                       # pragma: no cover
                wrng = "Cluster node %s not recognized. Falling back to vanilla " +\
                    "SLURM setup allocating one worker and one core per worker"
                log.warning(wrng%(socket.getfqdn()))
                processes_per_worker = 1
                n_cores = 1
                self.client = slurm_cluster_setup(partition=partition,                                  # type: ignore
                                                  n_cores=n_cores,
                                                  n_workers=n_workers,                                  # type: ignore
                                                  processes_per_worker=processes_per_worker,
                                                  mem_per_worker=mem_per_worker,
                                                  n_workers_startup=1,
                                                  timeout=setup_timeout,
                                                  interactive=setup_interactive,
                                                  interactive_wait=120,
                                                  start_client=True,
                                                  job_extra=[],
                                                  invalid_partitions=[])

            # If startup is aborted by user, get outta here
            if self.client is None:                                     # pragma: no cover
                msg = "%s Could not start distributed computing client. "
                raise ConnectionAbortedError(msg%(self.objName))

        # Set `n_workers` to no. of active workers in the initialized cluster
        self.n_workers = len(self.client.cluster.workers)               # type: ignore
        log.debug("Setting `n_workers = %d` based on active workers in %s",
                  self.n_workers, str(self.client))

        # If single output file saving was chosen, initialize distributed
        # lock for shared writing to container
        if self.kwargv.get("singleFile") is not None:
            msg = "Initializing distributed lock for writing to single shared results container"
            log.debug(msg)
            dd.lock.Lock(name=os.path.basename(self.results_container)) # type: ignore

        return

    def estimate_memuse(self) -> str:
        """
        A brute-force guessing approach to determine memory consumption of provided
        workload
        """

        # Let helper randomly pick some jobs and prepare corresponding args + kwargs
        dryRunIdx, dryRunArgs, dryRunKwargs = self._dryrun_setup()

        # Append new dummy keyword to return before any disk-writes happen
        # in case ACME handles results output
        if self.out_dir is not None:
            msg = "Appending `memEstRun` keyword to func_wrapper to prevent any disk-writes"
            log.debug(msg)
            for k in range(len(dryRunKwargs)):
                dryRunKwargs[k]["memEstRun"] = True

        # Set run-time for each job (in seconds) and allocate arrays for memory
        # consumption estimates (in GB)
        runTime = 30
        memPerSec = np.zeros((runTime,))
        memPerJob = np.zeros((len(dryRunIdx),))                         # type: ignore

        # Adequately warn about this heuristic gymnastics...
        log.info("Estimating memory footprint of %s", self.func.__name__)

        msg = "Running %d random workers evaluating %s for max. %d seconds"
        log.debug(msg%(len(dryRunIdx), self.func.__name__, runTime))    # type: ignore
        for i, idx in enumerate(dryRunIdx):                             # type: ignore

            # Set up dedicated process to execute user-provided function w/allocated args/kwargs
            proc = multiprocessing.Process(target=self.acme_func,
                                           args=dryRunArgs[i],
                                           kwargs=dryRunKwargs[i])

            # Run user-func for max. `runTime` seconds (or worker finishes),
            # get memory footprint every second
            proc.start()
            with tqdm.tqdm(desc=f"Launching worker #{idx}",                     # type: ignore
                           total=runTime,
                           bar_format=self.tqdmFormat,
                           position=0) as pbar:
                for k in range(runTime):
                    memPerSec[k] = psutil.Process(proc.pid).memory_info().rss / 1024 ** 3
                    time.sleep(1)
                    pbar.update(1)
                    if not proc.is_alive():
                        pbar.n = runTime
                        break
            proc.kill()

            # Compute peak memory consumption across `runTime` seconds
            memPerJob[i] = memPerSec.max()
            log.debug("Peak memory for worker #%d: %3.2f GB", idx, memPerJob[i])

        # Compute aggregate average memory consumption across all runs
        memUsage = memPerJob.mean()

        # Communicate results
        msg = "Estimated memory consumption across %d runs is %3.2f GB "
        log.info(msg%(len(dryRunIdx), memUsage))                        # type: ignore

        # Return specially formatted string
        mem_per_worker = "estimate_memuse:" + str(max(1, int(np.ceil(memUsage))))
        log.debug("Finished memory estimation, returning `mem_per_worker = %s`", mem_per_worker)
        return mem_per_worker

    def compute(
            self,
            debug: bool = False) -> Union[List, None]:
        """
        Perform the actual parallel execution of `func`

        If `debug` is `True`, use a single-threaded dask scheduler that does
        not actually process anything concurrently but uses the dask framework
        in a sequential setup.
        """

        # If `prepare_client` has not been called yet, don't attempt to compute anything
        if self.client is None:
            log.debug("No parallel computing client allocated, exiting")
            return None

        # Ensure `debug` is a simple Boolean flag
        if not isinstance(debug, bool):
            msg = "%s `debug` has to be `True` or `False`, not %s"
            raise TypeError(msg%(self.objName, str(type(debug))))
        log.debug("Found `debug = %s`", str(debug))

        # Check if the underlying parallel computing cluster hosts actually usable workers
        if count_online_workers(self.client.cluster) == 0:
            msg = "%s no active workers found in distributed computing client %s " +\
                "Consider running \n" +\
                "\timport dask.distributed as dd; dd.get_client().restart()\n" +\
                "If this fails to make workers come online, please use\n" +\
                "\timport acme; acme.cluster_cleanup()\n" +\
                "to shut down any defunct distributed computing clients"
            raise RuntimeError(msg%(self.objName, self.client))
        log.debug("Found %d workers in client %s",
                  count_online_workers(self.client.cluster), str(self.client))

        # Dask does not correctly forward the `sys.path` from the parent process
        # to its workers. Fix this.
        def init_acme(dask_worker, syspath):
            sys.path = list(syspath)
        self.client.register_worker_callbacks(setup=functools.partial(init_acme, syspath=sys.path))
        log.debug("Registered worker callback to forward `sys.path`")

        # Format positional arguments for worker-distribution: broadcast all
        # inputs that are used by all workers and create a list of references
        # to this (single!) future on the cluster for submission
        for ak, arg in enumerate(self.argv):
            if len(arg) == 1:
                ftArg = self.client.scatter(arg, broadcast=True)
                log.debug("Broadcasting single-element pos arg %s to client", str(arg))
                if isinstance(ftArg, collections.abc.Sized):
                    ftArg = ftArg[0]
                self.argv[ak] = [ftArg] * self.n_calls

        # Same as above but for keyword-arguments
        for name, value in self.kwargv.items():
            if len(value) == 1:
                ftVal = self.client.scatter(value, broadcast=True)[0]
                self.kwargv[name] = [ftVal] * self.n_calls
                log.debug("Broadcasting single-element kwarg `%s` to client", name)

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
            log.warning("Running in debug mode")
            with dask.config.set(scheduler='single-threaded'):
                log.debug("Using single-threaded scheduler to evaluate function")
                values = self.client.gather([self.client.submit(self.acme_func, *args, **kwargs) \
                    for args, kwargs in zip(zip(*self.argv), kwargList)])
                return values

        # Depending on the used dask cluster object, point to respective log info
        if isinstance(self.client.cluster, SLURMCluster):
            logFiles = self.client.cluster.job_header.split("--output=")[1].replace("%j", "{}")
            logDir = os.path.split(logFiles)[0]
        else:                                                           # pragma: no cover
            logFiles = []
            logDir = os.path.dirname(self.client.cluster.dashboard_link) + "/info/main/workers.html"
        msg = "Preparing %d parallel calls of `%s` using %d workers"
        log.info(msg%(self.n_calls, self.func.__name__, self.n_workers))
        msg = "Log information available at %s"
        log.debug(msg%(logDir))

        # Submit `self.n_calls` function calls to the cluster
        log.debug("Submitting %d function calls to client %s", self.n_calls, str(self.client))
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
        log.debug("Waiting %f seconds for futures", self.sleepTime)
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
            msg = "%s Parallel computation failed: %d/%d tasks failed or stalled. "
            msg = msg%(self.objName, totalTasks - finishedTasks, totalTasks)
            msg += "Concurrent computing scheduler log info: "
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
                    msg += "Please check SLURM logs in %s"%(logDir)

            # In case of a `LocalCluster`, syphon worker logs
            else:                                                       # pragma: no cover
                msg += "Parallel worker log details: \n"
                workerLogs = self.client.get_worker_logs().values()
                for wLog in workerLogs:
                    if "Failed" in wLog:
                        msg += wLog

            # Finally, raise an error and get outta here
            raise RuntimeError(msg)

        # Postprocessing of results
        values = self.post_process(futures)

        # Either return collected by-worker results or the filepaths of results
        return values

    def post_process(
            self,
            futures: dd.Future) -> Union[List, None]:
        """
        Local helper to post-process results on disk/in-memory

        The return `values` is either
        `None` : if neither in-memory results collection or auto-writing was requested
        list of file-names: if `write_worker_results` is `True`
        list of objects: if in-memory results collection was requested
        """

        # Deduce result output information
        write_worker_results = self.acme_func == self.func_wrapper
        single_file = False
        if write_worker_results:
            write_pickle = self.results_container is None
            if not write_pickle and self.kwargv.get("singleFile") is not None:
                single_file = True
        else:
            write_pickle = False
        msg = "Inferred that `write_worker_results = %s`, `single_file = %s`, `write_pickle = %s`"
        log.debug(msg, str(write_worker_results), str(single_file), str(write_pickle))

        # If wanted (not recommended) collect computed results in local memory
        if self.collect_results:
            if not isSpyModule:
                log.info("Gathering results in local memory")
            collected = self.client.gather(futures)
            log.debug("Gathered results from client in a %d-element list", len(collected))
            if self.result_shape is not None:
                log.debug("Returning single NumPy array of shape %s and type %s",
                          str(self.result_shape), str(self.result_dtype))
                values = []
                arrVal = np.empty(shape=self.result_shape, dtype=self.result_dtype)
                idx = [slice(None)] * len(self.result_shape)
                for i, res in enumerate(collected):
                    if not isinstance(res, (list, tuple)):
                        res = [res]
                    idx[self.stacking_dim] = i
                    arrVal[tuple(idx)] = res[0]
                    for r in res[1:]:
                        values.append(r)
                values.insert(0, arrVal)
                # If `values` is a single array, don't wrap it inside a list
                if len(values) == 1:
                    values = values[0]
            else:
                log.debug("Returning a list of values")
                values = collected
        else:
            values = None

        # Prepare final output message
        successMsg = "SUCCESS!"

        # If automatic results writing was requested, perform some housekeeping
        if write_worker_results:
            finalMsg = "Results have been saved to %s"
            if write_pickle:
                log.debug("Saved results as pickle files")
                values = list(self.kwargv["outFile"])
                finalMsg = finalMsg%(self.out_dir)
                log.debug("Returning a list of file-names")
            else:
                if single_file:
                    log.debug("Saved results to single shared container")
                    finalMsg = finalMsg%(self.results_container)
                    if values is None:
                        values = [self.results_container]
                        log.debug("Returning container name as single-element list")
                else:
                    log.debug("Scanning payload directory for emergency pickles")
                    picklesFound = False
                    values = []
                    for fname in self.kwargv["outFile"]:
                        pklName = fname.rstrip(".h5") + ".pickle"
                        if os.path.isfile(fname):
                            values.append(fname)
                        elif os.path.isfile(pklName):
                            values.append(pklName)
                            picklesFound = True
                            log.debug("Found emergency pickle %s", pklName)
                        else:
                            missing = fname.rstrip(".h5")
                            values.append("Missing %s"%(missing))
                            log.debug("Missing file %s", missing)
                    payloadDir = os.path.dirname(values[0])

                    # If pickles are found, remove global `results_container` as it
                    # would contain invalid file-links and move compute results out
                    # of payload dir
                    if picklesFound:
                        os.unlink(self.results_container)               # type: ignore
                        wrng = "Some compute runs could not be saved as HDF5, " +\
                            "collection container %s has been removed as it would " +\
                                "comprise invalid file-links"
                        log.warning(wrng, self.results_container)
                        self.results_container = None

                        # Move files out of payload dir and update return `values`
                        target = os.path.abspath(os.path.join(payloadDir, os.pardir))
                        for i, fname in enumerate(values):
                            shutil.move(fname, target)
                            self.kwargv["outFile"][i] = os.path.join(target, os.path.basename(fname))
                            log.debug("Moved %s to %s", fname, target)
                        values = list(self.kwargv["outFile"])
                        log.debug("Returning a list of file-names")
                        shutil.rmtree(payloadDir)
                        log.debug("Deleted payload directory %s", payloadDir)
                        successMsg = ""
                        finalMsg = finalMsg%(target)

                    # All good, no pickle gymnastics was needed
                    else:

                        # In case of multiple return values present in by-worker
                        # containers but missing in collection container (happens
                        # if `result_shape` is not `None` and data-sets have to
                        # be pre-allocated), create "symlinks" to corresponding
                        # missing returns
                        log.debug("No emergency pickles found")
                        if self.stacking_dim is not None:
                            msg = "Check if additional return values " +\
                                "need to be added to container with pre-allocated dataset"
                            log.debug(msg)
                            with h5py.File(self.results_container, "r") as h5r:
                                with h5py.File(values[0], "r") as h5Tmp:
                                    missingReturns = set(h5Tmp.keys()).difference(h5r.keys())
                            if len(missingReturns) > 0:
                                log.debug("Found return values to be added")
                                with h5py.File(self.results_container, "a") as h5r:
                                    for retVal in missingReturns:
                                        for i, fname in enumerate(values):
                                            relPath = os.path.join(os.path.basename(payloadDir), os.path.basename(fname))
                                            h5r[f"comp_{i}/{retVal}"] = h5py.ExternalLink(relPath, retVal)
                                            log.debug("Added return value via external link comp_%d/%s", i, retVal)

                        finalMsg = finalMsg%(self.results_container)
                        msg = "Container ready, links to data payload located in %s"
                        log.debug(msg, payloadDir)
                        log.debug("Returning a list of file-names")
        else:
            finalMsg = "Finished parallel computation"

        # Print final triumphant output message and force-flush all logging handlers
        if len(successMsg) > 0:
            log.announce(successMsg)                                    # type: ignore
        log.info(finalMsg)
        for h in log.handlers:
            if hasattr(h, "flush"):
                h.flush()
        return values

    def cleanup(self) -> None:
        """
        Shut down any ad-hoc distributed computing clients created by `prepare_client`
        """

        # If `prepare_client` has not been launched yet, just get outta here
        if not hasattr(self, "client"):                                 # pragma: no cover
            log.debug("Helper `prepare_client` not yet launched, exiting")
            return
        if self.stop_client and self.client is not None:
            log.debug("Found client %s, calling `cluster_cleanup`", str(self.client))
            cluster_cleanup(self.client)
            self.client = None
            return
        log.debug("Either `stop_client = False` or no client found, returning")

        return

    @staticmethod
    def func_wrapper(*args: Any, **kwargs: Optional[Any]) -> None:      # pragma: no cover
        """
        If the output of `func` is saved to disk, wrap `func` with this static
        method to take care of filling up HDF5/pickle files

        If writing to HDF5 fails, use an "emergency-pickling" mechanism to try
        to save the output of `func` using pickle instead
        """

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
        log.setLevel(logLevel)                                          # type: ignore
        for h in log.handlers:
            h.setLevel(logLevel)                                        # type: ignore

        # Call user-provided function
        result = func(*args, **kwargs)                                  # type: ignore

        # For memory estimation runs, don't start saving stuff
        if memEstRun:
            return

        # Save results: either (try to) use HDF5 or pickle stuff
        if fname.endswith(".h5"):                                       # type: ignore

            grpName = ""
            if singleFile:
                lock = dd.lock.Lock(name=os.path.basename(fname))       # type: ignore
                lock.acquire()
                grpName = f"comp_{taskID}/"

            if not isinstance(result, (list, tuple)):
                result = [result]

            try:

                with h5py.File(fname, "a") as h5f:
                    if stackingDim is None:
                        if not all(isinstance(value, (numbers.Number, str)) for value in result):
                            for rk, res in enumerate(result):
                                h5f.create_dataset(f"{grpName}result_{rk}", data=res)
                                log.debug("Created new dataset `result_%d` in %s", rk, fname)
                        else:
                            h5f.create_dataset(grpName + "result_0", data=result)
                            log.debug("Created new dataset `result_0` in %s", fname)
                    else:
                        if singleFile:
                            dset = h5f["result_0"]
                            idx = [slice(None)] * len(dset.shape)    # type: ignore
                            idx[stackingDim] = taskID                           # type: ignore
                            if None in dset.maxshape:
                                if len(result[0].shape) < len(idx):
                                    lenDim = list(set(result[0].shape).difference(dset.maxshape))
                                    if len(lenDim) == 0:
                                        lenDim = result[0].shape[0]
                                    else:
                                        lenDim = lenDim[0]
                                    actShape = tuple(spec if spec is not None else lenDim for spec in dset.maxshape)
                                else:
                                    actShape = list(result[0].shape)                    # type: ignore
                                    actShape[stackingDim] = dset.maxshape[stackingDim]  # type: ignore
                                    actShape = tuple(actShape)
                                dset.resize(actShape)
                            dset[tuple(idx)] = result[0]
                            log.debug("Wrote to pre-allocated dataset `result_0` in %s", fname)
                            for rk, res in enumerate(result[1:]):
                                h5f.create_dataset(f"{grpName}result_{rk + 1}", data=res)
                                log.debug("Created new dataset `result_%d` in %s", rk + 1, fname)
                        else:
                            for rk, res in enumerate(result):
                                h5f.create_dataset(f"{grpName}result_{rk}", data=res)
                                log.debug("Created new dataset `result_%d` in %s", rk, fname)

                if singleFile:
                    lock.release()

            except TypeError as exc:

                if ("has no native HDF5 equivalent" in str(exc) \
                    or "One of data, shape or dtype must be specified" in str(exc)) \
                        and not singleFile:
                    try:
                        os.unlink(fname)                                # type: ignore
                        pname = fname.rstrip(".h5") + ".pickle"         # type: ignore
                        with open(os.path.join(pname), "wb") as pkf:
                            pickle.dump(result, pkf)
                        msg = "Could not write %s results have been pickled instead: %s. Return values are most likely " +\
                            "not suitable for storage in HDF5 containers. Original error message: %s"
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
                raise exc

        else:

            try:
                with open(os.path.join(fname), "wb") as pkf:            # type: ignore
                    pickle.dump(result, pkf)
                    log.debug("Pickled to %s", fname)
            except pickle.PicklingError as pexc:
                err = "Could not pickle results to file %s. Original error message: %s"
                log.error(err, fname, str(pexc))
                raise pexc

        return
