#
# Memory profiling utilities for ACME
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import time
import multiprocessing
import psutil
import tqdm
import numpy as np
import logging
from typing import Optional, List, Tuple, Any, Callable
from numpy.typing import ArrayLike

# Local imports
from .config import ACMEConfig

# Fetch logger
log = logging.getLogger("ACME")


class MemoryEstimationError(Exception):
    """
    Memory estimation failed
    """
    pass


class MemoryProfiler:
    """
    Estimate memory consumption of user functions for SLURM resource allocation
    """

    def __init__(self, func: Callable, tqdm_format: str = "{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        """
        Initialize memory profiler

        Parameters
        ----------
        func : callable
            User function to profile
        tqdm_format : str
            Progress bar format string
        """
        self.func = func
        self.tqdm_format = tqdm_format

    def estimate_memory(
        self,
        dryrun_setup_func: Callable,
        output_dir: Optional[str] = None,
        run_time: int = 30
    ) -> str:
        """
        Estimate memory consumption by running sample jobs

        Parameters
        ----------
        dryrun_setup_func : callable
            Function that returns (indices, args, kwargs) for dryrun
        output_dir : str, optional
            Output directory path
        run_time : int
            Maximum run time per job in seconds

        Returns
        -------
        str
            Formatted memory string for SLURM (e.g., "estimate_memuse:4")
        """
        # Let helper randomly pick some jobs and prepare corresponding args + kwargs
        dry_run_idx, dry_run_args, dry_run_kwargs = dryrun_setup_func()

        # Append new dummy keyword to return before any disk-writes happen
        # in case ACME handles results output
        if output_dir is not None:
            msg = "Appending `memEstRun` keyword to func_wrapper to prevent any disk-writes"
            log.debug(msg)
            for k in range(len(dry_run_kwargs)):
                dry_run_kwargs[k]["memEstRun"] = True

        # Set run-time for each job (in seconds) and allocate arrays for memory
        # consumption estimates (in GB)
        mem_per_sec = np.zeros((run_time,))
        mem_per_job = np.zeros((len(dry_run_idx),))  # type: ignore

        # Adequately warn about this heuristic gymnastics...
        log.info("Estimating memory footprint of %s", self.func.__name__)

        msg = "Running %d random workers evaluating %s for max. %d seconds"
        log.debug(msg % (len(dry_run_idx), self.func.__name__, run_time))  # type: ignore
        for i, idx in enumerate(dry_run_idx):  # type: ignore

            # Set up dedicated process to execute user-provided function w/allocated args/kwargs
            proc = multiprocessing.Process(
                target=self.func, args=dry_run_args[i], kwargs=dry_run_kwargs[i]
            )

            # Run user-func for max. `runTime` seconds (or worker finishes),
            # get memory footprint every second
            proc.start()
            with tqdm.tqdm(
                desc=f"Launching worker #{idx}",  # type: ignore
                total=run_time,
                bar_format=self.tqdm_format,
                position=0,
            ) as pbar:
                for k in range(run_time):
                    mem_per_sec[k] = psutil.Process(proc.pid).memory_info().rss / 1024**3
                    time.sleep(1)
                    pbar.update(1)
                    if not proc.is_alive():
                        pbar.n = run_time
                        break
            proc.kill()

            # Compute peak memory consumption across `runTime` seconds
            mem_per_job[i] = mem_per_sec.max()
            log.debug("Peak memory for worker #%d: %3.2f GB", idx, mem_per_job[i])

        # Compute aggregate average memory consumption across all runs
        mem_usage = mem_per_job.mean()

        # Communicate results
        msg = "Estimated memory consumption across %d runs is %3.2f GB "
        log.info(msg % (len(dry_run_idx), mem_usage))  # type: ignore

        # Return specially formatted string
        mem_per_worker = "estimate_memuse:" + str(max(1, int(np.ceil(mem_usage))))
        log.debug(
            "Finished memory estimation, returning `mem_per_worker = %s`",
            mem_per_worker,
        )
        return mem_per_worker