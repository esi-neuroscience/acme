#
# Memory profiling utilities for ACME
#
# Copyright © 2026 Ernst Strüngmann Institute (ESI) of the Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import os
import shutil
import time
import multiprocessing
import psutil
import tqdm
import numpy as np
import logging
from typing import Optional, Callable

# Local imports
from .argument_processor import ArgumentProcessor
from .shared import user_yesno

# Fetch logger
log = logging.getLogger("ACME")


class MemoryProfiler:
    """
    Estimate memory consumption of user functions for SLURM resource allocation
    """

    def __init__(
        self,
        argprocessor: ArgumentProcessor,
        func: Callable,
        func_name: str,
        tqdm_format: str = "{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        """
        Initialize memory profiler

        Parameters
        ----------
        func : callable
            User function to profile
        tqdm_format : str
            Progress bar format string
        """
        self.processor = argprocessor
        self.func = func
        self.func_name = func_name
        self.tqdm_format = tqdm_format

    def estimate_memory(
        self,
        output_dir: Optional[str] = None,
        run_time: int = 30,
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
        dry_run_idx, dry_run_args, dry_run_kwargs = self.processor.dryrun_setup()

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
                    mem_per_sec[k] = (
                        psutil.Process(proc.pid).memory_info().rss / 1024**3
                    )
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

    def perform_dryrun(
        self, output_dir: Optional[str] = None, setup_interactive: bool = True
    ) -> bool:
        """
        Execute user function with one prepared randomly picked args, kwargs combo
        """

        # Let helper randomly pick a single scheduled job and prepare corresponding args + kwargs
        [dryRunIdx], [dryRunArgs], [dryRunKwargs] = self.processor.dryrun_setup(n_runs=1)  # type: ignore

        # Create log entry
        msg = (
            "Performing a single dry-run of %s simulating randomly "
            + "picked worker #%d with automatically distributed arguments"
        )
        log.info(msg, self.func_name, dryRunIdx)

        # Use resident memory size (in MB) to estimate job's memory footprint and measure elapsed time
        mem0 = psutil.Process().memory_info().rss / 1024**2
        log.debug("Initial memory consumption estimate: %3.f MB", mem0)
        log.debug("Starting dryrun")
        tic = time.perf_counter()
        self.func(*dryRunArgs, **dryRunKwargs)  # type: ignore
        toc = time.perf_counter()
        log.debug("Finished dryrun")
        mem1 = psutil.Process().memory_info().rss / 1024**2
        log.debug("Memory consumption estimate after dryrun: %3.f MB", mem1)

        # Remove any generated output files
        if output_dir is not None:
            log.debug(
                "Removing %s generated during dryrun",
                self.processor.kwargv["outFile"][dryRunIdx],
            )
            os.unlink(self.processor.kwargv["outFile"][dryRunIdx])

        # Compute elapsed time and memory usage
        elapsedTime = toc - tic
        memUsage = mem1 - mem0

        # Prepare info message
        memUnit = "MB"
        if memUsage > 1000:
            memUsage /= 1024
            memUnit = "GB"
        msg = (
            "Dry-run completed. Elapsed time is %f seconds, "
            + "estimated memory consumption was %3.2f %s."
        )
        log.info(msg, elapsedTime, memUsage, memUnit)

        # If the worker setup is supposed to be interactive, ask for confirmation
        # here as well; if execution is terminated, remove auto-generated output directory
        goOn = True
        if setup_interactive:
            msg = f"Do you want to continue executing {self.func_name} with the provided arguments?"
            if not user_yesno(msg, default="yes"):
                if output_dir is not None:
                    shutil.rmtree(output_dir, ignore_errors=True)
                goOn = False
        return goOn
