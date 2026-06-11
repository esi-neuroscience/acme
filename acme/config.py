#
# Configuration dataclasses for ACME
#
# Copyright © 2026 Ernst Strüngmann Institute (ESI) of the Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import logging
from dask.distributed import Client
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Dict, List
import numpy as np

# Local imports
from .validators import (
    validate_output_flags,
    validate_result_shape,
    validate_stop_client,
    validate_n_workers,
    validate_logfile,
    validate_outputdir,
    validate_boolean,
    validate_partition,
)
from .shared import is_slurm_node, _scalar_parser

# Fetch logger
log = logging.getLogger("ACME")


@dataclass
class ACMEConfig:
    """
    Configuration container for ACMEdaemon execution

    Centralizes all configuration parameters for parallel computation,
    making it easier to pass settings around and validate them.
    """

    # Basic state from ParallelMap
    func: Callable = None
    argv: List = None
    kwargv: Dict = field(default_factory=dict)
    n_calls: int = 1
    task_ids: List[int] = field(default_factory=list)
    has_slurm: bool = False

    # Execution settings
    n_workers: Union[int, str] = "auto"
    dryrun: bool = False

    # Resource management
    setup_timeout: int = 60
    setup_interactive: bool = True
    stop_client: Union[bool, str] = "auto"

    # Result handling
    write_worker_results: bool = True
    write_pickle: bool = False
    single_file: bool = False

    # Output configuration
    output_dir: Optional[str] = None
    result_shape: Optional[tuple[Optional[int], ...]] = None
    result_dtype: str = "float"

    # Cluster settings
    partition: str = "auto"
    mem_per_worker: str = "auto"

    # Logging
    verbose: Optional[bool] = None
    logfile: Optional[Union[bool, str]] = None
    
    # Cleanup configuration
    cleanup_threshold_days: Optional[int] = None

    # Internal state
    acme_func: Optional[Callable] = None
    collect_results: Optional[bool] = True
    results_container: Optional[str] = None
    stacking_dim: Optional[int] = None
    client: Optional[Client] = None

    # Static attributes not accessible by user
    tqdmFormat = (
        "{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    sleepTime = 0.1

    def validate(self) -> None:
        """
        Validate configuration using validators module

        The order of validation is important! For instance, `logfile` cannot be
        properly validated if `output_dir` has not been set up yet!

        Raises
        ------
        TypeError
            If configuration values have wrong types
        ValueError
            If configuration values are invalid
        """

        # Extract state from ParallelMap
        self.task_ids = list(range(self.n_calls))
        self.has_slurm = is_slurm_node()

        # Create config dictionary for boolean flags validation
        output_config = {
            "write_worker_results": self.write_worker_results,
            "write_pickle": self.write_pickle,
            "single_file": self.single_file,
            "output_dir": self.output_dir,
            "result_shape": self.result_shape,
        }
        validate_output_flags(output_config)

        # Validate output directory
        if self.write_worker_results:
            self.output_dir = validate_outputdir(self.output_dir, self.func, self.cleanup_threshold_days)
        else:
            self.output_dir = None

        # Validate verbose setting
        if self.verbose is not None:
            validate_boolean(self.verbose, name="verbose")

        # Validate logfile
        self.logfile = validate_logfile(
            self.logfile,
            self.write_worker_results,
            self.func,
            self.output_dir,
            self.verbose,
        )

        # Validate result shape if provided
        if self.result_shape is not None:
            self.result_shape, self.stacking_dim, self.result_dtype = (
                validate_result_shape(
                    self.result_shape,
                    self.result_dtype,
                    self.n_calls,
                    self.write_worker_results,
                )
            )
            if self.write_worker_results:
                self.kwargv["stackingDim"] = [self.stacking_dim]

        # Validate n_workers
        self.n_workers = validate_n_workers(
            self.n_workers, self.n_calls, self.has_slurm
        )

        # Validate Booleans
        validate_boolean(self.dryrun, "dryrun")
        validate_boolean(self.setup_interactive, "setup_interactive")

        # Validate timeout for starting computing client
        try:
            _scalar_parser(
                self.setup_timeout,
                varname="setup_timeout",
                ntype="int_like",
                lims=[0, np.inf],
            )
        except Exception as exc:
            log.error("Error parsing `setup_timeout`")
            raise exc

        # Validate stop_client
        validate_stop_client(self.stop_client)

        # Validate partition
        if self.has_slurm:
            validate_partition(self.partition)

        # Basic sanity check for mem spec
        if not isinstance(self.mem_per_worker, str):
            err = "Memory specification `mem_per_worker` has to be a string, not %s"
            log.error(err, str(type(self.mem_per_worker)))
            raise TypeError(err % (str(type(self.mem_per_worker))))

        return
