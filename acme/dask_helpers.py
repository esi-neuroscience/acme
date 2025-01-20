#
# Helper routines for working w/dask
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import os
import sys
import socket
import platform
import subprocess
import getpass
import time
import inspect
import textwrap
import numpy as np
from tqdm import tqdm
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, get_client, LocalCluster
from datetime import datetime, timedelta
from typing import List, Optional, Any, Union

# Local imports
from acme import __deprecated__, __deprecation_wrng__
from .shared import user_input, user_yesno, is_jupyter
from .spy_interface import scalar_parser, log

__all__: List["str"] = ["esi_cluster_setup", "local_cluster_setup", "cluster_cleanup", "slurm_cluster_setup"]


# Setup SLURM workers on the ESI HPC cluster
def esi_cluster_setup(
        partition: str,
        n_workers: int = 2,
        mem_per_worker: str = "auto",
        cores_per_worker: Optional[int] = None,
        n_workers_startup: int = 1,
        timeout: int = 60,
        interactive: bool = True,
        interactive_wait: int = 120,
        start_client: bool = True,
        job_extra: List = [],
        **kwargs: Optional[Any]) -> Union[Client, SLURMCluster, LocalCluster]:
    """
    Start a Dask distributed SLURM worker cluster on the ESI HPC infrastructure
    (or local multi-processing)

    Parameters
    ----------
    partition : str
        Name of SLURM partition/queue to start workers in. Use the command `sinfo`
        in the terminal to see a list of available SLURM partitions on the ESI HPC
        cluster.
    n_workers : int
        Number of SLURM workers (=jobs) to spawn
    mem_per_worker : str
        Memory booking for each worker. Can be specified either in megabytes
        (e.g., ``mem_per_worker = 1500MB``) or gigabytes (e.g., ``mem_per_worker = "2GB"``).
        If `mem_per_worker` is `"auto"` it is attempted to infer a sane default value
        from the chosen partition, e.g., for ``partition = "8GBS"`` `mem_per_worker` is
        automatically set to the allowed maximum of `'8GB'`. On the IBM POWER
        partition "E880", `mem_per_worker` is set to 16 GB if not provided.
        Note, even in queues with guaranteed memory bookings, it is possible to allocate less
        memory than the allowed maximum per worker to spawn numerous low-memory
        workers. See Examples for details.
    cores_per_worker : None or int
        Number of CPU cores allocated for each worker. If `None`, core-count
        is set based on partition settings (`DefMemPerCPU`) with respect to
        CPU architecture (minimum 1 on x86_64, and 4 on IBM POWER).
    n_workers_startup : int
        Number of spawned workers to wait for. If `n_workers_startup` is `1` (default),
        the code does not proceed until either 1 SLURM job is running or the
        `timeout` interval has been exceeded.
    timeout : int
        Number of seconds to wait for requested workers to start (see `n_workers_startup`).
    interactive : bool
        If `True`, user input is queried in case not enough workers (set by `n_workers_startup`)
        could be started in the provided waiting period (determined by `timeout`).
        The code waits `interactive_wait` seconds for a user choice - if none is
        provided, it continues with the current number of running workers (if greater
        than zero). If `interactive` is `False` and no worker could not be started
        within `timeout` seconds, a `TimeoutError` is raised.
    interactive_wait : int
        Countdown interval (seconds) to wait for a user response in case fewer than
        `n_workers_startup` workers could be started. If no choice is provided within
        the given time, the code automatically proceeds with the current number of
        active dask workers.
    start_client : bool
        If `True`, a distributed computing client is launched and attached to
        the dask worker cluster. If `start_client` is `False`, only a distributed
        computing cluster is started to which compute-clients can connect.
    job_extra : list
        Extra sbatch parameters to pass to SLURMCluster.
    **kwargs : dict
        Additional keyword arguments can be used to control job-submission details.

    Returns
    -------
    proc : object
        A distributed computing client (if ``start_client = True``) or
        a distributed computing cluster (otherwise).

    Examples
    --------
    The following command launches 10 SLURM workers with 2 gigabytes memory each
    in the `8GBS` partition

    >>> client = esi_cluster_setup(n_workers=10, partition="8GBS", mem_per_worker="2GB")

    Use default settings to start 2 SLURM workers in the IBM POWER E880 partition
    (allocating 4 cores and 16 GB memory per worker)

    >>> client = esi_cluster_setup(partition="E880")

    The underlying distributed computing cluster can be accessed using

    >>> client.cluster

    Notes
    -----
    The employed parallel computing engine relies on the concurrent processing library
    `Dask <https://docs.dask.org/en/latest/>`_. Thus, the distributed computing
    clients generated here are in fact instances of :class:`distributed.Client`.
    This function specifically acts  as a wrapper for :class:`dask_jobqueue.SLURMCluster`.
    Users familiar with Dask in general and its distributed scheduler and cluster
    objects in particular, may leverage Dask's entire API to fine-tune parallel
    processing jobs to their liking (if wanted).

    See also
    --------
    dask_jobqueue.SLURMCluster : launch a dask cluster of SLURM workers
    slurm_cluster_setup : start a distributed Dask cluster of parallel processing workers using SLURM
    local_cluster_setup : start a local Dask multi-processing cluster on the host machine
    cluster_cleanup : remove dangling parallel processing worker-clusters
    """

    # For later reference: dynamically fetch name of current function
    funcName = f"<{inspect.currentframe().f_code.co_name}>"     # type: ignore

    # Backwards compatibility: legacy keywords are converted to new nomenclature
    if any(kw in kwargs for kw in __deprecated__):
        log.warning(__deprecation_wrng__)
        n_workers = kwargs.pop("n_jobs", n_workers)                             # type: ignore
        mem_per_worker = kwargs.pop("mem_per_job", mem_per_worker)
        n_workers_startup = kwargs.pop("n_jobs_startup", n_workers_startup)     # type: ignore
        log.debug("Set `n_workers = n_jobs`, `mem_per_worker = mem_per_job`\
                  and `n_workers_startup = n_jobs_startup`")

    # Don't start a new cluster on top of an existing one
    try:
        client = get_client()
        log.debug("Found existing client")
        if count_online_workers(client.cluster) == 0:
            log.debug("No active workers detected in %s", str(client))
            cluster_cleanup(client)
        else:
            log.info("Found existing parallel computing client %s. \
                     Not starting new cluster.", str(client))
            if start_client:
                return client
            return client.cluster
    except ValueError:
        log.debug("No existing clients detected")

    # Check if SLURM's `sinfo` can be accessed
    log.debug("Test if `sinfo` is available")
    proc = subprocess.Popen("sinfo",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    _, err = proc.communicate()

    # Any non-zero return-code means SLURM is not ready to use
    if proc.returncode != 0:

        # SLURM is not installed: either allocate `LocalCluster` or just leave
        if proc.returncode > 0:
            if interactive:
                msg = f"{funcName} SLURM does not seem to be installed on this machine " +\
                    f"({socket.gethostname()}). Do you want to start a local multi-processing " +\
                    "computing client instead? "
                startLocal = user_yesno(msg, default="no")
            else:
                startLocal = True
            if startLocal:
                return local_cluster_setup(interactive=interactive)

        # SLURM is installed, but something's wrong
        msg = "Cannot access SLURM queuing system from node %s: %s "
        log.error(msg, socket.gethostname(), err)
        raise IOError("%s %s"%(funcName, msg%(socket.gethostname(), err)))

    # Use default by-worker process count or extract it from anonymous keyword args (if provided)
    processes_per_worker = kwargs.pop("processes_per_worker", 1)
    log.debug("Found `sinfo`, set `processes_per_worker` to %d", processes_per_worker)

    # Get micro-architecture of submitting host
    mArch = platform.machine()

    # If partition is "auto" use `mem_per_worker` to pick pseudo-optimal partition
    # Note: the `np.where` gymnastic below is necessary since `argmin` refuses
    # to return multiple matches; if `mem_per_worker` is 12, then ``memDiff = [4, 4, ...]``,
    # however, 8GB won't fit a 12GB worker, so we have to pick the second match 16GB
    if isinstance(partition, str):
        if partition == "auto":
            if not isinstance(mem_per_worker, str) or mem_per_worker.find("estimate_memuse:") < 0:
                msg = "Cannot auto-select partition without first invoking memory estimation in `ParallelMap`. "
                log.error(msg)
                raise IOError("%s %s"%(funcName, msg))
            memEstimate = int(mem_per_worker.replace("estimate_memuse:", ""))
            mem_per_worker = "auto"
            log.info("Automatically selecting SLURM partition...")
            availPartitions = _get_slurm_partitions()
            if mArch == "x86_64":
                gbQueues = np.unique([int(queue.split("GB")[0]) for queue in availPartitions if queue[0].isdigit()])
                memDiff = np.abs(gbQueues - memEstimate)
                queueIdx = np.where(memDiff == memDiff.min())[0][-1]
                partition = f"{gbQueues[queueIdx]}GBXS"
            else:
                partition = "E880"
                mem_per_worker = f"{memEstimate} GB"
            msg = "Picked partition %s based on estimated memory consumption of %d GB"
            log.info(msg, partition, memEstimate)
        else:
            if (partition == "E880" and mArch == "x86_64") or \
               (mArch == "ppc64le" and partition != "E880"):
                otherArch = list(set(["x86_64", "ppc64le"]).difference([mArch]))[0]
                msg = "Cannot start SLURM workers in partition %s with " +\
                    "architecture %s from submitting host with architecture %s. " +\
                    "Start x86_64 workers from esi-svhpc{1,2,3} and POWER workers from the hub."
                raise ValueError(msg%(partition, otherArch, mArch))

    # Convert "auto" memory selection query to `None` for easier handling below`
    if isinstance(mem_per_worker, str) and mem_per_worker == "auto":
        mem_per_worker = None                                       # type: ignore
        log.debug("Using auto-memory selection")

    # If not explicitly provided, extract by-worker CPU core count from
    # partition via `DefMeMPerCPU` and `mem_per_worker`
    if cores_per_worker is None:
        try:
            log.debug("Using `scontrol` to get partition info")
            pc = subprocess.run(f"scontrol -o show partition {partition}",
                                capture_output=True, check=True, shell=True, text=True)
            defMem = int(pc.stdout.strip().partition("DefMemPerCPU=")[-1].split()[0])
            log.debug("Found DefMemPerCPU=%d", defMem)
        except Exception as exc:
            msg = "Cannot fetch available memory per CPU in SLURM: %s"
            log.error(msg, str(exc))
            raise IOError("%s %s"%(funcName, msg%(str(exc))))

        # Use ESI-specific x86_64 partition layout (8GB(S/X/L), 16GB(S/X/L), ... )
        # to infer memory and core count; use "sane" (fat quotes) defaults on IBM POWER
        # (if nothing was provided, use 4 cores/16 GB per worker)
        if mArch == "x86_64":
            memPerCore = 8000
        else:
            if mem_per_worker is not None:
                try:
                    memVal = float(mem_per_worker.replace("MB", "").replace("GB", ""))
                except:
                    raise ValueError("Invalid value of `mem_per_worker`: %s"%mem_per_worker)
                if memVal <= 0:
                    raise ValueError("Value of `mem_per_worker` has to be > 0!")
                if "MB" in mem_per_worker:
                    defMem = int(memVal)
                else:
                    defMem = int(round(memVal * 1000))
                log.debug("Found `mem_per_worker` set to %d MB", defMem)
            else:
                defMem = 16000                  # default to 4 cores per worker if no mem req was specified
                mem_per_worker = f"{defMem} MB"
                log.debug("No `mem_per_worker` specified, using default of %d MB", defMem)
            memPerCore = 4000

        # Set core-count per worker (applies to both x86_64 and ppc64le)
        cores_per_worker = max(1, int(defMem / memPerCore))
        log.debug("Derived core-count from partition: `cores_per_worker=%d`", cores_per_worker)

    # Determine if `job_extra`` is a list (this is also checked in `slurm_cluster_setup`,
    # but we may need to append to it, so ensure that's possible)
    if not isinstance(job_extra, list):
        msg = "`job_extra` has to be a list, not %s"
        log.error(msg, str(type(job_extra)))
        raise TypeError("%s %s"%(funcName, msg%(str(type(job_extra)))))

    # If '--output' was not provided, append default output folder to `job_extra`
    if not any(option.startswith("--output") or option.startswith("-o") for option in job_extra):
        log.debug("Auto-populating `--output` setting for sbatch")
        usr = getpass.getuser()
        slurm_wdir = f"/cs/slurm/{usr}/{usr}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(slurm_wdir, exist_ok=True)
        log.debug("Using %s for slurm logs", slurm_wdir)
        out_files = os.path.join(slurm_wdir, "slurm-%j.out")
        job_extra.append(f"--output={out_files}")
        log.debug("Setting `--output=%s`", out_files)

    # Let the SLURM-specific setup function do the rest (returns client or cluster)
    return slurm_cluster_setup(partition, cores_per_worker, n_workers,
                               processes_per_worker, mem_per_worker,
                               n_workers_startup, timeout, interactive,
                               interactive_wait, start_client, job_extra,
                               invalid_partitions=["DEV", "ESI"], **kwargs)


# Setup SLURM cluster
def slurm_cluster_setup(
        partition: str = "partition_name",
        n_cores: int = 1,
        n_workers: int = 1,
        processes_per_worker: int = 1,
        mem_per_worker: Optional[str] = "1GB",
        n_workers_startup: int = 1,
        timeout: int = 60,
        interactive: bool = True,
        interactive_wait: int = 10,
        start_client: bool = True,
        job_extra: List = [],
        invalid_partitions: List = [],
        **kwargs: Optional[Any]) -> Union[Client, SLURMCluster, None]:
    """
    Start a distributed Dask cluster of parallel processing workers using SLURM

    **NOTE** If you are working on the ESI HPC cluster, please use
    :func:`~acme.esi_cluster_setup` instead!

    Parameters
    ----------
    partition : str
        Name of SLURM partition/queue to use
    n_cores : int
        Number of CPU cores per SLURM worker
    n_workers : int
        Number of SLURM workers (=jobs) to spawn
    processes_per_worker : int
        Number of processes to use per SLURM job (=worker). Should be greater
        than one only if the chosen partition contains nodes that expose multiple
        cores per job.
    mem_per_worker : str or None
        Memory allocation for each worker. If `None`, partition's `DefMemPerCPU` is queried.
    n_workers_startup : int
        Number of spawned SLURM workers to wait for. The code does not return until either
        `n_workers_startup` SLURM jobs are running or the `timeout` interval (see
        below) has been exceeded.
    timeout : int
        Number of seconds to wait for requested workers to start (see `n_workers_startup`).
    interactive : bool
        If `True`, user input is queried in case not enough workers (set by `n_workers_startup`)
        could be started in the provided waiting period (determined by `timeout`).
        The code waits `interactive_wait` seconds for a user choice - if none is
        provided, it continues with the current number of running workers (if greater
        than zero). If `interactive` is `False` and no worker could not be started
        within `timeout` seconds, a `TimeoutError` is raised.
    interactive_wait : int
        Countdown interval (seconds) to wait for a user response in case fewer than
        `n_workers_startup` workers could be started. If no choice is provided within
        the given time, the code automatically proceeds with the current number of
        active dask workers.
    start_client : bool
        If `True`, a distributed computing client is launched and attached to
        the dask worker cluster. If `start_client` is `False`, only a distributed
        computing cluster is started to which compute-clients can connect.
    job_extra : list
        Extra sbatch parameters to pass to SLURMCluster.
    invalid_partition : list
        List of partition names (strings) that are not available for launching
        dask workers.

    Returns
    -------
    proc : object or None
        A distributed computing client (if ``start_client = True``) or
        a distributed computing cluster (otherwise). If no SLURM workers
        can be started within the given timeout interval, `proc` is set
        to `None`.

    See also
    --------
    dask_jobqueue.SLURMCluster : launch a dask cluster of SLURM workers
    esi_cluster_setup : start a SLURM worker cluster on the ESI HPC infrastructure
    local_cluster_setup : start a local Dask multi-processing cluster on the host machine
    cluster_cleanup : remove dangling parallel processing worker-clusters
    """

    # For later reference: dynamically fetch name of current function
    funcName = f"<{inspect.currentframe().f_code.co_name}>"     # type: ignore

    # Backwards compatibility: legacy keywords are converted to new nomenclature
    if any(kw in kwargs for kw in __deprecated__):
        log.warning(__deprecation_wrng__)
        n_workers = kwargs.pop("n_jobs", n_workers)                                 # type: ignore
        processes_per_worker = kwargs.pop("workers_per_job", processes_per_worker)  # type: ignore
        mem_per_worker = kwargs.pop("mem_per_job", mem_per_worker)                  # type: ignore
        n_workers_startup = kwargs.pop("n_jobs_startup", n_workers_startup)         # type: ignore
        log.debug("Set `n_workers = n_jobs`, `processes_per_worker = workers_per_job`, \
                  `mem_per_worker = mem_per_job` \
                  and `n_workers_startup = n_jobs_startup`")

    # Retrieve all partitions currently available in SLURM
    availPartitions = _get_slurm_partitions()

    # Make sure we're in a valid partition
    if partition not in availPartitions:
        valid = list(set(availPartitions).difference(invalid_partitions))
        lgl = "'" + "or '".join(opt + "' " for opt in valid)
        msg = "Invalid partition selection %s, available SLURM partitions are %s"
        log.error(msg, str(partition), lgl)
        raise ValueError("%s %s"%(funcName, msg%(str(partition), lgl)))
    log.debug("Found `partition = %s`", partition)

    # Parse worker count
    try:
        scalar_parser(n_workers, varname="n_workers", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        log.error("Error parsing `n_workers`")
        raise exc
    log.debug("Using `n_workers = %d`", n_workers)

    # Get requested memory per worker
    if isinstance(mem_per_worker, str):
        if mem_per_worker == "auto":
            mem_per_worker = None                                       # type: ignore
            log.debug("Using auto-memory selection")
    if mem_per_worker is not None:
        msg = "`mem_per_worker` has to be a valid memory specifier (e.g., '8GB', '12000MB'), not %s"
        if not isinstance(mem_per_worker, str):
            log.error(msg, str(type(mem_per_worker)))
            raise TypeError("%s %s"%(funcName, msg%(str(type(mem_per_worker)))))
        if not any(szstr in mem_per_worker for szstr in ["MB", "GB"]):
            log.error(msg, mem_per_worker)
            raise ValueError("%s %s"%(funcName, msg%(mem_per_worker)))
        memNumeric = mem_per_worker.replace("MB", "").replace("GB", "")
        log.debug("Found `mem_per_worker = %s` in input args", mem_per_worker)
        try:
            memVal = float(memNumeric)
        except:
            log.error(msg, mem_per_worker)
            raise ValueError("%s %s"%(funcName, msg%(mem_per_worker)))
        if memVal <= 0:
            log.error(msg, mem_per_worker)
            raise ValueError("%s %s"%(funcName, msg%(mem_per_worker)))

    # Parse worker-waiter count
    try:
        scalar_parser(n_workers_startup, varname="n_workers_startup", ntype="int_like", lims=[0, np.inf])
    except Exception as exc:
        log.error("Error parsing `n_workers_startup`")
        raise exc
    log.debug("Using `n_workers_startup = %d`", n_workers_startup)

    # Get memory limit (*in MB*) of chosen partition (guaranteed to exist, cf. above)
    log.debug("Use `scontrol` to fetch partition's memory limit")
    pc = subprocess.run(f"scontrol -o show partition {partition}",
                        capture_output=True, check=True, shell=True, text=True)
    try:
        mem_lim = int(pc.stdout.strip().partition("MaxMemPerCPU=")[-1].split()[0]) - 500
    except IndexError:
        mem_lim = np.inf                                            # type: ignore
    log.debug("Found a limit of  %s MB", str(mem_lim))

    # Consolidate requested memory with chosen partition (or assign default memory)
    if mem_per_worker is None:
        if np.isinf(mem_lim):
            try:
                mem_per_worker = f"{int(pc.stdout.strip().partition('DefMemPerCPU=')[-1].split()[0]) - 500}MB"
            except IndexError:
                raise ValueError("Cannot infer any default memory setting from partition %s"%partition)
        else:
            mem_per_worker = str(mem_lim) + "MB"
        log.debug("Using partition limit of %s MB", str(mem_lim))
    else:
        if "MB" in mem_per_worker:
            mem_req = int(memVal)
        else:
            mem_req = int(round(memVal * 1000))
        if mem_req > mem_lim:
            msg = "`mem_per_worker` exceeds limit of %d MB for partition %s. " +\
                "Capping memory at partition limit. "
            log.warning(msg, mem_lim, partition)
            mem_per_worker = str(int(mem_lim)) + "MB"

    # Parse requested timeout period
    try:
        scalar_parser(timeout, varname="timeout", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        log.error("Error parsing `timeout`")
        raise exc
    log.debug("Using `timeout = %d`", timeout)

    # Parse requested interactive waiting period
    try:
        scalar_parser(interactive_wait, varname="interactive_wait", ntype="int_like",
                      lims=[0, np.inf])
    except Exception as exc:
        log.error("Error parsing `interactive_wait`")
        raise exc
    log.debug("Using `interactive_wait = %d`", interactive_wait)

    # Determine if cluster allocation is happening interactively
    if not isinstance(interactive, bool):
        msg = "`interactive` has to be Boolean, not %s"
        log.error(msg, str(type(interactive)))
        raise TypeError("%s %s"%(funcName, msg%(str(type(interactive)))))
    log.debug("Using `interactive = %s`", str(interactive))

    # Determine if a dask client was requested
    if not isinstance(start_client, bool):
        msg = "`start_client` has to be Boolean, not %s"
        log.error(msg, str(type(start_client)))
        raise TypeError("%s %s"%(funcName, msg%(str(type(start_client)))))
    log.debug("Using `start_client = %s`", str(start_client))

    # Determine if job_extra is a list
    if not isinstance(job_extra, list):
        msg = "`job_extra` has to be List, not %s"
        log.error(msg, str(type(job_extra)))
        raise TypeError("%s %s"%(funcName, msg%(str(type(job_extra)))))

    # Determine if job_extra options are valid
    for option in job_extra:
        msg = "`job_extra` has to be a valid sbatch option, not %s"
        if not isinstance(option, str):
            log.error(msg, str(type(option)))
            raise TypeError("%s %s"%(funcName, msg%(str(type(option)))))
        if not option[0] == "-":
            msg = "`job_extra` options must be flagged with - or --"
            log.error(msg)
            raise ValueError("%s %s"%(funcName, msg))
    log.debug("Using `job_extra = %s`", str(job_extra))

    # Ensure validity of requested worker processes
    try:
        scalar_parser(processes_per_worker, varname="processes_per_worker",
                      ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        log.error("Error parsing `processes_per_worker`")
        raise exc
    log.debug("Using `processes_per_worker = %d`", processes_per_worker)

    # Check for sanity of requested core count
    try:
        scalar_parser(n_cores, varname="n_cores",
                      ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        log.error("Error parsing `n_cores`")
        raise exc
    log.debug("Using `n_cores = %d`", n_cores)

    # Check validity of '--output' option if provided
    userOutSpec = [option.startswith("--output") or option.startswith("-o") for option in job_extra]
    if any(userOutSpec):
        userOut = job_extra[userOutSpec.index(True)]
        outSpec = userOut.split("=")
        if len(outSpec) != 2:
            msg = "SLURM output directory must be specified using -o/--output=/path/to/file, not %s"
            log.error(msg, userOut)
            raise ValueError("%s %s"%(funcName, msg%(userOut)))
        slurm_wdir = os.path.split(outSpec[1])[0]
        if len(slurm_wdir) > 0 and not os.path.isdir(os.path.expanduser(slurm_wdir)):
            msg = "SLURM output location has to be an existing directory, not %s"
            log.error(msg, slurm_wdir)
            raise ValueError("%s %s"%(funcName, msg%(slurm_wdir)))
    else:
        slurm_wdir = None
    log.debug("Using `local_directory = %s`", slurm_wdir)

    # Create `SLURMCluster` object using provided parameters
    log.debug("Instantiating `SLURMCluster` object")
    cluster = SLURMCluster(cores=n_cores,
                           job_cpu=n_cores,
                           memory=mem_per_worker,
                           processes=processes_per_worker,
                           local_directory=slurm_wdir,
                           queue=partition,
                           python=sys.executable,
                           job_directives_skip=["-t 00:30:00"],
                           job_extra_directives=job_extra)

    # Compute total no. of workers and up-scale cluster accordingly
    if n_workers_startup < n_workers:
        msg = "Requested worker-count %d exceeds `n_workers_startup = %d`, " +\
            "waiting for %d workers to come online"
        log.debug(msg, n_workers, n_workers_startup, n_workers_startup)
    cluster.scale(n_workers)

    # Fire up waiting routine to avoid returning an undercooked cluster
    if _cluster_waiter(cluster, funcName, n_workers, timeout, interactive, interactive_wait):
        return None

    # Kill a zombie cluster in non-interactive mode
    if not interactive and count_online_workers(cluster) == 0:
        cluster_cleanup(Client(cluster))
        msg = "SLURM workers could not be started within given time-out " +\
              "interval of %d seconds"
        log.error(msg, timeout)
        raise TimeoutError("%s %s"%(funcName, msg%(timeout)))

    # Highlight how to connect to dask performance monitor
    msg = "Parallel computing client ready, dashboard accessible at %s"
    log.info(msg, cluster.dashboard_link)

    # If client was requested, return that instead of the created cluster
    if start_client:
        return Client(cluster)
    return cluster


def _get_slurm_partitions() -> List:
    """
    Local helper to fetch all partitions defined in SLURM
    """

    # For later reference: dynamically fetch name of current function
    funcName = f"<{inspect.currentframe().f_code.co_name}>"     # type: ignore

    # Retrieve all partitions currently available in SLURM
    log.debug("Use `sinfo` to fetch available partitions")
    proc = subprocess.Popen("sinfo -h -o %P",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    out, err = proc.communicate()

    # Any non-zero return-code means SLURM is not ready to use
    if proc.returncode != 0:
        msg = "Error fetching SLURM partition setup from node %s: %s"
        log.error(msg, socket.gethostname(), err)
        raise IOError("%s %s"%(funcName, msg%(socket.gethostname(), err)))

    # Return formatted subprocess shell output
    log.debug("Found partitions: %s", out)
    return out.split()


def _cluster_waiter(
        cluster: SLURMCluster,
        funcName: str,
        total_workers: int,
        timeout: int,
        interactive: bool,
        interactive_wait: int) -> bool:
    """
    Local helper that can be called recursively
    """

    # Wait until all workers have been started successfully or we run out of time
    wrkrs = count_online_workers(cluster)
    to = str(timedelta(seconds=timeout))[2:]
    fmt = "{desc}: {n}/{total} \t[elapsed time {elapsed} | timeout at " + to + "]"
    ani = tqdm(desc=f"{funcName} SLURM workers ready", total=total_workers,
               leave=True, bar_format=fmt, initial=wrkrs, position=0)
    counter = 0
    while count_online_workers(cluster) < total_workers and counter < timeout:
        time.sleep(1)
        counter += 1
        ani.update(max(0, count_online_workers(cluster) - wrkrs))
        wrkrs = count_online_workers(cluster)
        ani.refresh()   # force refresh to display elapsed time every second
    ani.close()

    # If we ran out of time before all workers could be started, ask what to do
    if counter == timeout and interactive:                              # pragma: no cover
        msg = "SLURM workers could not be started within given time-out " +\
              "interval of %d seconds"
        log.info(msg, timeout)
        query = f"{funcName} Do you want to [k]eep waiting for 60s, [a]bort or " +\
                f"[c]ontinue with {wrkrs} workers?"
        choice = user_input(query, valid=["k", "a", "c"], default="c", timeout=interactive_wait)
        if choice == "k":
            return _cluster_waiter(cluster, funcName, total_workers, 60, True, 60)
        elif choice == "a":
            log.info("Closing cluster...")
            cluster_cleanup(Client(cluster))
            return True
        else:
            if wrkrs == 0:
                query = f"{funcName} Cannot continue with 0 workers. Do you want to " +\
                        "[k]eep waiting for 60s or [a]bort?"
                choice = user_input(query, valid=["k", "a"],
                                    default="a", timeout=60)
                if choice == "k":
                    _cluster_waiter(cluster, funcName, total_workers, 60, True, 60)
                else:
                    log.info("Closing cluster...")
                    cluster_cleanup(Client(cluster))
                    return True

    return False


def local_cluster_setup(
        n_workers: Optional[int] = None,
        mem_per_worker: Optional[str] = None,
        interactive: bool = True) -> Union[Client, None]:
    """
    Start a local distributed Dask multi-processing cluster

    Parameters
    ----------
    n_workers : int
        Number of local workers to start (this should align with the locally
        available hardware, see :class:`distributed.LocalCluster` for details)
    mem_per_worker : str
        Memory cap for each local worker (corresponds to the `memory_limit`
        keyword of a :class:`distributed.worker.Worker`)
    interactive : bool
        If `True`, a confirmation dialog is displayed to ensure proper encapsulation
        of calls to `local_cluster_setup` inside a script's main module block.
        See Notes for details. If `interactive` is `False`, the dialog is not shown.

    Returns
    -------
    client : distributed.Client or None
        A distributed computing client. If a client cannot be started,
        `proc` is set to `None`.

    Notes
    -----
    The way Python spawns new processes requires an explicit separation of initialization
    code (i.e., code blocks that should only be executed once) from the actual
    program code. Specifically, everything that is supposed to be invoked only
    once by the parent spawner must be encapsulated in a script's main block.
    Otherwise, initialization code is not only run once by the parent process but
    executed by every child process at import time.

    This means that starting a local multi-processing cluster *has* to be wrapped
    inside a script's main module block, otherwise, every child process created by
    the multi-processing cluster starts a multi-processing cluster itself and so
    on escalating to an infinite recursion. Thus, if `local_cluster_setup` is
    called inside a script, it has to be encapsulated in the script's main module
    block, i.e.,

    .. code-block:: python

        if __name__ == "__main__":
            ...
            local_cluster_setup()
            ...

    Note that this capsulation is **only** required inside Python scripts. Launching
    `local_cluster_setup` in the (i)Python shell or inside a Jupyter notebook
    does not suffer from this problem.
    A more in-depth technical discussion of this limitation can be found in
    `Dask's GitHub issue tracker <https://github.com/dask/distributed/issues/2520>`_.

    Examples
    --------
    The following command launches a local distributed computing cluster using
    all CPU cores available on the host machine

    >>> client = local_cluster_setup()

    The underlying distributed computing cluster can be accessed using

    >>> client.cluster

    See also
    --------
    distributed.LocalCluster : create local worker cluster
    esi_cluster_setup : Start a distributed Dask cluster using SLURM
    cluster_cleanup : remove dangling parallel processing worker-clusters
    """

    # For later reference: dynamically fetch name of current function
    funcName = f"<{inspect.currentframe().f_code.co_name}>"     # type: ignore

    # Determine if cluster allocation is happening interactively
    if not isinstance(interactive, bool):
        msg = "`interactive` has to be Boolean, not %s"
        log.error(msg, str(type(interactive)))
        raise TypeError("%s %s"%(funcName, msg%(str(type(interactive)))))
    log.debug("Using `interactive = %s`", str(interactive))

    if not is_jupyter():
        msg = """\
        If you use a script to start a local parallel computing client, please ensure
        the call to `local_cluster_setup` is wrapped inside a main module block, i.e.,

            if __name__ == "__main__":
                ...
                local_cluster_setup()
                ...

        Otherwise, a RuntimeError is raised due to an infinite recursion triggered by
        new processes being started before the calling process can finish its bootstrapping
        phase.
        """
        msg = textwrap.dedent(msg)
        log.debug(msg)

    # Additional safe-guard: if a script is executed, double-check with the user
    # for proper main idiom usage
    if interactive:                                                     # pragma: no cover
        msg = f"{funcName} If launched from a script, did you wrap your code " +\
            "inside a __main__ module block?"
        if not user_yesno(msg, default="no"):
            return None

    # Start the actual distributed client
    if n_workers is not None or mem_per_worker is not None:
        msg = "Starting `LocalCluster` with `n_workers = %s` and `memory_limit = %s`"
        log.debug(msg, str(n_workers), str(mem_per_worker))
        cluster = LocalCluster(n_workers=n_workers, memory_limit=mem_per_worker)
        client = Client(cluster)
    else:
        client = Client()
    msg = "Local parallel computing client ready, dashboard accessible at %s"
    log.info(msg, client.cluster.dashboard_link)
    return client


def cluster_cleanup(client: Optional[Client] = None) -> None:
    """
    Stop and close dangling parallel processing workers

    Parameters
    ----------
    client : dask distributed computing client or None
        Either a  :class:`distributed.Client` or `None`. If `None`, a
        global client is queried for and shut-down if found (without confirmation!).

    Returns
    -------
    Nothing : None

    See also
    --------
    esi_cluster_setup : Launch SLURM workers on the ESI compute cluster
    slurm_cluster_setup : start a distributed Dask cluster of parallel processing workers using SLURM
    local_cluster_setup : start a local Dask multi-processing cluster on the host machine
    """

    # For later reference: dynamically fetch name of current function
    funcName = f"<{inspect.currentframe().f_code.co_name}>"     # type: ignore

    # Attempt to establish connection to dask client
    if client is None:
        try:
            client = get_client()
        except ValueError:
            log.warning("No dangling clients or clusters found.")
            return
        except Exception as exc:                                        # pragma: no cover
            log.error("Error looking for dask client")
            raise exc
    else:
        if not isinstance(client, Client):
            msg = "`client` has to be a dask client object, not %s"
            log.error(msg, str(type(client)))
            raise TypeError("%s %s"%(funcName, msg%(str(type(client)))))
    log.debug("Found client %s", str(client))

    # Prepare message for prompt
    if client.cluster.__class__.__name__ == "LocalCluster":
        userClust = f"LocalCluster hosted on {client.scheduler_info()['address']}"
    else:
        userName = getpass.getuser()
        outDir = client.cluster.job_header.partition("--output=")[-1]
        jobID = outDir.partition(f"{userName}_")[-1].split(os.sep)[0]
        userClust = f"cluster {userName}_{jobID}"
    nWorkers = count_online_workers(client.cluster)

    # First gracefully shut down all workers, then close client
    client.retire_workers(list(client.scheduler_info()['workers']), close_workers=True)
    client.close()
    try:
        client.cluster.close()
    except Exception as exc:
        log.warning("Could not gracefully shut down cluster: %s", str(exc))

    # Communicate what just happened and get outta here
    msg = "Successfully shut down %s containing %d workers"
    log.info(msg, userClust, nWorkers)

    return


def count_online_workers(cluster: SLURMCluster) -> int:
    """
    Local replacement for the late `._count_active_workers` class method
    """
    return len([w["memory_limit"] for w in cluster.scheduler_info["workers"].values()])
