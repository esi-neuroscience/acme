# -*- coding: utf-8 -*-
#
# Helper routines for working w/dask
#

# Builtin/3rd party package imports
import os
import sys
import socket
import subprocess
import getpass
import time
import inspect
import textwrap
import numpy as np
from tqdm import tqdm
if sys.platform == "win32":
    # tqdm breaks term colors on Windows - fix that (tqdm issue #446)
    import colorama
    colorama.deinit()
    colorama.init(strip=False)
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, get_client, LocalCluster
from datetime import datetime, timedelta

# Local imports
from acme import __deprecated__, __deprecation_wrng__
from .shared import user_input, user_yesno
from .spy_interface import scalar_parser, log

__all__ = ["esi_cluster_setup", "local_cluster_setup", "cluster_cleanup", "slurm_cluster_setup"]


# Setup SLURM workers on the ESI HPC cluster
def esi_cluster_setup(partition="8GBXS",
                      n_workers=2,
                      mem_per_worker="auto",
                      n_workers_startup=1,
                      timeout=60,
                      interactive=True,
                      interactive_wait=120,
                      start_client=True,
                      job_extra=[],
                      **kwargs):
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
    mem_per_worker : None or str
        Memory booking for each worker. Can be specified either in megabytes
        (e.g., ``mem_per_worker = 1500MB``) or gigabytes (e.g., ``mem_per_worker = "2GB"``).
        If `mem_per_worker` is `None`, or `"auto"` it is attempted to infer a sane default value
        from the chosen partition, e.g., for ``partition = "8GBS"`` `mem_per_worker` is
        automatically set to the allowed maximum of `'8GB'`. However, even in
        queues with guaranteed memory bookings, it is possible to allocate less
        memory than the allowed maximum per worker to spawn numerous low-memory
        workers. See Examples for details.
    n_workers_startup : int
        Number of spawned workers to wait for. If `n_workers_startup` is `100` (default),
        the code does not proceed until either 100 SLURM jobs are running or the
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

    The underlying distributed computing cluster can be accessed using

    >>> client.cluster

    Notes
    -----
    The employed parallel computing engine relies on the concurrent processing library
    `Dask <https://docs.dask.org/en/latest/>`_. Thus, the distributed computing
    clients generated here are in fact instances of :class:`dask.distributed.Client`.
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
    funcName = "<{}>".format(inspect.currentframe().f_code.co_name)

    # Backwards compatibility: legacy keywords are converted to new nomenclature
    if any(kw in kwargs for kw in __deprecated__):
        log.warning("%s %s", funcName, __deprecation_wrng__)
        n_workers = kwargs.pop("n_jobs", n_workers)
        mem_per_worker = kwargs.pop("mem_per_job", mem_per_worker)
        n_workers_startup = kwargs.pop("n_jobs_startup", n_workers_startup)
        log.debug("%s Set `n_workers = n_jobs`, `mem_per_worker = mem_per_job`\
                  and `n_workers_startup = n_jobs_startup`", funcName)

    # Don't start a new cluster on top of an existing one
    try:
        client = get_client()
        log.debug("%s Found existing client", funcName)
        if count_online_workers(client.cluster) == 0:
            log.debug("%s No active workers detected in %s", funcName, str(client))
            cluster_cleanup(client)
        else:
            log.info("%s Found existing parallel computing client %s. \
                     Not starting new cluster.", funcName, str(client))
            if start_client:
                return client
            return client.cluster
    except ValueError:
        log.debug("%s No existing clients detected", funcName)

    # Check if SLURM's `sinfo` can be accessed
    log.debug("%s Test if `sinfo` is available", funcName)
    proc = subprocess.Popen("sinfo",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    _, err = proc.communicate()

    # Any non-zero return-code means SLURM is not ready to use
    if proc.returncode != 0:

        # SLURM is not installed: either allocate `LocalCluster` or just leave
        if proc.returncode > 0:
            if interactive:
                msg = "{name:s} SLURM does not seem to be installed on this machine " +\
                    "({host:s}). Do you want to start a local multi-processing " +\
                    "computing client instead? "
                startLocal = user_yesno(msg.format(name=funcName, host=socket.gethostname()),
                                        default="no")
            else:
                startLocal = True
            if startLocal:
                return local_cluster_setup(interactive=interactive, start_client=start_client)

        # SLURM is installed, but something's wrong
        msg = "%s Cannot access SLURM queuing system from node %s: %s "
        log.error(msg, funcName, socket.gethostname(), err)
        raise IOError(msg%(funcName, socket.gethostname(), err))

    # Use default by-worker process count or extract it from anonymous keyword args (if provided)
    processes_per_worker = kwargs.pop("processes_per_worker", 1)
    log.debug("%s Found `sinfo`, set `processes_per_worker` to %d", funcName, processes_per_worker)

    # If partition is "auto" use `mem_per_worker` to pick pseudo-optimal partition
    # Note: the `np.where` gymnastic below is necessary since `argmin` refuses
    # to return multiple matches; if `mem_per_worker` is 12, then ``memDiff = [4, 4, ...]``,
    # however, 8GB won't fit a 12GB worker, so we have to pick the second match 16GB
    if isinstance(partition, str) and partition == "auto":
        if not isinstance(mem_per_worker, str) or mem_per_worker.find("estimate_memuse:") < 0:
            msg = "%s cannot auto-select partition without first invoking memory estimation in `ParallelMap`. "
            log.error(msg, funcName)
            raise IOError(msg%(funcName))
        memEstimate = int(mem_per_worker.replace("estimate_memuse:" ,""))
        mem_per_worker = "auto"
        log.info("%s Automatically selecting SLURM partition...", funcName)
        availPartitions = _get_slurm_partitions(funcName)
        gbQueues = np.unique([int(queue.split("GB")[0]) for queue in availPartitions if queue[0].isdigit()])
        memDiff = np.abs(gbQueues - memEstimate)
        queueIdx = np.where(memDiff == memDiff.min())[0][-1]
        partition = "{}GBXS".format(gbQueues[queueIdx])
        msg = "%s Picked partition %s based on estimated memory consumption of %d GB"
        log.info(msg, funcName, partition, memEstimate)

    # Extract by-worker CPU core count from anonymous keyword args or...
    if kwargs.get("n_cores") is not None:
        n_cores = kwargs.pop("n_cores")
        log.debug("%s Set `n_cores = %d` from kwargs", funcName, n_cores)
    else:
        # ...get memory limit (*in MB*) of chosen partition and set core count
        # accordingly (multiple of 8 wrt to GB RAM)
        try:
            log.debug("%s Using `scontrol` to get partition info", funcName)
            pc = subprocess.run("scontrol -o show partition {}".format(partition),
                                capture_output=True, check=True, shell=True, text=True)
            defMem = int(pc.stdout.strip().partition("DefMemPerCPU=")[-1].split()[0])
            log.debug("%s Found DefMemPerCPU=%d", funcName, defMem)
        except Exception as exc:
            msg = "%s Cannot fetch available memory per CPU in SLURM: %s"
            log.error(msg, funcName, str(exc))
            raise IOError(msg%(funcName, str(exc)))
        n_cores = int(defMem / 8000)
        log.debug("%s Using `n_cores=%d`", funcName, n_cores)

    # Determine if `job_extra`` is a list (this is also checked in `slurm_cluster_setup`,
    # but we may need to append to it, so ensure that's possible)
    if not isinstance(job_extra, list):
        msg = "%s `job_extra` has to be a list, not %s"
        log.error(msg, funcName, str(type(job_extra)))
        raise TypeError(msg%(funcName, str(type(job_extra))))

    # If '--output' was not provided, append default output folder to `job_extra`
    if not any(option.startswith("--output") or option.startswith("-o") for option in job_extra):
        log.debug("%s Auto-populating `--output` setting for sbatch", funcName)
        usr = getpass.getuser()
        slurm_wdir = "/cs/slurm/{usr:s}/{usr:s}_{date:s}"
        slurm_wdir = slurm_wdir.format(usr=usr,
                                       date=datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(slurm_wdir, exist_ok=True)
        log.debug("%s Using %s for slurm logs", funcName, slurm_wdir)
        out_files = os.path.join(slurm_wdir, "slurm-%j.out")
        job_extra.append("--output={}".format(out_files))
        log.debug("%s Setting `--output=%s`", funcName, out_files)

    # Let the SLURM-specific setup function do the rest (returns client or cluster)
    log.debug("%s Calling `slurm_cluster_setup`", funcName)
    return slurm_cluster_setup(partition, n_cores, n_workers, processes_per_worker, mem_per_worker,
                               n_workers_startup, timeout, interactive, interactive_wait,
                               start_client, job_extra, invalid_partitions=["DEV", "PPC"], **kwargs)


# Setup SLURM cluster
def slurm_cluster_setup(partition="partition_name",
                        n_cores=1,
                        n_workers=1,
                        processes_per_worker=1,
                        mem_per_worker="1GB",
                        n_workers_startup=1,
                        timeout=60,
                        interactive=True,
                        interactive_wait=10,
                        start_client=True,
                        job_extra=[],
                        invalid_partitions=[],
                        **kwargs):
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
    mem_per_worker : str
        Memory allocation for each worker
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
    proc : object
        A distributed computing client (if ``start_client = True``) or
        a distributed computing cluster (otherwise).

    See also
    --------
    dask_jobqueue.SLURMCluster : launch a dask cluster of SLURM workers
    esi_cluster_setup : start a SLURM worker cluster on the ESI HPC infrastructure
    local_cluster_setup : start a local Dask multi-processing cluster on the host machine
    cluster_cleanup : remove dangling parallel processing worker-clusters
    """

    # For later reference: dynamically fetch name of current function
    funcName = "<{}>".format(inspect.currentframe().f_code.co_name)

    # Backwards compatibility: legacy keywords are converted to new nomenclature
    if any(kw in kwargs for kw in __deprecated__):
        log.warning("%s %s", funcName, __deprecation_wrng__)
        n_workers = kwargs.pop("n_jobs", n_workers)
        processes_per_worker = kwargs.pop("workers_per_job", processes_per_worker)
        mem_per_worker = kwargs.pop("mem_per_job", mem_per_worker)
        n_workers_startup = kwargs.pop("n_jobs_startup", n_workers_startup)
        log.debug("%s Set `n_workers = n_jobs`, `processes_per_worker = workers_per_job`, \
                  `mem_per_worker = mem_per_job` \
                  and `n_workers_startup = n_jobs_startup`", funcName)

    # Retrieve all partitions currently available in SLURM
    availPartitions = _get_slurm_partitions(funcName)

    # Make sure we're in a valid partition
    if partition not in availPartitions:
        valid = list(set(availPartitions).difference(invalid_partitions))
        lgl = "'" + "or '".join(opt + "' " for opt in valid)
        msg = "%s Invalid partition selection %s, available SLURM partitions are %s"
        log.error(msg, funcName, str(partition), lgl)
        raise ValueError(msg%(funcName, str(partition), lgl))
    log.debug("%s Found `partition = %s`", funcName, partition)

    # Parse worker count
    try:
        scalar_parser(n_workers, varname="n_workers", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        log.error("%s Error parsing `n_workers`", funcName)
        raise exc
    log.debug("%s Using `n_workers = %d`", funcName, n_workers)

    # Get requested memory per worker
    if isinstance(mem_per_worker, str):
        if mem_per_worker == "auto":
            mem_per_worker = None
            log.debug("%s Using auto-memory selection", funcName)
    if mem_per_worker is not None:
        msg = "%s `mem_per_worker` has to be a valid memory specifier (e.g., '8GB', '12000MB'), not %s"
        if not isinstance(mem_per_worker, str):
            log.error(msg, funcName, str(type(mem_per_worker)))
            raise TypeError(msg%(funcName, str(type(mem_per_worker))))
        if not any(szstr in mem_per_worker for szstr in ["MB", "GB"]):
            log.error(msg, funcName, mem_per_worker)
            raise ValueError(msg%(funcName, mem_per_worker))
        memNumeric = mem_per_worker.replace("MB","").replace("GB","")
        log.debug("%s Found `mem_per_worker = %s` in input args", funcName, mem_per_worker)
        try:
            memVal = float(memNumeric)
        except:
            log.error(msg, funcName, mem_per_worker)
            raise ValueError(msg%(funcName, mem_per_worker))
        if memVal <= 0:
            log.error(msg, funcName, mem_per_worker)
            raise ValueError(msg%(funcName, mem_per_worker))

    # Parse worker-waiter count
    try:
        scalar_parser(n_workers_startup, varname="n_workers_startup", ntype="int_like", lims=[0, np.inf])
    except Exception as exc:
        log.error("%s Error parsing `n_workers_startup`", funcName)
        raise exc
    log.debug("%s Using `n_workers_startup = %d`", funcName, n_workers_startup)

    # Get memory limit (*in MB*) of chosen partition (guaranteed to exist, cf. above)
    log.debug("%s Use `scontrol` to fetch partition's memory limit")
    pc = subprocess.run("scontrol -o show partition {}".format(partition),
                        capture_output=True, check=True, shell=True, text=True)
    try:
        mem_lim = int(pc.stdout.strip().partition("MaxMemPerCPU=")[-1].split()[0])
    except IndexError:
        try:
            mem_lim = int(pc.stdout.strip().partition("DefMemPerCPU=")[-1].split()[0])
        except IndexError:
            mem_lim = np.inf
    log.debug("%s Found a limit of  %s MB", funcName, str(mem_lim))

    # Consolidate requested memory with chosen partition (or assign default memory)
    if mem_per_worker is None:
        mem_per_worker = str(mem_lim) + "MB"
        log.debug("%s Using partition limit of %s MB", funcName, str(mem_lim))
    else:
        if "MB" in mem_per_worker:
            mem_req = int(mem_per_worker[:mem_per_worker.find("MB")])
        else:
            mem_req = int(round(float(mem_per_worker[:mem_per_worker.find("GB")]) * 1000))
        if mem_req > mem_lim:
            msg = "%s `mem_per_worker` exceeds limit of %d MB for partition %s. " +\
                "Capping memory at partition limit. "
            log.warning(msg, funcName, mem_lim, partition)
            mem_per_worker = str(int(mem_lim)) + "MB"

    # Parse requested timeout period
    try:
        scalar_parser(timeout, varname="timeout", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        log.error("%s Error parsing `timeout`", funcName)
        raise exc
    log.debug("%s Using `timeout = %d`", funcName, timeout)

    # Parse requested interactive waiting period
    try:
        scalar_parser(interactive_wait, varname="interactive_wait", ntype="int_like",
                      lims=[0, np.inf])
    except Exception as exc:
        log.error("%s Error parsing `interactive_wait`", funcName)
        raise exc
    log.debug("%s Using `interactive_wait = %d`", funcName, interactive_wait)

    # Determine if cluster allocation is happening interactively
    if not isinstance(interactive, bool):
        msg = "%s `interactive` has to be Boolean, not %s"
        log.error(msg, funcName, str(type(interactive)))
        raise TypeError(msg%(funcName, str(type(interactive))))
    log.debug("%s Using `interactive = %s`", funcName, str(interactive))

    # Determine if a dask client was requested
    if not isinstance(start_client, bool):
        msg = "%s `start_client` has to be Boolean, not %s"
        log.error(msg, funcName, str(type(start_client)))
        raise TypeError(msg%(funcName, str(type(start_client))))
    log.debug("%s Using `start_client = %s`", funcName, str(start_client))

    # Determine if job_extra is a list
    if not isinstance(job_extra, list):
        msg = "%s `job_extra` has to be List, not %s"
        log.error(msg, funcName, str(type(job_extra)))
        raise TypeError(msg%(funcName, str(type(job_extra))))

    # Determine if job_extra options are valid
    for option in job_extra:
        msg = "%s `job_extra` has to be a valid sbatch option, not %s"
        if not isinstance(option, str):
            log.error(msg, funcName, str(type(option)))
            raise TypeError(msg%(funcName, str(type(option))))
        if not option[0] == "-":
            msg = "%s `job_extra` options must be flagged with - or --"
            log.error(msg, funcName)
            raise ValueError(msg%(funcName))
    log.debug("%s Using `job_extra = %s`", funcName, str(job_extra))

    # Ensure validity of requested worker processes
    try:
        scalar_parser(processes_per_worker, varname="processes_per_worker",
                      ntype="int_like", lims=[1, 16])
    except Exception as exc:
        log.error("%s Error parsing `processes_per_worker`", funcName)
        raise exc
    log.debug("%s Using `processes_per_worker = %d`", funcName, processes_per_worker)

    # Check for sanity of requested core count
    try:
        scalar_parser(n_cores, varname="n_cores",
                      ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        log.error("%s Error parsing `n_cores`", funcName)
        raise exc
    log.debug("%s Using `n_cores = %d`", funcName, n_cores)

    # Check validity of '--output' option if provided
    userOutSpec = [option.startswith("--output") or option.startswith("-o") for option in job_extra]
    if any(userOutSpec):
        userOut = job_extra[userOutSpec.index(True)]
        outSpec = userOut.split("=")
        if len(outSpec) != 2:
            msg = "%s the SLURM output directory must be specified using -o/--output=/path/to/file, not %s"
            log.error(msg, funcName, userOut)
            raise ValueError(msg%(funcName, userOut))
        slurm_wdir = os.path.split(outSpec[1])[0]
        if len(slurm_wdir) > 0 and not os.path.isdir(os.path.expanduser(slurm_wdir)):
            msg = "%s SLURM output location has to be an existing directory, not %s"
            log.error(msg, funcName, slurm_wdir)
            raise ValueError(msg%(funcName, slurm_wdir))
    else:
        slurm_wdir = None
    log.debug("%s Using `local_directory = %s`", funcName, slurm_wdir)

    # Create `SLURMCluster` object using provided parameters
    log.debug("%s Instantiating `SLURMCluster` object", funcName)
    cluster = SLURMCluster(cores=n_cores,
                           memory=mem_per_worker,
                           processes=processes_per_worker,
                           local_directory=slurm_wdir,
                           queue=partition,
                           python=sys.executable,
                           job_directives_skip=["-t", "--mem"],
                           job_extra_directives=job_extra)
                           # interface="asdf", # interface is set via `psutil.net_if_addrs()`

    # Compute total no. of workers and up-scale cluster accordingly
    if n_workers_startup < n_workers:
        msg = "%s Requested worker-count %d exceeds `n_workers_startup = %d`, " +\
            "waiting for %d workers to come online"
        log.debug(msg, funcName, n_workers, n_workers_startup, n_workers_startup)
    cluster.scale(n_workers)

    # Fire up waiting routine to avoid returning an undercooked cluster
    if _cluster_waiter(cluster, funcName, n_workers, timeout, interactive, interactive_wait):
        return

    # Kill a zombie cluster in non-interactive mode
    if not interactive and count_online_workers(cluster) == 0:
        cluster.close()
        msg = "%s SLURM workers could not be started within given time-out " +\
              "interval of %d seconds"
        log.error(msg, funcName, timeout)
        raise TimeoutError(msg%(funcName, timeout))

    # Highlight how to connect to dask performance monitor
    msg = "%s Parallel computing client ready, dashboard accessible at %s"
    log.info(msg, funcName, cluster.dashboard_link)

    # If client was requested, return that instead of the created cluster
    if start_client:
        return Client(cluster)
    return cluster


def _get_slurm_partitions(funcName):
    """
    Local helper to fetch all partitions defined in SLURM
    """

    # For later reference: dynamically fetch name of current function
    funcName = "<{}>".format(inspect.currentframe().f_code.co_name)

    # Retrieve all partitions currently available in SLURM
    log.debug("%s Use `sinfo` to fetch available partitions", funcName)
    proc = subprocess.Popen("sinfo -h -o %P",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    out, err = proc.communicate()

    # Any non-zero return-code means SLURM is not ready to use
    if proc.returncode != 0:
        msg = "%s Error fetching SLURM partition setup from node %s: %s"
        log.error(msg, funcName, socket.gethostname(), err)
        raise IOError(msg%(funcName, socket.gethostname(), err))

    # Return formatted subprocess shell output
    log.debug("%s Found partitions: %s", funcName, out)
    return out.split()


def _cluster_waiter(cluster, funcName, total_workers, timeout, interactive, interactive_wait):
    """
    Local helper that can be called recursively
    """

    # Wait until all workers have been started successfully or we run out of time
    wrkrs = count_online_workers(cluster)
    to = str(timedelta(seconds=timeout))[2:]
    fmt = "{desc}: {n}/{total} \t[elapsed time {elapsed} | timeout at " + to + "]"
    ani = tqdm(desc="{} SLURM workers ready".format(funcName), total=total_workers,
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
    if counter == timeout and interactive:
        msg = "%s SLURM workers could not be started within given time-out " +\
              "interval of %d seconds"
        log.info(msg, funcName, timeout)
        query = "{name:s} Do you want to [k]eep waiting for 60s, [a]bort or " +\
                "[c]ontinue with {wrk:d} workers?"
        choice = user_input(query.format(name=funcName, wrk=wrkrs),
                            valid=["k", "a", "c"], default="c", timeout=interactive_wait)

        if choice == "k":
            return _cluster_waiter(cluster, funcName, total_workers, 60, True, 60)
        elif choice == "a":
            log.info("%s Closing cluster...", funcName)
            cluster.close()
            return True
        else:
            if wrkrs == 0:
                query = "{} Cannot continue with 0 workers. Do you want to " +\
                        "[k]eep waiting for 60s or [a]bort?"
                choice = user_input(query.format(funcName), valid=["k", "a"],
                                    default="a", timeout=60)
                if choice == "k":
                    _cluster_waiter(cluster, funcName, total_workers, 60, True, 60)
                else:
                    log.info("%s Closing cluster...", funcName)
                    cluster.close()
                    return True

    return False


def local_cluster_setup(n_workers=None,
                        mem_per_worker=None,
                        interactive=True,
                        start_client=True):
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
    start_client : bool
        If `True`, a distributed computing client is launched and attached to
        the workers. If `start_client` is `False`, only a distributed
        computing cluster is started to which compute-clients can connect.

    Returns
    -------
    proc : object
        A distributed computing client (if ``start_client = True``) or
        a distributed computing cluster (otherwise).

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
    funcName = "<{}>".format(inspect.currentframe().f_code.co_name)

    # Determine if cluster allocation is happening interactively
    if not isinstance(interactive, bool):
        msg = "%s `interactive` has to be Boolean, not %s"
        log.error(msg, funcName, str(type(interactive)))
        raise TypeError(msg%(funcName, str(type(interactive))))
    log.debug("%s Using `interactive = %s`", funcName, str(interactive))

    # Determine if a dask client was requested
    if not isinstance(start_client, bool):
        msg = "%s `start_client` has to be Boolean, not $s"
        log.error(msg, funcName, str(type(start_client)))
        raise TypeError(msg%(funcName, str(type(start_client))))
    log.debug("%s Using `start_client = %s`", funcName, str(start_client))

    # Check, if we're running inside a Jupyter notebook...
    try:
        ipy = get_ipython()
        if ipy.__class__.__name__ == "ZMQInteractiveShell":
            maybeScript = False # Jupyter Notebook
            log.debug("%s Running in a Jupyter Notebook", funcName)
        else:
            maybeScript = True  # iPython shell
            log.debug("%s Running in an iPython shell", funcName)
    except NameError:
        maybeScript = True      # Python shell
        log.debug("%s Running in a standard Python shell", funcName)

    # ...if not, print warning/info message
    if maybeScript:
        msg = """\
        %s If you use a script to start a local parallel computing client, please ensure
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
        log.debug(msg, funcName)

    # Additional safe-guard: if a script is executed, double-check with the user
    # for proper main idiom usage
    if interactive:
        msg = "{name:s} If launched from a script, did you wrap your code " +\
            "inside a __main__ module block?"
        if not user_yesno(msg.format(name=funcName), default="no"):
            return

    # Start the actual distributed client
    if n_workers is not None or mem_per_worker is not None:
        msg = "%s Starting `LocalCluster` with `n_workers = %s` and `memory_limit = %s`"
        log.debug(msg, funcName, str(n_workers), str(mem_per_worker))
        cluster = LocalCluster(n_workers=n_workers, memory_limit=mem_per_worker)
        client = Client(cluster)
    else:
        client = Client()
    msg = "%s Local parallel computing client ready, dashboard accessible at %s"
    log.info(msg, funcName, client.cluster.dashboard_link)
    if start_client:
        return client
    return client.cluster


def cluster_cleanup(client=None):
    """
    Stop and close dangling parallel processing workers

    Parameters
    ----------
    client : dask distributed computing client or None
        Either a concrete `dask client object <https://distributed.dask.org/en/latest/client.html>`_
        or `None`. If `None`, a global client is queried for and shut-down
        if found (without confirmation!).

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
    funcName = "<{}>".format(inspect.currentframe().f_code.co_name)

    # Attempt to establish connection to dask client
    if client is None:
        try:
            client = get_client()
        except ValueError:
            msg = "%s No dangling clients or clusters found."
            log.warning(msg, funcName)
            return
        except Exception as exc:
            log.error("%s Error looking for dask client", funcName)
            raise exc
    else:
        if not isinstance(client, Client):
            msg = "%s `client` has to be a dask client object, not %s"
            log.error(msg, funcName, str(type(client)))
            raise TypeError(msg%(funcName, str(type(client))))
    log.debug("%s Found client %s", funcName, str(client))

    # Prepare message for prompt
    if client.cluster.__class__.__name__ == "LocalCluster":
        userClust = "LocalCluster hosted on {}".format(client.scheduler_info()["address"])
    else:
        userName = getpass.getuser()
        outDir = client.cluster.job_header.partition("--output=")[-1]
        jobID = outDir.partition("{}_".format(userName))[-1].split(os.sep)[0]
        userClust = "cluster {0}_{1}".format(userName, jobID)
    nWorkers = count_online_workers(client.cluster)

    # If connection was successful, first close the client, then the cluster
    client.close()
    client.cluster.close()

    # Communicate what just happened and get outta here
    msg = "%s Successfully shut down %s containing %d workers"
    log.info(msg, funcName, userClust, nWorkers)

    return


def count_online_workers(cluster):
    """
    Local replacement for the late `._count_active_workers` class method
    """
    return len([w["memory_limit"] for w in cluster.scheduler_info["workers"].values()])
