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

# Local imports: differentiate b/w being imported as Syncopy sub-package or
# standalone ACME module: if imported by Syncopy, use some lambda magic to avoid
# circular imports due to (at import-time) only partially initialized Syncopy
from .shared import user_input, user_yesno
if "syncopy" in sys.modules:
    isSpyModule = True
    import syncopy as spy
    customIOError = lambda msg : spy.shared.errors.SPYIOError(msg)
    customValueError = lambda legal=None, varname=None, actual=None : \
        spy.shared.errors.SPYValueError(legal=legal, varname=varname, actual=actual)
    customTypeError = lambda val, varname=None, expected=None : \
        spy.shared.errors.SPYTypeError(val, varname=varname, expected=expected)
    scalar_parser = lambda var, varname="", ntype=None, lims=None : \
        spy.shared.parsers.scalar_parser(var, varname=varname, ntype=ntype, lims=lims)
else:
    isSpyModule = False
    from warnings import showwarning
    import logging
    from .shared import _scalar_parser as scalar_parser
    customIOError = IOError
    customValueError = lambda legal=None, varname=None, actual=None : ValueError(legal)
    customTypeError = lambda msg, varname=None, expected=None : TypeError(msg)

from dask_jobqueue import SLURMCluster
from dask.distributed import Client, get_client
from datetime import datetime, timedelta

# Be optimistic: prepare success message to be used throughout this module
_successMsg = "{name:s} Cluster dashboard accessible at {dash:s}"

__all__ = ["esi_cluster_setup", "local_cluster_setup", "cluster_cleanup", "slurm_cluster_setup"]


# Setup SLURM workers on the ESI HPC cluster
def esi_cluster_setup(partition="8GBXS", n_workers=2, mem_per_worker="auto", n_workers_startup=100,
                      timeout=60, interactive=True, interactive_wait=120, start_client=True,
                      job_extra=[], **kwargs):
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
    slurm_cluster_setup : start a distributed Dask cluster of parallel processing workers using SLURM
    local_cluster_setup : start a local Dask multi-processing cluster on the host machine
    cluster_cleanup : remove dangling parallel processing worker-clusters
    """

    # Re-direct printing/warnings to ACME logger outside of SyNCoPy
    customPrint, _ = _logging_setup()

    # For later reference: dynamically fetch name of current function
    funcName = "{pre:s}<{pkg:s}{name:s}> ".format(pre="Syncopy " if isSpyModule else "",
                                                 pkg="ACME: " if isSpyModule else "",
                                                 name=inspect.currentframe().f_code.co_name)

    # Don't start a new cluster on top of an existing one
    try:
        client = get_client()
        msg = "{}Found existing parallel computing client {}. Not starting new cluster."
        customPrint(msg.format(funcName, str(client)))
        if start_client:
            return client
        return client.cluster
    except ValueError:
        pass

    # Check if SLURM's `sinfo` can be accessed
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
        msg = "{preamble:s}SLURM queuing system from node {node:s}. " +\
              "Original error message below:\n{error:s}"
        raise customIOError(msg.format(preamble=funcName + " Cannot access " if not isSpyModule else "",
                                       node=socket.gethostname(),
                                       error=err))

    # Use default by-worker process count or extract it from anonymous keyword args (if provided)
    processes_per_worker = kwargs.pop("processes_per_worker", 1)

    # If partition is "auto" use `mem_per_worker` to pick pseudo-optimal partition
    # Note: the `np.where` gymnastic below is necessary since `argmin` refuses
    # to return multiple matches; if `mem_per_worker` is 12, then ``memDiff = [4, 4, ...]``,
    # however, 8GB won't fit a 12GB worker, so we have to pick the second match 16GB
    if isinstance(partition, str) and partition == "auto":
        if not isinstance(mem_per_worker, str) or mem_per_worker.find("estimate_memuse:") < 0:
            msg = "{preamble:s}automatic partition selector without first invoking memory estimation in `ParallelMap`. "
            raise customIOError(msg.format(preamble=funcName + " Cannot access " if not isSpyModule else ""))
        memEstimate = int(mem_per_worker.replace("estimate_memuse:" ,""))
        mem_per_worker = "auto"
        customPrint("{}Automatically selecting SLURM partition...".format(funcName))
        availPartitions = _get_slurm_partitions(funcName)
        gbQueues = np.unique([int(queue.split("GB")[0]) for queue in availPartitions if queue[0].isdigit()])
        memDiff = np.abs(gbQueues - memEstimate)
        queueIdx = np.where(memDiff == memDiff.min())[0][-1]
        partition = "{}GBXS".format(gbQueues[queueIdx])
        msg = "{preamble:s}Picked partition {p:s} based on estimated memory consumption of {m:d} GB"
        customPrint(msg.format(preamble=funcName, p=partition, m=memEstimate))

    # Extract by-worker CPU core count from anonymous keyword args or...
    if kwargs.get("n_cores") is not None:
        n_cores = kwargs.pop("n_cores")
    else:
        # ...get memory limit (*in MB*) of chosen partition and set core count
        # accordingly (multiple of 8 wrt to GB RAM)
        try:
            pc = subprocess.run("scontrol -o show partition {}".format(partition),
                                capture_output=True, check=True, shell=True, text=True)
            defMem = int(pc.stdout.strip().partition("DefMemPerCPU=")[-1].split()[0])
        except Exception as exc:
            msg = "{preamble:s}available memory per CPU in chosen SLURM partition. " +\
                "Original error message below:\n{error:s}"
            raise customIOError(msg.format(preamble=funcName + " Cannot access " if not isSpyModule else "",
                                           error=str(exc)))
        n_cores = int(defMem / 8000)

    # Determine if `job_extra`` is a list (this is also checked in `slurm_cluster_setup`,
    # but we may need to append to it, so ensure that's possible)
    if not isinstance(job_extra, list):
        msg = "{} `job_extra` has to be List, not {}"
        raise customTypeError(job_extra if isSpyModule else msg.format(funcName, str(job_extra)),
                              varname="job_extra",
                              expected="list")

    # If '--output' was not provided, append default output folder to `job_extra`
    if not any(option.startswith("--output") or option.startswith("-o") for option in job_extra):
        usr = getpass.getuser()
        slurm_wdir = "/cs/slurm/{usr:s}/{usr:s}_{date:s}"
        slurm_wdir = slurm_wdir.format(usr=usr,
                                       date=datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(slurm_wdir, exist_ok=True)
        out_files = os.path.join(slurm_wdir, "slurm-%j.out")
        job_extra.append("--output={}".format(out_files))

    # Let the SLURM-specific setup function do the rest (returns client or cluster)
    return slurm_cluster_setup(partition, n_cores, n_workers, processes_per_worker, mem_per_worker,
                               n_workers_startup, timeout, interactive, interactive_wait,
                               start_client, job_extra, invalid_partitions=["DEV", "PPC"], **kwargs)


# Setup SLURM cluster
def slurm_cluster_setup(partition, n_cores, n_workers, processes_per_worker, mem_per_worker,
                        n_workers_startup, timeout, interactive, interactive_wait,
                        start_client, job_extra, invalid_partitions=[]):
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
    esi_cluster_setup : start a SLURM worker cluster on the ESI HPC infrastructure
    local_cluster_setup : start a local Dask multi-processing cluster on the host machine
    cluster_cleanup : remove dangling parallel processing worker-clusters
    """

    # Re-direct printing/warnings to ACME logger outside of SyNCoPy
    customPrint, customWarning = _logging_setup()

    # For later reference: dynamically fetch name of current function
    funcName = "{pre:s}<{pkg:s}{name:s}>".format(pre="Syncopy " if isSpyModule else "",
                                                 pkg="ACME: " if isSpyModule else "",
                                                 name=inspect.currentframe().f_code.co_name)

    # Retrieve all partitions currently available in SLURM
    availPartitions = _get_slurm_partitions(funcName)

    # Make sure we're in a valid partition
    if partition not in availPartitions:
        valid = list(set(availPartitions).difference(invalid_partitions))
        lgl = "'" + "or '".join(opt + "' " for opt in valid)
        msg = "{} Invalid partition selection {}, available SLURM partitions are {}"
        raise customValueError(legal=lgl if isSpyModule else msg.format(funcName, str(partition), lgl),
                               varname="partition",
                               actual=partition)

    # Parse worker count
    try:
        scalar_parser(n_workers, varname="n_workers", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc

    # Get requested memory per worker
    if isinstance(mem_per_worker, str):
        if mem_per_worker == "auto":
            mem_per_worker = None
    if mem_per_worker is not None:
        msg = "{} `mem_per_worker` has to be a valid memory specifier (e.g., '8GB', '12000MB'), not {}"
        lgl = "string representation of requested memory (e.g., '8GB', '12000MB')"
        if not isinstance(mem_per_worker, str):
            raise customTypeError(mem_per_worker if isSpyModule else msg.format(funcName, str(mem_per_worker)),
                                  varname="mem_per_worker",
                                  expected="string")
        if not any(szstr in mem_per_worker for szstr in ["MB", "GB"]):
            raise customValueError(legal=lgl if isSpyModule else msg.format(funcName, str(mem_per_worker)),
                                   varname="mem_per_worker",
                                   actual=mem_per_worker)
        memNumeric = mem_per_worker.replace("MB","").replace("GB","")
        try:
            memVal = float(memNumeric)
        except:
            raise customValueError(legal=lgl if isSpyModule else msg.format(funcName, str(mem_per_worker)),
                                   varname="mem_per_worker",
                                   actual=mem_per_worker)
        if memVal <= 0:
            raise customValueError(legal=lgl if isSpyModule else msg.format(funcName, str(mem_per_worker)),
                                   varname="mem_per_worker",
                                   actual=mem_per_worker)

    # Parse worker-waiter count
    try:
        scalar_parser(n_workers_startup, varname="n_workers_startup", ntype="int_like", lims=[0, np.inf])
    except Exception as exc:
        raise exc

    # Get memory limit (*in MB*) of chosen partition (guaranteed to exist, cf. above)
    pc = subprocess.run("scontrol -o show partition {}".format(partition),
                        capture_output=True, check=True, shell=True, text=True)
    try:
        mem_lim = int(pc.stdout.strip().partition("MaxMemPerCPU=")[-1].split()[0])
    except IndexError:
        try:
            mem_lim = int(pc.stdout.strip().partition("DefMemPerCPU=")[-1].split()[0])
        except IndexError:
            mem_lim = np.inf

    # Consolidate requested memory with chosen partition (or assign default memory)
    if mem_per_worker is None:
        mem_per_worker = str(mem_lim) + "MB"
    else:
        if "MB" in mem_per_worker:
            mem_req = int(mem_per_worker[:mem_per_worker.find("MB")])
        else:
            mem_req = int(round(float(mem_per_worker[:mem_per_worker.find("GB")]) * 1000))
        if mem_req > mem_lim:
            msg = "{name:s}`mem_per_worker` exceeds limit of {lim:d}GB for partition {par:s}. " +\
                "Capping memory at partition limit. "
            customWarning(msg.format(name=funcName + " " if not isSpyModule else "", lim=mem_lim, par=partition),
                          RuntimeWarning, __file__, inspect.currentframe().f_lineno)
            mem_per_worker = str(int(mem_lim)) + "GB"

    # Parse requested timeout period
    try:
        scalar_parser(timeout, varname="timeout", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc

    # Parse requested interactive waiting period
    try:
        scalar_parser(interactive_wait, varname="interactive_wait", ntype="int_like",
                      lims=[0, np.inf])
    except Exception as exc:
        raise exc

    # Determine if cluster allocation is happening interactively
    if not isinstance(interactive, bool):
        msg = "{} `interactive` has to be Boolean, not {}"
        raise customTypeError(interactive if isSpyModule else msg.format(funcName, str(interactive)),
                              varname="interactive",
                              expected="bool")

    # Determine if a dask client was requested
    if not isinstance(start_client, bool):
        msg = "{} `start_client` has to be Boolean, not {}"
        raise customTypeError(start_client if isSpyModule else msg.format(funcName, str(start_client)),
                              varname="start_client",
                              expected="bool")

    # Determine if job_extra is a list
    if not isinstance(job_extra, list):
        msg = "{} `job_extra` has to be List, not {}"
        raise customTypeError(job_extra if isSpyModule else msg.format(funcName, str(job_extra)),
                              varname="job_extra",
                              expected="list")

    # Determine if job_extra options are valid
    for option in job_extra:
        msg = "{} `job_extra` has to be a valid sbatch option, not {}"
        if not isinstance(option, str):
            raise customTypeError(option if isSpyModule else msg.format(funcName, str(option)),
                                  varname="option",
                                  expected="string")
        if not option[0] == "-":
            lgl = "job_extra options should be flagged with - or --"
            raise customValueError(legal=lgl, varname="option", actual=option)

    # Ensure validity of requested worker processes
    try:
        scalar_parser(processes_per_worker, varname="processes_per_worker",
                      ntype="int_like", lims=[1, 16])
    except Exception as exc:
        raise exc

    # Check for sanity of requested core count
    try:
        scalar_parser(n_cores, varname="n_cores",
                      ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc

    # Check validity of '--output' option if provided
    userOutSpec = [option.startswith("--output") or option.startswith("-o") for option in job_extra]
    if any(userOutSpec):
        userOut = job_extra[userOutSpec.index(True)]
        outSpec = userOut.split("=")
        if len(outSpec) != 2:
            lgl = "the SLURM output directory must be specified using -o/--output=/path/to/file"
            raise customValueError(legal=lgl if isSpyModule else "{} {}, not {}".format(funcName, lgl, userOut),
                                   varname="job_extra",
                                   actual=userOut)
        slurm_wdir = os.path.split(outSpec[1])[0]
        if len(slurm_wdir) > 0:
            if isSpyModule:
                try:
                    spy.shared.parsers.io_parser(slurm_wdir, varname="job_extra", isfile=False)
                except Exception as exc:
                    raise exc
            else:
                msg = "{} `slurmWorkingDirectory` has to be an existing directory, not {}"
                if not os.path.isdir(os.path.expanduser(slurm_wdir)):
                    raise ValueError(msg.format(funcName, str(slurm_wdir)))

    # Create `SLURMCluster` object using provided parameters
    cluster = SLURMCluster(cores=n_cores,
                           memory=mem_per_worker,
                           processes=processes_per_worker,
                           local_directory=slurm_wdir,
                           queue=partition,
                           python=sys.executable,
                           header_skip=["-t", "--mem"],
                           job_extra=job_extra)
                           # interface="asdf", # interface is set via `psutil.net_if_addrs()`

    # Compute total no. of workers and up-scale cluster accordingly
    total_workers = n_workers * processes_per_worker
    worker_count = min(total_workers, n_workers_startup)
    if worker_count < total_workers:
        # cluster.adapt(minimum=worker_count, maximum=total_workers)
        cluster.scale(total_workers)
        msg = "{} Requested worker-count {} exceeds `n_workers_startup`: " +\
            "waiting for {} workers to come online, then proceed"
        customPrint(msg.format(funcName, total_workers, n_workers_startup))
    else:
        cluster.scale(total_workers)

    # Fire up waiting routine to avoid unfinished cluster setups
    if _cluster_waiter(cluster, funcName, worker_count, timeout, interactive, interactive_wait):
        return

    # Kill a zombie cluster in non-interactive mode
    if not interactive and _count_running_workers(cluster) == 0:
        cluster.close()
        err = "SLURM workers could not be started within given time-out " +\
              "interval of {0:d} seconds"
        raise TimeoutError(err.format(timeout))

    # Highlight how to connect to dask performance monitor
    customPrint(_successMsg.format(name=funcName, dash=cluster.dashboard_link))

    # If client was requested, return that instead of the created cluster
    if start_client:
        return Client(cluster)
    return cluster


def _get_slurm_partitions(funcName):
    """
    Coming soon...
    """

    # Retrieve all partitions currently available in SLURM
    proc = subprocess.Popen("sinfo -h -o %P",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    out, err = proc.communicate()

    # Any non-zero return-code means SLURM is not ready to use
    if proc.returncode != 0:
        msg = "{preamble:s}SLURM queuing system from node {node:s}. " +\
              "Original error message below:\n{error:s}"
        raise customIOError(msg.format(preamble=funcName + " Cannot access " if not isSpyModule else "",
                                       node=socket.gethostname(),
                                       error=err))

    # Return formatted subprocess shell output
    return out.split()


def _cluster_waiter(cluster, funcName, total_workers, timeout, interactive, interactive_wait):
    """
    Local helper that can be called recursively
    """

    # Re-direct printing/warnings to ACME logger outside SyNCoPy
    customPrint, _ = _logging_setup()

    # Wait until all workers have been started successfully or we run out of time
    wrkrs = _count_running_workers(cluster)
    to = str(timedelta(seconds=timeout))[2:]
    fmt = "{desc}: {n}/{total} \t[elapsed time {elapsed} | timeout at " + to + "]"
    ani = tqdm(desc="{} SLURM workers ready".format(funcName), total=total_workers,
               leave=True, bar_format=fmt, initial=wrkrs, position=0)
    counter = 0
    while _count_running_workers(cluster) < total_workers and counter < timeout:
        time.sleep(1)
        counter += 1
        ani.update(max(0, _count_running_workers(cluster) - wrkrs))
        wrkrs = _count_running_workers(cluster)
        ani.refresh()   # force refresh to display elapsed time every second
    ani.close()

    # If we ran out of time before all workers could be started, ask what to do
    if counter == timeout and interactive:
        msg = "{name:s} SLURM workers could not be started within given time-out " +\
              "interval of {time:d} seconds"
        customPrint(msg.format(name=funcName, time=timeout))
        query = "{name:s} Do you want to [k]eep waiting for 60s, [a]bort or " +\
                "[c]ontinue with {wrk:d} workers?"
        choice = user_input(query.format(name=funcName, wrk=wrkrs),
                            valid=["k", "a", "c"], default="c", timeout=interactive_wait)

        if choice == "k":
            return _cluster_waiter(cluster, funcName, total_workers, 60, True, 60)
        elif choice == "a":
            customPrint("{} Closing cluster...".format(funcName))
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
                    customPrint("{} Closing cluster...".format(funcName))
                    cluster.close()
                    return True

    return False


def local_cluster_setup(interactive=True, start_client=True):
    """
    Start a local distributed Dask multi-processing cluster

    Parameters
    ----------
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
    esi_cluster_setup : Start a distributed Dask cluster using SLURM
    cluster_cleanup : remove dangling parallel processing worker-clusters
    """

    # Re-direct printing/warnings to ACME logger outside of SyNCoPy
    customPrint, _ = _logging_setup()

    # For later reference: dynamically fetch name of current function
    funcName = "{pre:s}<{pkg:s}{name:s}>".format(pre="Syncopy " if isSpyModule else "",
                                                 pkg="ACME: " if isSpyModule else "",
                                                 name=inspect.currentframe().f_code.co_name)

    # Determine if cluster allocation is happening interactively
    if not isinstance(interactive, bool):
        msg = "{} `interactive` has to be Boolean, not {}"
        raise customTypeError(interactive if isSpyModule else msg.format(funcName, str(interactive)),
                              varname="interactive",
                              expected="bool")

    # Determine if a dask client was requested
    if not isinstance(start_client, bool):
        msg = "{} `start_client` has to be Boolean, not {}"
        raise customTypeError(start_client if isSpyModule else msg.format(funcName, str(start_client)),
                              varname="start_client",
                              expected="bool")

    # Check, if we're running inside a Jupyter notebook...
    try:
        ipy = get_ipython()
        if ipy.__class__.__name__ == "ZMQInteractiveShell":
            maybeScript = False # Jupyter Notebook
        else:
            maybeScript = True  # iPython shell
    except NameError:
        maybeScript = True      # Python shell

    # ...if not, print warning/info message
    if maybeScript:
        msg = """\
        {name:s}If you use a script to start a local parallel computing client, please ensure
        the call to `local_cluster_setup` is wrapped inside a main module block, i.e.,

            if __name__ == "__main__":
                ...
                local_cluster_setup()
                ...

        Otherwise, a RuntimeError is raised due to an infinite recursion triggered by
        new processes being started before the calling process can finish its bootstrapping
        phase.
        """
        customPrint(textwrap.dedent(msg.format(name=funcName + " " if not isSpyModule else "")))

    # Additional safe-guard: if a script is executed, double-check with the user
    # for proper main idiom usage
    if interactive:
        msg = "{name:s}If launched from a script, did you wrap your code inside a main module block?"
        if not user_yesno(msg.format(name=funcName + " " if not isSpyModule else ""), default="no"):
            return

    # Start the actual distributed client
    client = Client()
    successMsg = "{name:s} Local parallel computing client ready. \n" + _successMsg
    customPrint(successMsg.format(name=funcName, dash=client.cluster.dashboard_link))
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
    """

    # Re-direct printing/warnings to ACME logger outside SyNCoPy
    customPrint, customWarning = _logging_setup()

    # For later reference: dynamically fetch name of current function
    funcName = "{pre:s}<{pkg:s}{name:s}>".format(pre="Syncopy " if isSpyModule else "",
                                                 pkg="ACME: " if isSpyModule else "",
                                                 name=inspect.currentframe().f_code.co_name)

    # Attempt to establish connection to dask client
    if client is None:
        try:
            client = get_client()
        except ValueError:
            msg = "{name:s}No dangling clients or clusters found."
            customWarning(msg.format(name="" if isSpyModule else funcName + " "),
                          RuntimeWarning,
                          __file__,
                          inspect.currentframe().f_lineno)
            return
        except Exception as exc:
            raise exc
    else:
        if not isinstance(client, Client):
            msg = "{} `client` has to be a dask client object, not {}"
            customTypeError(client if isSpyModule else msg.format(funcName, str(client)),
                            varname="client",
                            expected="dask client object")

    # Prepare message for prompt
    if client.cluster.__class__.__name__ == "LocalCluster":
        userClust = "LocalCluster hosted on {}".format(client.scheduler_info()["address"])
    else:
        userName = getpass.getuser()
        outDir = client.cluster.job_header.partition("--output=")[-1]
        jobID = outDir.partition("{}_".format(userName))[-1].split(os.sep)[0]
        userClust = "cluster {0}_{1}".format(userName, jobID)
    nWorkers = len(client.cluster.workers)

    # If connection was successful, first close the client, then the cluster
    client.close()
    client.cluster.close()

    # Communicate what just happened and get outta here
    msg = "{fname:s} Successfully shut down {cname:s} containing {nj:d} workers"
    customPrint(msg.format(fname=funcName,
                     nj=nWorkers,
                     cname=userClust))

    return


def _count_running_workers(cluster):
    """
    Local replacement for the late `._count_active_workers` class method
    """
    return len(cluster.scheduler_info.get('workers'))


def _logging_setup():
    """
    Local helper to customize warning and print functionality at runtime
    """
    if isSpyModule:
        pFunc = print
        wFunc = lambda msg, kind, caller, lineno : spy.shared.errors.SPYWarning(msg, caller=caller)
    else:
        pFunc = print
        wFunc = showwarning
        allLoggers = list(logging.root.manager.loggerDict.keys())
        idxList = [allLoggers.index(loggerName) for loggerName in allLoggers \
            for moduleName in ["ACME", "ParallelMap"] if moduleName in loggerName]
        if len(idxList) > 0:
            logger = logging.getLogger(allLoggers[idxList[0]])
            pFunc = logger.info
            wFunc = lambda msg, kind, caller, lineno: logger.warning(msg)
    return pFunc, wFunc
