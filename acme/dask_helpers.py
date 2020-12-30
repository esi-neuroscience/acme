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
import multiprocessing
from warnings import showwarning
import numpy as np
from tqdm import tqdm
if sys.platform == "win32":
    # tqdm breaks term colors on Windows - fix that (tqdm issue #446)
    import colorama
    colorama.deinit()
    colorama.init(strip=False)

# Local imports: differentiate b/w being imported as Syncopy sub-package or
# standalone ACME module
try:
    import syncopy
    isSpyModule = True
except ImportError:
    isSpyModule = False
from .shared import user_input, user_yesno
if isSpyModule:
    from syncopy import __dask__
    from syncopy.shared.parsers import scalar_parser, io_parser
    from syncopy.shared.errors import (SPYValueError, SPYTypeError, SPYIOError,
                                    SPYWarning)
else:
    import logging
    from warnings import showwarning

    from .shared import _scalar_parser as scalar_parser
    __dask__ = True
if __dask__:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, get_client
    from datetime import datetime, timedelta

__all__ = ["esi_cluster_setup", "cluster_cleanup"]


# Setup SLURM cluster
def esi_cluster_setup(partition="8GBS", n_jobs=2, mem_per_job="auto", n_jobs_startup=100,
                      timeout=60, interactive=True, interactive_wait=120, start_client=True,
                      **kwargs):
    """
    Start a distributed Dask cluster of parallel processing workers using SLURM
    (or local multi-processing)

    Parameters
    ----------
    partition : str
        Name of SLURM partition/queue to start workers in. Use the command `sinfo`
        in the terminal to see a list of available SLURM partitions on the ESI HPC
        cluster.
    n_jobs : int
        Number of SLURM jobs (=workers) to spawn
    mem_per_job : None or str
        Memory booking for each job. Can be specified either in megabytes
        (e.g., ``mem_per_job = 1500MB``) or gigabytes (e.g., ``mem_per_job = "2GB"``).
        If `mem_per_job` is `None`, or `"auto"` it is attempted to infer a sane default value
        from the chosen partition, e.g., for ``partition = "8GBS"`` `mem_per_job` is
        automatically set to the allowed maximum of `'8GB'`. However, even in
        queues with guaranted memory bookings, it is possible to allocate less
        memory than the allowed maximum per job to spawn numerous low-memory
        jobs. See Examples for details.
    n_jobs_startup : int
        Number of spawned jobs to wait for. If `n_jobs_startup` is `100` (default),
        the code does not proceed until either 100 SLURM jobs are running or the
        `timeout` interval has been exceeded.
    timeout : int
        Number of seconds to wait for requested jobs to start up (see `n_jobs_startup`).
    interactive : bool
        If `True`, user input is queried in case not enough jobs (set by `n_jobs_startup`)
        could be started in the provided waiting period (determined by `timeout`).
        The code waits `interactive_wait` seconds for a user choice - if none is
        provided, it continues with the current number of spawned jobs (if greater
        than zero). If `interactive` is `False` and no job could not be started
        within `timeout` seconds, a `TimeoutError` is raised.
    interactive_wait : int
        Countdown interval (seconds) to wait for a user response in case fewer than
        `n_jobs_startup` workers could be started. If no choice is provided within
        the given time, the code automatically proceeds with the current number of
        active workers.
    start_client : bool
        If `True`, a distributed computing client is launched and attached to
        the workers. If `start_client` is `False`, only a distributed
        computing cluster is started to which compute-clients can connect.
    **kwargs : dict
        Additional keyword arguments can be used to control job-submission details.

    Returns
    -------
    proc : object
        A distributed computing client (if ``start_client = True``) or
        a distributed computing cluster (otherwise).

    Examples
    --------
    The following command launches 10 SLURM jobs with 2 gigabytes memory each
    in the `8GBS` partition

    >>> client = esi_cluster_setup(n_jobs=10, partition="8GBS", mem_per_job="2GB")

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
    cluster_cleanup : remove dangling parallel processing job-clusters
    """

    # Re-direct printing/warnings to ACME logger outside SyNCoPy
    print, showwarning = _logging_setup()

    # For later reference: dynamically fetch name of current function
    funcName = ""
    if isSpyModule:
        funcName = "Syncopy "
    funcName = funcName + "<{}>".format(inspect.currentframe().f_code.co_name)

    # Be optimistic: prepare success message
    successMsg = "{name:s} Cluster dashboard accessible at {dash:s}"

    # Retrieve all partitions currently available in SLURM
    proc = subprocess.Popen("sinfo -h -o %P",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    out, err = proc.communicate()

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
                client = Client()
                successMsg = "{name:s} Local parallel computing client ready. \n" + successMsg
                print(successMsg.format(name=funcName, dash=client.cluster.dashboard_link))
                if start_client:
                    return client
                return client.cluster
            return

        # SLURM is installed, but something's wrong
        msg = "SLURM queuing system from node {node:s}. " +\
              "Original error message below:\n{error:s}"
        if isSpyModule:
            raise SPYIOError(msg.format(node=socket.gethostname(), error=err))
        else:
            raise IOError("{} Cannot access ".format(funcName) +\
                          msg.format(node=socket.gethostname(), error=err))
    options = out.split()

    # Make sure we're in a valid partition (exclude IT partitions from output message)
    if partition not in options:
        valid = list(set(options).difference(["DEV", "PPC"]))
        lgl = "'" + "or '".join(opt + "' " for opt in valid)
        if isSpyModule:
            raise SPYValueError(legal=lgl, varname="partition", actual=partition)
        else:
            msg = "{} Invalid partition selection {}, available SLURM partitions are {}"
            raise ValueError(msg.format(funcName, str(partition), lgl))

    # Parse job count
    try:
        scalar_parser(n_jobs, varname="n_jobs", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc

    # Get requested memory per job
    if isinstance(mem_per_job, str):
        if mem_per_job == "auto":
            mem_per_job = None
    if mem_per_job is not None:
        msg = "{} `mem_per_job` has to be a valid memory specifier (e.g., '8GB', '12000MB'), not {}"
        if not isinstance(mem_per_job, str):
            if isSpyModule:
                raise SPYTypeError(mem_per_job, varname="mem_per_job", expected="string")
            else:
                raise TypeError(msg.format(funcName, str(mem_per_job)))
        if not any(szstr in mem_per_job for szstr in ["MB", "GB"]):
            lgl = "string representation of requested memory (e.g., '8GB', '12000MB')"
            if isSpyModule:
                raise SPYValueError(legal=lgl, varname="mem_per_job", actual=mem_per_job)
            else:
                raise ValueError(msg.format(funcName, str(mem_per_job)))

    # Parse job-waiter count
    try:
        scalar_parser(n_jobs_startup, varname="n_jobs_startup", ntype="int_like", lims=[0, np.inf])
    except Exception as exc:
        raise exc

    # Query memory limit of chosen partition and ensure that `mem_per_job` is
    # set for partitions w/o limit
    idx = partition.find("GB")
    if idx > 0:
        mem_lim = int(partition[:idx]) * 1000
    else:
        if partition == "PREPO":
            mem_lim = 16000
        else:
            if mem_per_job is None:
                lgl = "explicit memory amount as required by partition '{}'"
                if isSpyModule:
                    raise SPYValueError(legal=lgl.format(partition),
                                        varname="mem_per_job", actual=mem_per_job)
                else:
                    msg = "{} `mem_per_job`: expected " + lgl + " not {}"
                    raise ValueError(msg.format(funcName, partition, mem_per_job))
        mem_lim = np.inf

    # Consolidate requested memory with chosen partition (or assign default memory)
    if mem_per_job is None:
        mem_per_job = str(mem_lim) + "MB"
    else:
        if "MB" in mem_per_job:
            mem_req = int(mem_per_job[:mem_per_job.find("MB")])
        else:
            mem_req = int(round(float(mem_per_job[:mem_per_job.find("GB")]) * 1000))
        if mem_req > mem_lim:
            msg = "`mem_per_job` exceeds limit of {lim:d}GB for partition {par:s}. " +\
                "Capping memory at partition limit. "
            if isSpyModule:
                SPYWarning(msg.format(lim=mem_lim, par=partition))
            else:
                msg = "{name:s} " + msg
                showwarning(msg.format(name=funcName, lim=mem_lim, par=partition),
                                       RuntimeWarning, __file__, inspect.currentframe().f_lineno)
            mem_per_job = str(int(mem_lim)) + "GB"

    # Parse requested timeout period
    try:
        scalar_parser(timeout, varname="timeout", ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc

    # Parse requested interactive waiting period
    try:
        scalar_parser(interactive_wait, varname="interactive_wait", ntype="int_like", lims=[0, np.inf])
    except Exception as exc:
        raise exc

    # Determine if cluster allocation is happening interactively
    if not isinstance(interactive, bool):
        if isSpyModule:
            raise SPYTypeError(interactive, varname="interactive", expected="bool")
        else:
            msg = "{} `interactive` has to be Boolean, not {}"
            raise TypeError(msg.format(funcName, str(interactive)))

    # Determine if a dask client was requested
    if not isinstance(start_client, bool):
        if isSpyModule:
            raise SPYTypeError(start_client, varname="start_client", expected="bool")
        else:
            msg = "{} `start_client` has to be Boolean, not {}"
            raise TypeError(msg.format(funcName, str(interactive)))

    # Set/get "hidden" kwargs
    workers_per_job = kwargs.get("workers_per_job", 1)
    try:
        scalar_parser(workers_per_job, varname="workers_per_job",
                      ntype="int_like", lims=[1, 8])
    except Exception as exc:
        raise exc

    n_cores = kwargs.get("n_cores", 1)
    try:
        scalar_parser(n_cores, varname="n_cores",
                      ntype="int_like", lims=[1, np.inf])
    except Exception as exc:
        raise exc

    slurm_wdir = kwargs.get("slurmWorkingDirectory", None)
    if slurm_wdir is None:
        usr = getpass.getuser()
        slurm_wdir = "/mnt/hpx/slurm/{usr:s}/{usr:s}_{date:s}"
        slurm_wdir = slurm_wdir.format(usr=usr,
                                       date=datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(slurm_wdir, exist_ok=True)
    else:
        if isSpyModule:
            try:
                io_parser(slurm_wdir, varname="slurmWorkingDirectory", isfile=False)
            except Exception as exc:
                raise exc
        else:
            msg = "{} `slurmWorkingDirectory` has to be an existing directory, not {}"
            if not isinstance(slurm_wdir, str):
                raise TypeError(msg.format(funcName, str(slurm_wdir)))
            if not os.path.isdir(os.path.expanduser(slurm_wdir)):
                raise ValueError(msg.format(funcName, str(slurm_wdir)))

    # Hotfix for upgraded cluster-nodes: point to correct Python executable if working from /home
    pyExec = sys.executable
    if sys.executable.startswith("/home"):
        pyExec = "/mnt/gs" + sys.executable

    # Create `SLURMCluster` object using provided parameters
    out_files = os.path.join(slurm_wdir, "slurm-%j.out")
    cluster = SLURMCluster(cores=n_cores,
                           memory=mem_per_job,
                           processes=workers_per_job,
                           local_directory=slurm_wdir,
                           queue=partition,
                           python=pyExec,
                           header_skip=["-t", "--mem"],
                           job_extra=["--output={}".format(out_files)])
                           # interface="asdf", # interface is set via `psutil.net_if_addrs()`
                           # job_extra=["--hint=nomultithread",
                           #            "--threads-per-core=1"]

    # Compute total no. of workers and up-scale cluster accordingly
    total_workers = n_jobs * workers_per_job
    worker_count = min(total_workers, n_jobs_startup)
    if worker_count < total_workers:
        # cluster.adapt(minimum=worker_count, maximum=total_workers)
        cluster.scale(total_workers)
        msg = "{} Requested job-count {} exceeds `n_jobs_startup`: " +\
            "waiting for {} jobs to come online, then proceed"
        print(msg.format(funcName, total_workers, n_jobs_startup))
    else:
        cluster.scale(total_workers)

    # Fire up waiting routine to avoid unfinished cluster setups
    if _cluster_waiter(cluster, funcName, worker_count, timeout, interactive, interactive_wait):
        return

    # Kill a zombie cluster in non-interactive mode
    if not interactive and _count_running_workers(cluster) == 0:
        cluster.close()
        err = "SLURM jobs could not be started within given time-out " +\
              "interval of {0:d} seconds"
        raise TimeoutError(err.format(timeout))

    # Highlight how to connect to dask performance monitor
    print(successMsg.format(name=funcName, dash=cluster.dashboard_link))

    # If client was requested, return that instead of the created cluster
    if start_client:
        return Client(cluster)
    return cluster


def _cluster_waiter(cluster, funcName, total_workers, timeout, interactive, interactive_wait):
    """
    Local helper that can be called recursively
    """

    # Re-direct printing/warnings to ACME logger outside SyNCoPy
    print, _ = _logging_setup()

    # Wait until all workers have been started successfully or we run out of time
    wrkrs = _count_running_workers(cluster)
    to = str(timedelta(seconds=timeout))[2:]
    fmt = "{desc}: {n}/{total} \t[elapsed time {elapsed} | timeout at " + to + "]"
    ani = tqdm(desc="{} SLURM workers ready".format(funcName), total=total_workers,
               leave=True, bar_format=fmt, initial=wrkrs)
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
        msg = "{name:s} SLURM jobs could not be started within given time-out " +\
              "interval of {time:d} seconds"
        print(msg.format(name=funcName, time=timeout))
        query = "{name:s} Do you want to [k]eep waiting for 60s, [a]bort or " +\
                "[c]ontinue with {wrk:d} workers?"
        choice = user_input(query.format(name=funcName, wrk=wrkrs), valid=["k", "a", "c"], default="c", timeout=interactive_wait)

        if choice == "k":
            return _cluster_waiter(cluster, funcName, total_workers, 60, True, 60)
        elif choice == "a":
            print("{} Closing cluster...".format(funcName))
            cluster.close()
            return True
        else:
            if wrkrs == 0:
                query = "{} Cannot continue with 0 workers. Do you want to " +\
                        "[k]eep waiting for 60s or [a]bort?"
                choice = user_input(query.format(funcName), valid=["k", "a"], default="a", timeout=60)
                if choice == "k":
                    _cluster_waiter(cluster, funcName, total_workers, 60, True, 60)
                else:
                    print("{} Closing cluster...".format(funcName))
                    cluster.close()
                    return True

    return False

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
    print, showwarning = _logging_setup()

    # For later reference: dynamically fetch name of current function
    funcName = ""
    if isSpyModule:
        funcName = "Syncopy "
    funcName = funcName + "<{}>".format(inspect.currentframe().f_code.co_name)

    # Attempt to establish connection to dask client
    if client is None:
        try:
            client = get_client()
        except ValueError:
            msg = "No dangling clients or clusters found."
            if isSpyModule:
                SPYWarning(msg)
            else:
                msg = "{name:s} " + msg
                showwarning(msg.format(name=funcName), RuntimeWarning,
                            __file__, inspect.currentframe().f_lineno)
            return
        except Exception as exc:
            raise exc
    else:
        if not isinstance(client, Client):
            if isSpyModule:
                raise SPYTypeError(client, varname="client", expected="dask client object")
            else:
                msg = "{} `client` has to be a dask client object, not {}"
                raise TypeError(msg.format(funcName, str(client)))

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
    print(msg.format(fname=funcName,
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
    Local helper for in-place substitutions of `print` and `showwarning` in ACME standalone mode
    """
    pFunc = print
    wFunc = showwarning
    if not isSpyModule:
        allLoggers = list(logging.root.manager.loggerDict.keys())
        idxList = [allLoggers.index(loggerName) for loggerName in allLoggers \
            for moduleName in ["ACME", "ParallelMap"] if moduleName in loggerName]
        if len(idxList) > 0:
            logger = logging.getLogger(allLoggers[idxList[0]])
            pFunc = logger.info
            wFunc = lambda msg, wrngType, fileName, lineNo: logger.warning(msg)
    return pFunc, wFunc
