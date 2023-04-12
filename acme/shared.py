#
# Auxiliaries needed across the entire package
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import os
import sys
import socket
import subprocess
import inspect
import logging
import traceback
import multiprocessing
import time
import numpy as np
import dask.distributed as dd
from tqdm import tqdm
from logging import handlers

# Local imports
from acme import __version__
from . import dask_helpers as dh

callCount = 0
callMax = 1000000

__all__ = []


def sizeOf(obj, varname):
    """
    Estimate memory consumption of Python objects

    Parameters
    ----------
    obj : Python object
        Any valid Python object whose memory footprint is of interest.

    Returns
    -------
    objsize : float
        Approximate memory footprint of `obj` in megabytes (MB).

    Notes
    -----
    Memory consumption is is estimated by recursively calling :meth:`sys.getsizeof`.
    Circular object references are followed up to a (preset) maximal recursion
    depth. This method was inspired by a routine in
    `Nifty <https://github.com/mwojnars/nifty/blob/master/util.py>`_.
    """

    # Keep track of the no. of recursive calls
    global callCount

    # For later reference: dynamically fetch name of current function
    funcName = "<{}>".format(inspect.currentframe().f_code.co_name)

    # Protect against circular object references
    callCount += 1
    if callCount >= callMax:
        msg = "%s maximum recursion depth %s exceeded while processing %s"
        raise RecursionError(msg%(funcName, callMax, varname))

    # Use `sys.getsizeof` to estimate memory consumption of primitive objects
    objsize = sys.getsizeof(obj) / 1024**2
    if isinstance(obj, dict):
        return objsize + sum(list(map(sizeOf, obj.keys(), [varname] * len(obj.keys())))) + sum(list(map(sizeOf, obj.values(), [varname] * len(obj.values()))))
    if isinstance(obj, (list, tuple, set)):
        return objsize + sum(list(map(sizeOf, obj, [varname] * len(obj))))
    return objsize


def is_slurm_node():
    """
    Returns `True` if code is running on a SLURM-managed cluster node, `False`
    otherwise
    """

    # Fetch ACME logger
    log = logging.getLogger("ACME")

    # Simply test if the srun command is available
    log.debug("Test if `sinfo` is available")
    out, _ = subprocess.Popen("sinfo --version",
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, shell=True).communicate()
    return len(out) > 0


def is_esi_node():
    """
    Returns `True` if code is running on an ESI cluster node, `False` otherwise
    """

    # Fetch ACME logger and write debug message
    log = logging.getLogger("ACME")
    log.debug("Test if hostname matches the pattern 'esi-sv*'")
    return socket.gethostname().startswith("esi-sv") and os.path.isdir("/cs")


def _scalar_parser(var, varname="varname", ntype="int_like", lims=[-np.inf, np.inf]):
    """
    ACME-specific version of Syncopy's `scalar_parser` (used for cross-compatibility)
    """

    # Get name of calling method/function
    log = logging.getLogger("ACME")
    funcName = "<{}>".format(inspect.currentframe().f_back.f_code.co_name)

    # Make sure `var` is a scalar-like number
    msg = "%s `%s` has to be %s between %s and %s, not %s"
    if np.issubdtype(type(var), np.number):
        error = False
        if ntype == "int_like":
            scalartype = "an integer"
            if round(var) != var:
                error = True
        else:
            scalartype = "a number"
        if var < lims[0] or var > lims[1]:
            error = True
        if error:
            raise ValueError(msg%(funcName,
                                  varname,
                                  scalartype,
                                  str(lims[0]),
                                  str(lims[1]),
                                  str(var)))
    else:
        msg = "%s `%s` has to be a scalar, not %s"
        raise TypeError(msg%(funcName, varname, str(type(var))))

    return


def user_yesno(msg, default=None):
    """
    ACME specific version of user-input query
    """

    # Parse optional `default` answer
    valid = {"yes": True, "y": True, "ye":True, "no":False, "n":False}
    if default is None:
        suffix = " [y/n] "
    elif default == "yes":
        suffix = " [Y/n] "
    elif default == "no":
        suffix = " [y/N] "

    # Wait for valid user input, if received return `True`/`False`
    while True:
        choice = input(msg + suffix).lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid.keys():
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def user_input(msg, valid, default=None, timeout=None):
    """
    ACME specific version of user-input query
    """

    # Add trailing whitespace to `msg` if not already present and append
    # default reply (if provided)
    suffix = "" + " " * (not msg.endswith(" "))
    if default is not None:
        default = default.replace("[", "").replace("]","")
        assert default in valid
        suffix = "[Default: '{}'] ".format(default)
    query = msg + suffix

    if timeout is None:
        return _get_user_input(query, valid, default)
    else:
        procQueue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_queuing_input,
                                       args=(procQueue,
                                             sys.stdin.fileno(),
                                             query,
                                             valid,
                                             default)
                                       )
        proc.start()
        countdown = tqdm(desc="Time remaining", leave=True, bar_format="{desc}: {n}  ",
                         initial=timeout, position=1)
        ticker = 0
        while procQueue.empty() and ticker < timeout:
            time.sleep(1)
            ticker += 1
            countdown.n = timeout - ticker
            countdown.refresh()   # force refresh to display elapsed time every second
        countdown.close()
        proc.terminate()

        if not procQueue.empty():
            choice = procQueue.get()
        else:
            choice = default
        return choice


def _get_user_input(query, valid, default):
    """
    Performs the actual input query
    """

    # Wait for valid user input and return choice upon receipt
    while True:
        choice = input(query)
        if default is not None and choice == "":
            return default
        elif choice in valid:
            return choice
        else:
            print("Please respond with '" + \
                "or '".join(opt + "' " for opt in valid) + "\n")


def _queuing_input(procQueue, stdin_fd, query, valid, default):
    """
    Target routine to tie subprocess to (in case input-query is time-restricted)
    """
    sys.stdin = os.fdopen(stdin_fd)
    procQueue.put(_get_user_input(query, valid, default))




def ctrlc_catcher(*excargs, **exckwargs):
    """
    Custom Traceback for properly handling CTRL + C interrupts while parallel
    computations are running
    """

    # Depending on the number of input arguments, we're either in Jupyter/iPython
    # or "regular" Python - this matters for actually handling the raised exception
    if len(excargs) == 3:
        isipy = False
        etype, evalue, etb = excargs
    else:
        shell, = excargs
        etype, evalue, etb = sys.exc_info()
        try:                            # careful: if iPython is used to launch a script, ``get_ipython`` is not defined
            ipy = get_ipython()
            isipy = True
            sys.last_traceback = etb    # smartify ``sys``
        except NameError:
            isipy = False

    # Prepare to log any uncaught exceptions
    log = logging.getLogger("ACME")

    # The only exception we really care about is a `KeyboardInterrupt`: if CTRL + C
    # is pressed, ensure graceful shutdown of any parallel processing clients
    if issubclass(etype, KeyboardInterrupt):
        try:
            client = dd.get_client()
        except ValueError:
            client = None
        if client is not None:
            for st in client.futures.values():
                st.cancel()
            client.futures.clear()
            dh.cluster_cleanup(client)
            log.debug("CTRL + C acknowledged, client and workers successfully killed")

    # Relay exception handling back to appropriate system tools
    if isipy:
        shell.ipyTBshower(shell, exc_tuple=(etype, evalue, etb), **exckwargs)
    else:
        sys.__excepthook__(etype, evalue, etb)

    # Write to all logging locations, manually print traceback to file (stdout
    # printing was handled above)
    log.error("Exception received.")
    memHandler = [h for h in log.handlers if isinstance(h, handlers.MemoryHandler)][0]
    if memHandler.target is not None:
        memHandler.acquire()
        with open(memHandler.target.baseFilename, "a") as logfile:
            logfile.write("".join(traceback.format_exception_only(etype, evalue)))
            logfile.write("".join(traceback.format_tb(etb)))
        memHandler.release()

    return
