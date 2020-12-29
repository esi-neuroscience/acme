# -*- coding: utf-8 -*-
#
# Auxiliaries needed across the entire package
#

# Builtin/3rd party package imports
import os
import sys
import subprocess
import inspect
import numbers
import logging
import warnings
import datetime
import multiprocessing
import time
import numpy as np
from tqdm import tqdm

callCount = 0
callMax = 50000

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

    # Protect against circular object references
    callCount += 1
    if callCount >= callMax:
        msgName = sys._getframe().f_back.f_code.co_name
        msg = "{} maximum recursion depth {} exceeded when processing {}"
        raise RecursionError(msg.format(msgName, callMax, varname))

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

    # Simply test if the srun command is available
    out, _ = subprocess.Popen("srun --version",
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, shell=True).communicate()
    if len(out) > 0:
        return True
    else:
        return False


def _scalar_parser(var, varname="varname", ntype="int_like", lims=[-np.inf, np.inf]):
    """
    ACME-specific version of Syncopy's `scalar_parser` (used for cross-compatibility)
    """

    # Get name of calling method/function
    caller = "<{}>".format(inspect.currentframe().f_back.f_code.co_name)

    # Make sure `var` is a scalar-like number
    msg = "{caller:s} `{varname:s}` has to be {scalartype:s} between {lower:s} and {upper:s}, not {var:s}"
    if isinstance(var, numbers.Number):
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
            raise ValueError(msg.format(caller=caller,
                                        varname=varname,
                                        scalartype=scalartype,
                                        lower=str(lims[0]),
                                        upper=str(lims[1]),
                                        var=str(var)))
    else:
        msg = "{caller:s} `{varname:s}` has to be a scalar, not {var:s}"
        raise TypeError(msg.format(caller=caller, varname=varname, var=str(var)))

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


def prepare_log(func, caller=None, logfile=False, verbose=None):
    """
    Convenience function to set up ACME logger

    Parameters
    ----------
    func : callable
        User-provided function to be called concurrently by ACME
    caller : None or str
        Routine/class that initiated logging (presumable :class:~`acme.ParallelMap`
        or :class:~`acme.ACMEDaemon`)
    logfile : None or bool or str
        If `True` an auto-generated log-file is set up. If `logfile` is a string
        it is interpreted as file-name for a new log-file (must not exist). If
        `False` or `None` logging information is streamed to stdout only.
    verbose : bool or None
        If `None`, the logging-level only contains messages of `'INFO'` priority and
        higher (`'WARNING'` and `'ERROR'`). If `verbose` is `True`, logging is
        performed on ``DEBUG`', `'INFO`', `'WARNING'` and `'ERROR'` levels. If
        `verbose` is `False` only `'WARNING'` and `'ERROR'` messages are propagated.

    Returns
    -------
    log : logger object
        A Python :class:`logging.Logger` instance
    """

    # If not provided, get name of calling method/function
    if caller is None:
        caller = "<{}>".format(inspect.currentframe().f_back.f_code.co_name)
    elif not isinstance(caller, str):
        msg = "{} `caller` has to be a string, not {}"
        raise TypeError(msg.format(inspect.currentframe().f_back.f_code.co_name),
                        str(caller))

    # Basal sanity check for Boolean flag
    if verbose is not None and not isinstance(verbose, bool):
        msg = "{} `verbose` has to be `True`, `False` or `None`, not {}"
        raise TypeError(msg.format(caller, str(verbose)))

    # Either parse provided `logfile` or set up an auto-generated file
    msg = "{} `logfile` has to be `None`, `True`, `False` or a valid file-name, not {}"
    if logfile is None or isinstance(logfile, bool):
        if logfile is True:
            logfile = os.path.dirname(os.path.abspath(inspect.getfile(func)))
            logfile = os.path.join(logfile, "ACME_{func:s}_{date:s}.log")
            logfile = logfile.format(func=func.__name__,
                                     date=datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        else:
            logfile = None
    elif isinstance(logfile, str):
        if os.path.isdir(logfile):
            raise IOError(msg.format(caller, "a directory"))
        logfile = os.path.abspath(os.path.expanduser(logfile))
    else:
        raise TypeError(msg.format(caller, str(logfile)))
    if logfile is not None and os.path.isfile(logfile):
        msg = "{} log-file {} already exists, appending to it"
        warnings.showwarning(msg.format(caller, logfile), RuntimeWarning,
                             __file__, inspect.currentframe().f_lineno)

    # Set logging verbosity based on `verbose` flag
    if verbose is None:
        loglevel = logging.INFO
    elif verbose is True:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.WARNING
    log = logging.getLogger(caller)
    log.setLevel(loglevel)

    # Create logging formatter
    formatter = logging.Formatter("%(name)s %(levelname)s: %(message)s")

    # Output handlers: print log messages to `stderr` via `StreamHandler` as well
    # as to a provided text file `logfile using a `FileHandler`
    if len(log.handlers) == 0:
        stdoutHandler = logging.StreamHandler()
        stdoutHandler.setLevel(loglevel)
        stdoutHandler.setFormatter(formatter)
        log.addHandler(stdoutHandler)
    if logfile is not None and \
        all(not isinstance(handler, logging.FileHandler) for handler in log.handlers):
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setLevel(loglevel)
        fileHandler.setFormatter(formatter)
        log.addHandler(fileHandler)

    return log
