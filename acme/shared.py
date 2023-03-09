# -*- coding: utf-8 -*-
#
# Auxiliaries needed across the entire package
#

# Builtin/3rd party package imports
import os
import sys
import socket
import subprocess
import inspect
import logging
import traceback
import warnings
import datetime
import multiprocessing
import time
import numpy as np
import dask.distributed as dd
from tqdm import tqdm

# from .dask_helpers import cluster_cleanup
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
    out, _ = subprocess.Popen("sinfo --version",
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, shell=True).communicate()
    return len(out) > 0


def is_esi_node():
    """
    Returns `True` if code is running on an ESI cluster node, `False` otherwise
    """
    return socket.gethostname().startswith("esi-sv") and os.path.isdir("/cs")


def _scalar_parser(var, varname="varname", ntype="int_like", lims=[-np.inf, np.inf]):
    """
    ACME-specific version of Syncopy's `scalar_parser` (used for cross-compatibility)
    """

    # Get name of calling method/function
    caller = "<{}>".format(inspect.currentframe().f_back.f_code.co_name)

    # Make sure `var` is a scalar-like number
    msg = "{caller:s} `{varname:s}` has to be {scalartype:s} between {lower:s} and {upper:s}, not {var:s}"
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
    # formatter = logging.Formatter("%(name)s %(levelname)s: %(message)s")
    formatter = AcmeFormatter("%(name)s %(levelname)s %(message)s")

    # Output handlers: print log messages to `stderr` via `StreamHandler` as well
    # as to a provided text file `logfile using a `FileHandler`.
    # Note: avoid adding the same log-file location as distinct handlers to the logger
    # in case `ParallelMap` is executed repeatedly; also remove existing non-default
    # logfile handlers to avoid generating multiple logs (and accidental writes to existing logs)
    if len(log.handlers) == 0:
        stdoutHandler = logging.StreamHandler()
        stdoutHandler.setLevel(loglevel)
        stdoutHandler.setFormatter(formatter)
        log.addHandler(stdoutHandler)
    if logfile is not None:
        fHandlers = [h for h in log.handlers if isinstance(h, logging.FileHandler)]
        for handler in fHandlers:
            if handler.baseFilename == logfile:
                break
            else:
                log.handlers.remove(handler)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setLevel(loglevel)
        fileHandler.setFormatter(formatter)
        log.addHandler(fileHandler)

    # Start log w/version info
    log.info("\x1b[1mThis is ACME v. %s\x1b[0m", __version__)

    return log


class AcmeFormatter(logging.Formatter):
    """
    Adapted from https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
    """

    green = "\x1b[92m"
    gray = "\x1b[90m"
    blue = "\x1b[38;5;39m"
    magenta = "\x1b[35m"
    red = "\x1b[38;5;196m"
    bold = "\x1b[1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()

        fmtName = fmt.partition("%(name)s")
        fmtName = fmtName[0] + self.bold + fmtName[1] + self.reset + fmtName[2]
        fmt = "".join(fmtName)

        fmtLvl = fmt.partition("%(levelname)s")
        fmtDebug = fmtLvl[0] + self.bold + self.green + \
            "# " + fmtLvl[1] + " #" + self.reset + self.gray + fmtLvl[2] + self.reset
        fmtInfo = fmtLvl[0] + self.bold + self.blue + \
            "- " + fmtLvl[1] + " -" + self.reset + fmtLvl[2]
        fmtWarn = fmtLvl[0] + self.bold + self.magenta + \
            "! " + fmtLvl[1] + " !" + self.reset + fmtLvl[2]
        fmtError = fmtLvl[0] + self.bold + self.red + \
            "> " + fmtLvl[1] + " <" + self.reset + self.red + fmtLvl[2] + self.reset

        self.FORMATS = {
            logging.DEBUG: "".join(fmtDebug),
            logging.INFO: "".join(fmtInfo),
            logging.WARNING: "".join(fmtWarn),
            logging.ERROR: "".join(fmtError),
            logging.CRITICAL: "".join(fmtError),
        }

    def format(self, record):
        logFmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(logFmt)
        return formatter.format(record)


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
    fHandlers = [h for h in log.handlers if isinstance(h, logging.FileHandler)]
    for handler in fHandlers:
        handler.acquire()
        with open(handler.baseFilename, "a") as logfile:
            logfile.write("".join(traceback.format_exception_only(etype, evalue)))
            logfile.write("".join(traceback.format_tb(etb)))
        handler.release()

    return
