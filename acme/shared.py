#
# Auxiliaries needed across the entire package
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
import select
import subprocess
import inspect
import logging
import psutil
import traceback
import numpy as np
import dask.distributed as dd
from logging import handlers
from typing import Any, Optional, List

# Local imports
from acme import __version__
from . import dask_helpers as dh

callCount = 0
callMax = 1000000

__all__: List["str"] = []


def sizeOf(
        obj: Any,
        varname: str) -> float:
    """
    Estimate memory consumption of Python objects

    Parameters
    ----------
    obj : Python object
        Any valid Python object whose memory footprint is of interest.
    varname : str
        Assigned name of `obj`

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
    funcName = f"<{inspect.currentframe().f_code.co_name}>"     # type: ignore

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


def is_slurm_node() -> bool:
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


def is_esi_node() -> bool:
    """
    Returns `True` if code is running on an ESI cluster node, `False` otherwise
    """

    # Fetch ACME logger and write debug message
    log = logging.getLogger("ACME")
    log.debug("Test if hostname matches the pattern 'esi-sv*'")
    return socket.gethostname().startswith("esi-sv") and os.path.isdir("/cs")


def is_bic_node() -> bool:
    """
    Returns `True` if code is running on a CoBIC cluster node, `False` otherwise
    """

    # Fetch ACME logger and write debug message
    log = logging.getLogger("ACME")
    log.debug("Test if hostname matches the pattern 'bic-sv*'")
    return socket.gethostname().startswith("bic-sv") and os.path.isdir("/mnt/hpc")


def is_x86_node() -> bool:
    """
    Returns `True` if code is running on an x86_64 node, `False` otherwise
    """

    # Fetch ACME logger and write debug message
    log = logging.getLogger("ACME")
    log.debug("Test if host is x86_64 micro-architecture")
    return platform.machine() == "x86_64"


def get_interface(ipaddress : str) -> str:                                      # pragma: no cover
    """
    Returns the name of the first network interface associated to `ipaddress`
    """

    log = logging.getLogger("ACME")
    log.debug("Scanning for NIC associated to %s", ipaddress)
    for iface, iflist in psutil.net_if_addrs().items():
        for psobj in iflist:
            if ipaddress in psobj.address:
                return iface
    err = "IP address %s not associated to any NIC"
    log.error(err, ipaddress)
    raise ValueError(err%(ipaddress))


def get_free_port(                                                              # pragma: no cover
        lo : int,
        hi: int) -> int:
    """
    Returns lowest open port in the given range `lo` to `hi`
    """

    log = logging.getLogger("ACME")
    log.debug("Looking for open ports in the range %d to %d", lo, hi)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = lo
    while port <= hi:
        try:
            sock.bind(("", port))
            sock.close()
            return port
        except OSError:
            port += 1
    err = "Could not find open port in the range %d-%d"
    log.error(err, lo, hi)
    raise IOError(err%(lo,hi))


def _scalar_parser(
        var: Any,
        varname: str = "varname",
        ntype: str = "int_like",
        lims: List = [-np.inf, np.inf]) -> None:
    """
    ACME-specific version of Syncopy's `scalar_parser` (used for cross-compatibility)
    """

    # Get name of calling method/function
    funcName = f"<{inspect.currentframe().f_code.co_name}>"     # type: ignore

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


def user_yesno(                                                         # pragma: no cover
        msg: str,
        default: Optional[str] = None) -> bool:
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


def user_input(                                                         # pragma: no cover
        msg: str,
        valid: Optional[List] = None,
        default: Optional[str] = None,
        timeout: Optional[float] = None) -> str:
    """
    ACME specific version of user-input query
    """

    # Prepare to log any uncaught exceptions
    log = logging.getLogger("ACME")

    # Add trailing whitespace to `msg` if not already present and append
    # default reply (if provided)
    suffix = "" + " " * (not msg.endswith(" "))
    if default is not None:
        default = default.replace("[", "").replace("]","")
        if valid is not None:
            assert default in valid
        suffix = f"[Default: '{default}'] "
    query = msg + suffix

    # Jupyter only supports hard-blocking `input` fields
    if is_jupyter():
        log.debug("Running inside Jupyter notebook, deactivating timeout")
        timeout = None

    # Wait for user I/O
    print(query)
    while True:
        if timeout is None:
            choice = input()
        else:
            stdin, _, _ = select.select([sys.stdin], [], [], timeout)
            if stdin:
                choice = sys.stdin.readline().strip()
            else:
                if default is None:
                    err = f"No response received within the given timeout of {timeout} seconds. "
                    raise TimeoutError(err)
        if default is not None and choice == "":
            return default
        elif valid is not None and choice not in valid:
            print("Please respond with " + " or ".join(valid))
        else:
            return choice


def is_jupyter() -> bool:
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"    # type: ignore
    except NameError:
        return False


def ctrlc_catcher(                                                              # pragma: no cover
        *excargs: Any,
        **exckwargs: Optional[Any]) -> None:
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
            get_ipython()               # type: ignore
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
    if isipy:                                                           # pragma: no cover
        shell.ipyTBshower(shell, exc_tuple=(etype, evalue, etb), **exckwargs)
    else:
        sys.__excepthook__(etype, evalue, etb)

    # Write to all logging locations, manually print traceback to file (stdout
    # printing was handled above)
    log.error("Exception received.")
    memHandler = [h for h in log.handlers if isinstance(h, handlers.MemoryHandler)][0]
    if memHandler.target is not None:                                   # pragma: no cover
        memHandler.acquire()
        with open(memHandler.target.baseFilename, "a", encoding="utf-8") as logfile:    # type: ignore
            logfile.write("".join(traceback.format_exception_only(etype, evalue)))
            logfile.write("".join(traceback.format_tb(etb)))
        memHandler.release()

    return
