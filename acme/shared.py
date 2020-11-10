# -*- coding: utf-8 -*-
#
# Auxiliaries needed across the entire package
#

# Builtin/3rd party package imports
import sys
import subprocess
import inspect
import numbers
import numpy as np

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
    caller = "<{}>".format(inspect.currentframe().f_code.co_name)

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
                                        lower=lims[0],
                                        upper=lims[1],
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


def user_input(msg, valid, default=None):
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

    # Wait for valid user input and return choice upon receipt
    while True:
        choice = input(msg + suffix)
        if default is not None and choice == "":
            return default
        elif choice in valid:
            return choice
        else:
            print("Please respond with '" + \
                  "or '".join(opt + "' " for opt in valid) + "\n")

