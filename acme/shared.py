# -*- coding: utf-8 -*-
#
# Auxiliaries needed across the entire package
#

# Builtin/3rd party package imports
import sys

callCount = 0
callMax = 50000

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
