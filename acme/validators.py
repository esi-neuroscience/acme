#
# Validation utilities for ACME configuration
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import numbers
import numpy as np
import tempfile
import os
import datetime
import inspect
import getpass
import logging
from typing import Optional, Tuple, Union, Callable

# Local imports
from .shared import _scalar_parser, is_esi_node, is_bic_node

# Fetch logger
log = logging.getLogger("ACME")


def validate_pmap(obj) -> None:
    """
    Validate that obj is a ParallelMap instance

    Parameters
    ----------
    obj : object
        Object to validate
    obj_name : str
        Name of the object for error messages

    Raises
    ------
    TypeError
        If obj is not a ParallelMap instance
    """
    try:
        p_class_name = obj.__class__.__name__
    except:
        p_class_name = "notParallelMap"

    if p_class_name != "ParallelMap":
        err = "`pmap` has to be a `ParallelMap` instance, not %s"
        log.error(err, str(type(obj)))
        raise TypeError(err % (str(type(obj))))

    return


def validate_output_flags(config: dict) -> None:
    """
    Validate boolean flag combinations for output writing (write_worker_results, single_file, write_pickle)

    Parameters
    ----------
    config : dict
        Configuration dictionary containing boolean flags

    Raises
    ------
    TypeError
        If boolean flags are not actually booleans
    ValueError
        If incompatible flag combinations are detected
    """

    # Extract Booleans from dictionary and assign defaults
    write_worker_results = config.get("write_worker_results", True)
    write_pickle = config.get("write_pickle", False)
    single_file = config.get("single_file", False)

    # Basal sanity checks
    if not isinstance(write_worker_results, bool):
        err = "`write_worker_results` has to be `True` or `False`, not %s"
        log.error(err, str(write_worker_results))
        raise TypeError(err % (str(write_worker_results)))
    log.debug("Found `write_worker_results = %s`", str(write_worker_results))

    if not isinstance(single_file, bool):
        err = "`single_file` has to be `True` or `False`, not %s"
        log.error(err, str(single_file))
        raise TypeError(err % (str(single_file)))
    log.debug("Found `single_file = %s`", str(single_file))

    if not isinstance(write_pickle, bool):
        err = "`write_pickle` has to be `True` or `False`, not %s"
        log.error(err, str(write_pickle))
        raise TypeError(err % (str(write_pickle)))
    log.debug("Found `write_pickle = %s`", str(write_pickle))

    # Check compatibility of provided optional args
    if not write_worker_results and write_pickle:
        log.warning(
            "Pickling of results only possible if `write_worker_results` is `True`"
        )

    if not write_worker_results and config.get("output_dir"):
        log.warning(
            "Output directory specification has no effect if `write_worker_results` is `False`."
        )

    if write_pickle and config.get("result_shape"):
        log.warning(
            "Pickling of results does not support output array shape specification."
        )

    if not write_worker_results and single_file:
        log.warning(
            "Generating a single output file only possible if `write_worker_results` is `True`."
        )

    if write_pickle and single_file:
        err = "Pickling of results does not support single output file creation."
        log.error(err)
        raise ValueError(err)

    return


def validate_result_shape(
    shape: Optional[tuple], result_dtype: str, n_calls: int, write_worker_results: bool
) -> Tuple[Optional[tuple], Optional[int], Optional[np.dtype]]:
    """
    Validate and normalize result_shape specification

    Parameters
    ----------
    shape : tuple or None
        Result shape specification
    result_dtype : str
        Result data type specification
    n_calls : int
        Number of function calls
    write_worker_results : bool
        Whether worker results are written to disk

    Returns
    -------
    tuple
        (validated_shape, stacking_dim, validated_dtype)
        - validated_shape: Normalized shape tuple or None
        - stacking_dim: Index of stacking dimension or None
        - validated_dtype: NumPy dtype or None

    Raises
    ------
    TypeError
        If shape or dtype have wrong type
    ValueError
        If shape has invalid format or values
    """
    stacking_dim = None
    validated_dtype = None

    if shape is not None:
        # Check validity of output shape/dtype specifications
        if not isinstance(shape, (list, tuple)):
            err = "`result_shape` has to be either `None` or tuple, not %s"
            log.error(err, str(type(shape)))
            raise TypeError(err % (str(type(shape))))

        if not isinstance(result_dtype, str):
            err = "`result_dtype` has to be a string, not %s"
            log.error(err, str(type(shape)))
            raise TypeError(err % (str(type(shape))))

        if sum(spec is None for spec in shape) != 1:
            err = "`result_shape` must contain exactly one `None` entry"
            log.error(err)
            raise ValueError(err)

        r_shape = list(shape)
        stacking_dim = shape.index(None)
        r_shape[stacking_dim] = n_calls

        if not write_worker_results and any(np.isinf(spec) for spec in r_shape):
            err = "using `np.inf` in `result_shape` is only valid if `write_worker_results` is `True`"
            log.error(err)
            raise ValueError(err)

        if r_shape.count(np.inf) > 1:
            err = "cannot use more than one `np.inf` in `result_shape`"
            log.error(err)
            raise ValueError(err)

        if not all(isinstance(spec, numbers.Number) for spec in r_shape):
            err = "`result_shape` must only contain numerical values"
            log.error(err)
            raise ValueError(err)

        if any(
            spec < 0 or int(spec) != spec or np.isnan(spec)
            for spec in r_shape
            if not np.isinf(spec)
        ):
            err = "`result_shape` must only contain non-negative integers"
            log.error(err)
            raise ValueError(err)

        validated_shape = tuple(r_shape)
        msg = "Found `result_shape = %s`. Set stacking dimension to %d"
        log.debug(msg, str(shape), stacking_dim)

        try:
            validated_dtype = np.dtype(result_dtype)
        except Exception as exc:
            err = "`result_dtype` has to be a valid NumPy datatype specification. "
            err += "Original error message below:\n%s"
            log.error(err, str(exc))
            raise TypeError(err % (str(exc)))

        log.debug("Set `result_dtype = %s", validated_dtype)

    else:
        validated_shape = None
        log.debug("Found `result_shape = %s`", str(shape))
        log.debug("Found `result_dtype = %s`", str(result_dtype))

    return validated_shape, stacking_dim, validated_dtype


def validate_logfile(
    logfile_spec,
    write_worker_results: bool,
    func: Optional[Callable] = None,
    out_dir: Optional[str] = None,
    verbose: Optional[bool] = None,
) -> Optional[str]:
    """
    Validate and normalize logfile specification

    Parameters
    ----------
    logfile_spec : None, bool, or str
        Logfile specification
    write_worker_results : bool
        Whether worker results are written to disk
    func : callable, optional
        User function being executed
    out_dir : str, optional
        Output directory path (can be None for auto-detection)

    Returns
    -------
    str or None
        Normalized absolute logfile path or None

    Raises
    ------
    IOError
        If logfile is a directory
    TypeError
        If logfile has invalid type
    """

    # Unless specifically disabled by the user, enable progress-tracking
    # in a log-file if results are auto-generated
    if logfile_spec is None and write_worker_results is True:
        logfile_spec = True

    if func is None:
        func = lambda: None

    # Either parse provided `logfile` or set up an auto-generated file;
    # After this test, `logfile` is either a filename or `None`
    err = "`logfile` has to be `None`, `True`, `False` or a valid file-name, not %s"
    if logfile_spec is None or isinstance(logfile_spec, bool):
        if logfile_spec is True:
            if write_worker_results and out_dir is not None:
                logfile = out_dir
            else:
                # Use current working directory as fallback
                try:
                    # Try to use inspect to get function location
                    logfile = os.path.dirname(os.path.abspath(inspect.getfile(func)))
                except (TypeError, OSError):
                    # Fallback to temp directory
                    logfile = tempfile.gettempdir()

            logfile = os.path.join(
                logfile,
                f"ACME_{func.__name__}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
            )
        else:
            logfile = None
    elif isinstance(logfile_spec, str):
        if os.path.isdir(logfile_spec):
            fault = "a directory"
            log.error(err, fault)
            raise IOError(err % (fault))
        logfile = os.path.abspath(os.path.expanduser(logfile_spec))
    else:
        log.error(err, str(type(logfile_spec)))
        raise TypeError(err % (str(type(logfile_spec))))

    return logfile


def validate_stop_client(stop_client: Union[bool, str]) -> None:
    """
    Validate and normalize stop_client parameter

    Parameters
    ----------
    stop_client : "auto", True, or False
        Stop client specification

    Raises
    ------
    ValueError
        If stop_client is not "auto", True, or False
    TypeError
        If stop_client has incorrect type
    """
    err = "`stop_client` has to be 'auto' or Boolean, not %s"

    if isinstance(stop_client, str):
        if stop_client != "auto":
            log.error(err, stop_client)
            raise ValueError(err % (stop_client))
    else:
        validate_boolean(stop_client, "stop_client")
    return


def validate_n_workers(
    n_workers: Union[int, str], n_calls: int, has_slurm: bool
) -> Union[int, None]:
    """
    Validate and normalize n_workers parameter

    Parameters
    ----------
    n_workers : "auto" or positive integer
        Number of workers specification
    n_calls : positive integer
        Number of task calls
    has_slurm : True or False
        True if SLURM is available

    Returns
    -------
    int or None
        Normalized value

    Raises
    ------
    ValueError
        If n_workers is invalid
    TypeError
        If n_workers has incorrect type
    """
    err = "`n_workers` has to be 'auto' or an integer >= 1, not %s"

    if isinstance(n_workers, str):
        if n_workers != "auto":
            log.error(err, n_workers)
            raise ValueError(err % (n_workers))
        if has_slurm:
            nwrk = n_calls
        else:
            nwrk = None
        log.debug("Changing `n_workers` from `'auto'` to %s", str(nwrk))
    else:
        try:
            _scalar_parser(
                n_workers, varname="n_workers", ntype="int_like", lims=[1, np.inf]
            )
        except Exception as exc:
            log.error(err, n_workers)
            raise exc
        nwrk = n_workers

    return nwrk


def validate_outputdir(
    output_dir: Union[str, None], func: Optional[Callable] = None
) -> str:
    """
    Coming soon...
    """

    # Check validity of output dir specification
    if not isinstance(output_dir, (type(None), str)):
        err = "`output_dir` has to be either `None` or str, not %s"
        log.error(err, str(type(output_dir)))
        raise TypeError(err % (str(type(output_dir))))
    log.debug("Found `output_dir = %s`", str(output_dir))

    # If provided, standardize output dir spec, otherwise use default locations
    if output_dir is not None:
        outDir = os.path.abspath(os.path.expanduser(output_dir))
    else:
        # On the ESI cluster, save results on HPC mount, otherwise use location of `func`
        if is_esi_node() or is_bic_node():
            outDir = f"/mnt/hpc/home/{getpass.getuser()}/"
        else:  # pragma: no cover
            if func is None:
                func = inspect.currentframe()
            outDir = os.path.dirname(os.path.abspath(inspect.getfile(func)))
        outDir = os.path.join(
            outDir, f"ACME_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        )
    log.debug("Using output directory %s", outDir)

    return outDir


def validate_boolean(flag: bool, name: str = "varname") -> None:
    """
    Coming soon...
    """

    # Ensure `flag` is a simple Boolean
    log.debug("Parsing %s", name)
    if not isinstance(flag, bool):
        err = "`%s` has to be `True` or `False`, not %s"
        log.error(err, name, str(type(flag)))
        raise TypeError(err % (name, str(type(flag))))
    return


def validate_partition(partition: str) -> None:
    """
    Coming soon
    """
    if not isinstance(partition, str):
        err = "`partition` has to be 'auto' or a valid SLURM partition name, not %s"
        log.error(err, str(type(partition)))
        raise TypeError(err % (str(type(partition))))
    if partition == "auto":
        if is_esi_node() or is_bic_node():
            wrn = "Automatic SLURM partition selection is experimental"
            log.warning(wrn)
        else:
            err = (
                "Automatic SLURM partition selection currently only available "
                + "on ESI/CoBIC HPC clusters "
            )
            log.error(err)
            raise ValueError(err)
    log.debug("Found SLURM partition selection %s", partition)
    return
