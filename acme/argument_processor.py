#
# Argument processing utilities for ACME
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import numpy as np
import collections
import logging
from typing import List, Dict, Any, Tuple, Optional
from numpy.typing import ArrayLike
from collections.abc import Sized

# Fetch logger
log = logging.getLogger("ACME")


class ArgumentProcessingError(Exception):
    """
    Argument processing failed
    """
    pass


class ArgumentProcessor:
    """
    Handle argument preparation and distribution across workers
    """

    @staticmethod
    def dryrun_setup(
        argv: List[List[Any]],
        kwargv: Dict[str, List[Any]],
        n_calls: int,
        n_runs: Optional[int] = None
    ) -> Tuple[ArrayLike, List[List[Any]], List[Dict[str, Any]]]:
        """
        Pick scheduled jobs at random and extract corresponding args + kwargs

        Parameters
        ----------
        argv : list of list
            Positional arguments for each call
        kwargv : dict of list
            Keyword arguments for each call
        n_calls : int
            Total number of function calls
        n_runs : int, optional
            Number of jobs to pick for dryrun

        Returns
        -------
        tuple
            (indices, args_list, kwargs_list)
        """
        # If not provided, attempt to infer a "sane" default for the number of jobs to pick
        if n_runs is None:
            n_runs = min(
                n_calls, max(5, min(1, int(0.05 * n_calls)))
            )
        log.debug("Picking %d jobs at random", n_runs)

        # Randomly pick `n_runs` jobs and extract positional and keyword args
        dry_run_idx = np.random.choice(n_calls, size=min(n_runs, n_calls), replace=False)
        dry_run_args = []
        dry_run_kwargs = []
        for idx in dry_run_idx:
            dry_run_args.append(
                [arg[idx] if len(arg) > 1 else arg[0] for arg in argv]
            )
            dry_run_kwargs.append(
                [
                    {
                        key: value[idx] if len(value) > 1 else value[0]
                        for key, value in kwargv.items()
                    }
                ][0]
            )
        return dry_run_idx, dry_run_args, dry_run_kwargs

    @staticmethod
    def broadcast_arguments(
        argv: List[List[Any]],
        kwargv: Dict[str, List[Any]],
        n_calls: int,
        client: Any,
        logger: logging.Logger
    ) -> Tuple[List[List[Any]], Dict[str, List[Any]]]:
        """
        Broadcast single-element arguments via scatter()

        Parameters
        ----------
        argv : list of list
            Positional arguments
        kwargv : dict of list
            Keyword arguments
        n_calls : int
            Number of function calls
        client : dask.distributed.Client
            Dask client for scattering
        logger : logging.Logger
            Logger for debugging

        Returns
        -------
        tuple
            (broadcasted_argv, broadcasted_kwargv)
        """
        # Format positional arguments for worker-distribution: broadcast all
        # inputs that are used by all workers and create a list of references
        # to this (single!) future on the cluster for submission
        for ak, arg in enumerate(argv):
            if len(arg) == 1:
                 ft_arg = client.scatter(arg, broadcast=True)
                 logger.debug("Broadcasting single-element pos arg %s to client", str(arg))
                 if isinstance(ft_arg, Sized):
                     ft_arg = ft_arg[0]
                 argv[ak] = [ft_arg] * n_calls

        # Same as above but for keyword-arguments
        for name, value in kwargv.items():
            if len(value) == 1:
                ft_val = client.scatter(value, broadcast=True)[0]
                kwargv[name] = [ft_val] * n_calls
                logger.debug("Broadcasting single-element kwarg `%s` to client", name)

        return argv, kwargv

    @staticmethod
    def format_kwarg_list(
        kwargv: Dict[str, List[Any]],
        n_calls: int
    ) -> List[Dict[str, Any]]:
        """
        Convert parallel keyword args to list of kwarg dictionaries

        Parameters
        ----------
        kwargv : dict of list
            Keyword arguments for each call
        n_calls : int
            Number of function calls

        Returns
        -------
        list of dict
            List of keyword argument dictionaries
        """
        # Re-format keyword arguments to be usable with single-to-many arg submission.
        # Idea: with `n_calls = 3` and ``kwargv = {'a': [5, 5, 5], 'b': [6, 6, 6]}``
        # then ``kwarg_list = [{'a': 5, 'b': 6}, {'a': 5, 'b': 6}, {'a': 5, 'b': 6}]``
        kwarg_list = []
        kwarg_keys = kwargv.keys()
        kwarg_vals = list(kwargv.values())
        for nc in range(n_calls):
            kw_dict = {}
            for kc, key in enumerate(kwarg_keys):
                # Handle single-element lists by repeating the single value
                if len(kwarg_vals[kc]) == 1:
                    kw_dict[key] = kwarg_vals[kc][0]
                else:
                    kw_dict[key] = kwarg_vals[kc][nc]
            kwarg_list.append(kw_dict)

        return kwarg_list