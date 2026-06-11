#
# Argument processing utilities for ACME
#
# Copyright © 2026 Ernst Strüngmann Institute (ESI) of the Max Planck Society
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


class ArgumentProcessor:
    """
    Handle argument preparation and distribution across workers
    """

    def __init__(
        self, argv: List[List[Any]], kwargv: Dict[str, List[Any]], n_calls: int
    ):
        """
        Initialize argument processing class

        Parameters
        ----------
        argv : list of list
            Positional arguments for each call
        kwargv : dict of list
            Keyword arguments for each call
        n_calls : int
            Total number of function calls
        """
        self.argv = argv
        self.kwargv = kwargv
        self.n_calls = n_calls

    def dryrun_setup(
        self,
        n_runs: Optional[int] = None,
    ) -> Tuple[ArrayLike, List[List[Any]], List[Dict[str, Any]]]:
        """
        Pick scheduled jobs at random and extract corresponding args + kwargs

        Parameters
        ----------
        n_runs : int, optional
            Number of jobs to pick for dryrun

        Returns
        -------
        tuple
            (indices, args_list, kwargs_list)
        """
        # If not provided, attempt to infer a "sane" default for the number of jobs to pick
        if n_runs is None:
            n_runs = min(self.n_calls, max(5, min(1, int(0.05 * self.n_calls))))
        log.debug("Picking %d jobs at random", n_runs)

        # Randomly pick `n_runs` jobs and extract positional and keyword args
        dry_run_idx = np.random.choice(
            self.n_calls, size=min(n_runs, self.n_calls), replace=False
        )
        dry_run_args = []
        dry_run_kwargs = []
        for idx in dry_run_idx:
            dry_run_args.append(
                [arg[idx] if len(arg) > 1 else arg[0] for arg in self.argv]
            )
            dry_run_kwargs.append(
                [
                    {
                        key: value[idx] if len(value) > 1 else value[0]
                        for key, value in self.kwargv.items()
                    }
                ][0]
            )
        return dry_run_idx, dry_run_args, dry_run_kwargs

    def broadcast_arguments(
        self,
        client: Any,
    ) -> Tuple[List[List[Any]], Dict[str, List[Any]]]:
        """
        Broadcast single-element arguments via scatter()

        Parameters
        ----------
        client : dask.distributed.Client
            Dask client for scattering

        Returns
        -------
        tuple
            (broadcasted_argv, broadcasted_kwargv)
        """
        # Format positional arguments for worker-distribution: broadcast all
        # inputs that are used by all workers and create a list of references
        # to this (single!) future on the cluster for submission
        for ak, arg in enumerate(self.argv):
            if len(arg) == 1:
                ft_arg = client.scatter(arg, broadcast=True)
                log.debug("Broadcasting single-element pos arg %s to client", str(arg))
                if isinstance(ft_arg, Sized):
                    ft_arg = ft_arg[0]
                self.argv[ak] = [ft_arg] * self.n_calls

        # Same as above but for keyword-arguments
        for name, value in self.kwargv.items():
            if len(value) == 1:
                ft_val = client.scatter(value, broadcast=True)[0]
                self.kwargv[name] = [ft_val] * self.n_calls
                log.debug("Broadcasting single-element kwarg `%s` to client", name)

        return self.argv, self.kwargv

    def format_kwarg_list(self) -> List[Dict[str, Any]]:
        """
        Convert parallel keyword args to list of kwarg dictionaries

        Returns
        -------
        list of dict
            List of keyword argument dictionaries
        """
        # Re-format keyword arguments to be usable with single-to-many arg submission.
        # Idea: with `self.n_calls = 3` and ``kwargv = {'a': [5, 5, 5], 'b': [6, 6, 6]}``
        # then ``kwarg_list = [{'a': 5, 'b': 6}, {'a': 5, 'b': 6}, {'a': 5, 'b': 6}]``
        kwarg_list = []
        kwarg_keys = self.kwargv.keys()
        kwarg_vals = list(self.kwargv.values())
        for nc in range(self.n_calls):
            kw_dict = {}
            for kc, key in enumerate(kwarg_keys):
                # Handle single-element lists by repeating the single value
                if len(kwarg_vals[kc]) == 1:
                    kw_dict[key] = kwarg_vals[kc][0]
                else:
                    kw_dict[key] = kwarg_vals[kc][nc]
            kwarg_list.append(kw_dict)

        return kwarg_list
