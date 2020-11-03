# -*- coding: utf-8 -*-
#
# Computational scaffolding for user-interface
#

# Builtin/3rd party package imports
import time
import getpass
import datetime
import inspect
import warnings
import os
import h5py
import dask.distributed as dd
import dask.bag as db
import numpy as np

# Local imports
from .dask_helpers import esi_cluster_setup
import acme.shared as acs


# Main context manager for parallel execution of user-defined functions
class ACMEdaemon(object):

    msgName = "<ACMEdaemon>"

    def __init__(
        self,
        pmap=None,
        func=None,
        argv=None,
        kwargv=None,
        n_calls=None,
        n_jobs="auto",
        write_worker_results=True,
        partition="auto",
        mem_per_job="auto",
        setup_timeout=180,
        setup_interactive=True):

        # The only error checking happening in `__init__`
        if pmap is not None:
            if pmap.__class__.__name__ != "ParallelMap":
                msg = "{} `pmap` has to be a `ParallelMap` instance, not {}"
                raise TypeError(msg.format(self.msgName, str(pmap)))

        # Input pre-processed by a `ParallelMap` object takes precedence over keyword args
        self.initialize(getattr(pmap, "func", func),
                        getattr(pmap, "argv", argv),
                        getattr(pmap, "kwargv", kwargv),
                        getattr(pmap, "n_inputs", n_calls))

        # Set up output handler
        self.prepare_output(write_worker_results)

        # Either use existing dask client or start a fresh instance
        self.prepare_client(n_jobs=n_jobs,
                            partition=partition,
                            mem_per_job=mem_per_job,
                            setup_timeout=setup_timeout,
                            setup_interactive=setup_interactive)

        # # >>>>>>>>>>>>>> DICT CONVERSION
        # tt = [[{key: val} for val in testDict[key]] for key in testDict.keys()]
        # bag <- zip(*tt)
        # testDict = {'a':[2,2,2], 'b':[5,6,7]}
        # tt = [[{'a': 2}, {'a': 2}, {'a': 2}], [{'b': 5}, {'b': 6}, {'b': 7}]]
        # list(zip(*tt)) = [({'a': 2}, {'b': 5}), ({'a': 2}, {'b': 6}), ({'a': 2}, {'b': 7})]

    def initialize(self, func, argv, kwargv, n_calls):

        # Ensure `func` is callable
        if not callable(func):
            msg = "{} first input has to be a callable function, not {}"
            raise TypeError(msg.format(self.msgName, str(type(func))))

        # Next, vet `n_calls` which is needed to validate `argv` and `kwargv`
        try:
            acs._scalar_parser(n_calls, varname="n_calls", ntype="int_like", lims=[2, np.inf])
        except Exception as exc:
            raise exc

        # Ensure all elements of `argv` are list-like with length `n_calls`
        msg = "{} `argv` has to be a list with list-like elements of length {}"
        if not isinstance(argv, (list, tuple)):
            raise TypeError(msg.format(self.msgName, n_calls))
        try:
            validArgv = all(len(arg) == n_calls for arg in argv)
        except TypeError:
            raise TypeError(msg.format(self.msgName, n_calls))
        if not validArgv:
            raise ValueError(msg.format(self.msgName, n_calls))

        # Ensure all keys of `kwargv` have length `n_calls`
        msg = "{} `kwargv` has to be a dictionary with list-like elements of length {}"
        try:
            validKwargv = all(len(value) == n_calls for value in kwargv.values())
        except TypeError:
            raise TypeError(msg.format(self.msgName, n_calls))
        if not validKwargv:
            raise ValueError(msg.format(self.msgName, n_calls))

        # Basal sanity checks have passed, keep the provided input signature
        self.func = func
        self.argv = argv
        self.kwargv = kwargv
        self.n_calls = n_calls

    def prepare_output(self, write_worker_results):

        if not isinstance(write_worker_results, bool):
            msg = "{} `write_worker_results` has to be `True` or `False`, not {}"
            raise TypeError(msg.format(self.msgName, str(write_worker_results)))

        if write_worker_results:
            outDir = "/mnt/hpx/home/{usr:s}/ACME_{date:s}"
            self.outDir = outDir.format(usr=getpass.getuser(),
                                        date=datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
            try:
                os.makedirs(self.outDir)
            except Exception as exc:
                msg = "{} automatic creation of output folder {} failed. Original error message below:\n{}"
                raise OSError(msg.format(self.msgName, self.outDir, str(exc)))
            self.acmeFunc = self.func_wrapper
        else:
            if self.kwargv.get("workerID") is None:
                msg = "{} `write_worker_results` is `False` and `workerID` is not a keyword argument of {}." +\
                    "Results will be collected in memory by caller - this might be slow and can lead " +\
                    "to excessive memory consumption. "
                warnings.showwarning(msg.format(self.msgName, self.func.__name__),
                                     RuntimeWarning, __file__, inspect.currentframe().f_lineno)
            self.acmeFunc = self.func

    def prepare_client(
        self,
        n_jobs="auto",
        partition="auto",
        mem_per_job="auto",
        setup_timeout=180,
        setup_interactive=True):

        # Check if a dask client is already running
        try:
            self.client = dd.get_client()
            return
        except ValueError:
            pass

        # If things are running locally, simply fire up a dask-distributed client,
        # otherwise go through the motions of preparing a full cluster job swarm
        if not acs.is_slurm_node:
            self.client = esi_cluster_setup(interactive=False)

        else:

            # If `partition` is "auto", use `select_queue` to heuristically determine
            # the "best" SLURM queue
            if not isinstance(partition, str):
                msg = "{} `partition` has to be 'auto' or a valid SLURM partition name, not {}"
                raise TypeError(msg.format(self.msgName, str(partition)))
            if partition == "auto":
                msg = "{} WARNING: Automatic SLURM queueing selection not implemented yet. " +\
                    "Falling back on default '8GBS' partition. "
                print(msg.format(self.msgName))
                partition = "8GBS"
                # self.select_queue()

            # Either use `n_jobs = n_calls` (default) or parse provided value
            if isinstance(n_jobs, str):
                if n_jobs != "auto":
                    raise ValueError(msg.format(self.msgName, n_jobs))
                self.n_jobs = self.n_calls
            else:
                try:
                    acs._scalar_parser(n_jobs, varname="n_jobs", ntype="int_like", lims=[2, np.inf])
                except Exception as exc:
                    raise exc

            # All set, remaining input processing is done by `esi_cluster_setup`
            self.client = esi_cluster_setup(partition=partition, n_jobs=n_jobs,
                                            mem_per_job=mem_per_job, timeout=setup_timeout,
                                            interactive=setup_interactive, start_client=True)


    def select_queue(self):

        # FIXME: Very much WIP
        # Scratchpad, nothing final yet
        nSamples = min(self.n_calls, max(5, min(1, int(0.05*self.n_calls))))
        dryRunInputs = np.random.choice(self.n_calls, size=nSamples, replace=False).tolist()

        dryRun0 = dryRunInputs.pop()
        args = [arg[dryRun0] for arg in self.argv]
        kwargs = [{key:value[dryRun0] for key, value in self.kwargv.items()}][0]

        # use multi-processing module to launch `func` in background; terminate after
        # 60sec, get memory consumption, do this for all `dryRunInputs` -> pick
        # "shortest" queue w/appropriate memory (e.g., 16GBS, not 16GBXL)
        tic = time.perf_counter()
        self.func(*args, **kwargs)
        toc = time.perf_counter()

    def compute(self):

        # argBag = db.from_sequence(self.argv, npartitions=self.n_calls)
        argBag = [[arg[k] for arg in self..argv] for k in range(self.n_calls)]
        kwargBag = db.from_sequence(zip(*[[{key: val} for val in self.kwargv[key]] for key in kwargv.keys()]))

        # FIXME: check client for alive workers...
        results = mainBag.map(self.computeFunction, *bags, **self.cfg)

    @staticmethod
    def func_wrapper(func, outDir, *args, **kwargs):
        result = func(*args, **kwargs)
        if outDir is not None:
            fname = "{}_{}.h5".format(func.__name__, kwargs["workerID"])
            with h5py.File(os.path.join(outDir, fname), "w") as h5f:
                h5f.create_dataset("result", data=result)
