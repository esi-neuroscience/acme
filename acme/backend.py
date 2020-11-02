# -*- coding: utf-8 -*-
#
# Computational scaffolding for user-interface
#

# Builtin/3rd party package imports
import time
import dask.distributed as dd

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


        # Check if a dask client is already running
        try:
            self.client = dd.get_client()
        except ValueError:
            self.prepare_client()


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

    def allocate_output(self, write_worker_results):

        if not isinstance(write_worker_results, bool):
            msg = "{} `write_worker_results` has to be `True` or `False`, not {}"
            raise TypeError(msg.format(self.msgName, str(write_worker_results)))

        if write_worker_results:
            acmeFunc = self.wrappedFunc
        else:
            acmeFunc = self.func

    # def wrappedFunc()


    def prepare_client(self, ncalls, n_jobs,
        write_worker_results=True,
        partition="auto",
        mem_per_job="auto",
        setup_timeout=180,
        setup_interactive=True):

        # If things are running locally, simply fire up a dask-distributed client,
        # otherwise go through the motions of preparing a full cluster job swarm
        if not acs.is_slurm_node:
            self.client = esi_cluster_setup(interactive=False)
        else:

            # If `partition` is "auto", use `select_queue` to heuristically determine
            # the "best" SLURM queue for the job at hand
            if not isinstance(partition, str):
                msg = "{} `partition` has to be 'auto' or a valid SLURM partition name, not {}"
                raise TypeError(msg.format(self.msgName, str(partition)))
            if partition == "auto":
                self.select_queue()

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

        cleanup = False
        if parallel is None or parallel is True:
            if spy.__dask__:
                try:
                    dd.get_client()
                    parallel = True
                except ValueError:
                    if parallel is True:
                        objList = []
                        argList = list(args)
                        nTrials = 0
                        for arg in args:
                            if hasattr(arg, "trials"):
                                objList.append(arg)
                                nTrials = max(nTrials, len(arg.trials))
                                argList.remove(arg)
                        nObs = len(objList)
                        msg = "Syncopy <{fname:s}> Launching parallel computing client " +\
                            "to process {no:d} objects..."
                        print(msg.format(fname=func.__name__, no=nObs))
                        client = esi_cluster_setup(n_jobs=nTrials, interactive=False)
                        cleanup = True
                        if len(objList) > 1:
                            for obj in objList:
                                results.append(func(obj, *argList, **kwargs))
                    else:
                        parallel = False
            else:
                wrng = \
                "dask seems not to be installed on this system. " +\
                "Parallel processing capabilities cannot be used. "
                SPYWarning(wrng)
                parallel = False

        # Add/update `parallel` to/in keyword args
        kwargs["parallel"] = parallel

        if len(results) == 0:
            results = func(*args, **kwargs)
        if cleanup:
            cluster_cleanup(client=client)

    def select_queue(self):

        nSamples = min(self.n_calls, max(5, min(1, int(0.05*self.n_calls))))
        dryRunInputs = np.random.choice(self.n_calls, size=nSamples, replace=False).tolist()

        dryRun0 = dryRunInputs.pop()
        args = [arg[dryRun0] for arg in self.argv]
        kwargs = [{key:value[dryRun0] for key, value in self.kwargv.items()}][0]

        tic = time.perf_counter()
        self.func(*args, **kwargs)
        toc = time.perf_counter()

        pass
