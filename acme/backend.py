# -*- coding: utf-8 -*-
#
# Computational scaffolding for user-interface
#

# Builtin/3rd party package imports
import numbers
import subprocess
import dask.distributed as dd

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
        msg = "{} `n_calls` has to be an integer >= 2, not {}"
        if isinstance(n_calls, numbers.Number):
            if n_calls < 1 or round(n_calls) != n_calls:
                raise ValueError(msg.format(self.msgName, n_calls))
        else:
            raise TypeError(msg.format(self.msgName, n_calls))

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

        # Check, if we're running on a SLURM-enabled node - if yes, retrieve available queues
        out, err = subprocess.Popen("sinfo -h -o %P",
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, shell=True).communicate()
        if len(err) > 0:

            # SLURM is not installed, proceed with `LocalCluster`
            if "sinfo: not found" in err:
                if interactive:
                    msg = "{name:s} SLURM does not seem to be installed on this machine " +\
                        "({host:s}). Do you want to start a local multi-processing " +\
                        "computing client instead? "
                    startLocal = user_yesno(msg.format(name=funcName, host=socket.gethostname()),
                                            default="no")
                else:
                    startLocal = True
                if startLocal:
                    client = Client()
                    successMsg = "{name:s} Local parallel computing client ready. \n" + successMsg
                    print(successMsg.format(name=funcName, dash=client.cluster.dashboard_link))
                    if start_client:
                        return client
                    return client.cluster
                return

            # SLURM is installed, but something's wrong
            msg = "SLURM queuing system from node {node:s}. " +\
                "Original error message below:\n{error:s}"
            raise SPYIOError(msg.format(node=socket.gethostname(), error=err))
        options = out.split()


        if not isinstance(partition, str):
            msg = "{} `partition` has to be 'auto' or a valid SLURM partition name, not {}"
            raise TypeError(msg.format(self.msgName, str(partition)))
        if partition == "auto":
            self.select_queue()

        msg = "{} `n_jobs` has to be 'auto' or an integer >= 2, not {}"
        if isinstance(n_jobs, str):
            if n_jobs != "auto":
                raise ValueError(msg.format(self.msgName, n_jobs))
        elif isinstance(n_jobs, numbers.Number):
            if n_jobs < 1 or round(n_jobs) != n_jobs:
                raise ValueError(msg.format(self.msgName, n_jobs))
        else:
            raise TypeError(msg.format(self.msgName, n_jobs))

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

        self.func(*args, **kwargs)

        pass
