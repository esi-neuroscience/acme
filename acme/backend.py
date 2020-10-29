# -*- coding: utf-8 -*-
# 
# Computational scaffolding for user-interface
# 

# Builtin/3rd party package imports
import numbers
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

        # Either use input processed by `ParallelMap` or provided keyword args
        self.initialize(getattr(pmap, "func", func), 
                        getattr(pmap, "argv", argv), 
                        getattr(pmap, "kwargv", kwargv),
                        getattr(pmap, "n_inputs", n_calls))

        # Check if a dask client is already running
        try:
            self.client = dd.get_client()
            start_client = False
        except ValueError:
            self.prepare_client()
            start_client = True
            
          
    def initialize(self, func, argv, kwargv, n_calls):

        if not callable(func):
            msg = "{} first input has to be a callable function, not {}"
            raise TypeError(msg.format(self.msgName, str(type(func))))
        
        msg = "{} `n_calls` has to be an integer >= 2, not {}"
        if isinstance(n_calls, numbers.Number):
            if n_calls < 1 or round(n_calls) != n_calls:
                raise ValueError(msg.format(self.msgName, n_calls))
        else:
            raise TypeError(msg.format(self.msgName, n_calls))
        
        # ensure all elements of argv have len n_calls
        # ensure all keys of kwargv have len n_calls
        
        self.func = func 
        self.argv = argv
        self.kwargv = kwargv
        self.n_calls = n_calls
            
    def choose_queue():
        pass

    def prepare_client(self, ncalls, n_jobs, 
        write_worker_results=True, 
        partition="auto", 
        mem_per_job="auto",
        setup_timeout=180, 
        setup_interactive=True):

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
