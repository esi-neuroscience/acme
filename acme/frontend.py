# -*- coding: utf-8 -*-
# 
# User-exposed interface of acme
# 

# Builtin/3rd party package imports
import numbers
import inspect
import numpy as np

# Local imports
from .backend import ACMEdaemon


# Main context manager for parallel execution of user-defined functions
class ParallelMap(object):
    
    msgName = "<ParallelMap>"
    argv = []
    ArgV = []
    kwargv = {}
    KwargV = {}
    
    def __init__(
        self, 
        func, 
        *args, 
        n_inputs="auto",
        write_worker_results=True, 
        partition="auto", 
        n_jobs="auto", 
        mem_per_job="auto",
        setup_timeout=180, 
        setup_interactive=True, 
        start_client=True,        
        **kwargs):
        """
        Coming soon...
        """
        
        self.prep_input(func, n_inputs, *args, **kwargs)
        
        
    def prep_input(self, func, n_inputs, *args, **kwargs):
        
        if not callable(func):
            msg = "{} first input has to be a callable function, not {}"
            raise TypeError(msg.format(self.msgName, str(type(func))))
        
        msg = "{} `n_inputs` has to be 'auto' or an integer number >= 2, not {}"
        if isinstance(n_inputs, str):
            if n_inputs != "auto":
                raise ValueError(msg.format(self.msgName, n_inputs))
            guessInputs = True
        elif isinstance(n_inputs, numbers.Number):
            if n_inputs < 1 or round(n_inputs) != n_inputs:
                raise ValueError(msg.format(self.msgName, n_inputs))
            guessInputs = False
        else:
            raise TypeError(msg.format(self.msgName, n_inputs))
        
        funcSignature = inspect.signature(func)
        funcPosArgs = [name for name, value in funcSignature.parameters.items()\
            if value.default == value.empty]
        funcKwargs = [name for name, value in funcSignature.parameters.items()\
            if value.default != value.empty]
        
        if len(args) != len(funcPosArgs):
            msg = "{} {} expects {} positional arguments ({}), found {}"
            validArgs = "'" + "'".join(arg + "', " for arg in funcPosArgs)[:-2]
            raise ValueError(msg.format(self.msgName, 
                                        func.__name__, 
                                        len(funcPosArgs),
                                        validArgs, 
                                        len(args)))
        if len(kwargs) > len(funcKwargs):
            msg = "{} {} accepts at maximum {} keyword arguments ({}), found {}"
            validArgs = "'" + "'".join(arg + "', " for arg in funcKwargs)[:-2]
            raise ValueError(msg.format(self.msgName, 
                                        func.__name__, 
                                        len(funcKwargs),
                                        validArgs, 
                                        len(kwargs)))

        if guessInputs:
            argLens = []
            for arg in args:
                if isinstance(arg, (list, tuple)):
                    argLens.append(len(arg))
                elif isinstance(arg, np.ndarray):
                    if len(arg.squeeze().shape) == 1:
                        argLens.append(arg.squeeze().size)
                        
            for name, value in kwargs.items():
                defaultValue = funcSignature.parameters[name].default 
                if isinstance(value, (list, tuple)):
                    if isinstance(defaultValue, (list, tuple)):
                        if len(defaultValue) != len(value):
                            argLens.append(len(value))
                elif isinstance(value, np.ndarray) and not isinstance(defaultValue, np.ndarray):
                    if len(value.squeeze().shape) == 1:
                        argLens.append(value.squeeze().size)
                        
            n_inputs = len(set(argLens))
            if n_inputs > 1:
                raise ValueError("Cannot guess, please specify n_input")

        for ak, arg in enumerate(args):
            if isinstance(arg, (list, tuple)):
                if len(arg) == n_inputs:
                    continue
            elif isinstance(arg, np.ndarray):
                if len(arg.squeeze().shape) == 1 and arg.squeeze().size == n_inputs:
                    continue
            self.argv[ak] = [arg] * n_inputs
            
        for name, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                if len(value) == n_inputs:
                    continue
            elif isinstance(value, np.ndarray) and \
                not isinstance(funcSignature.parameters[name].default, np.ndarray):
                if len(value.squeeze().shape) == 1 and arg.squeeze().size == n_inputs:
                    continue
            self.kwargv[name] = [value] * n_inputs
            
            
    def __enter__(self):
        return self.file_obj
    
    def __exit__(self, type, value, traceback):
        self.file_obj.close()
        
