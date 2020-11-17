# -*- coding: utf-8 -*-
#
# User-exposed interface of acme
#

# Builtin/3rd party package imports
import inspect
import numpy as np

# Local imports
from .backend import ACMEdaemon
import acme.shared as acs

__all__ = ["ParallelMap"]


# Main context manager for parallel execution of user-defined functions
class ParallelMap(object):

    msgName = "<ParallelMap>"
    argv = None
    kwargv = None
    func = None
    n_inputs = None
    log = None
    _maxArgSize = 1024

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
        stop_client="auto",
        verbose=None,
        logfile=None,
        **kwargs):
        """
        Coming soon...
        """

        # First and foremost, set up logging system (if wanted)
        self.log = acs.prepare_log(func, logfile=logfile, verbose=verbose)

        # Either guess `n_inputs` or use provided value to duplicate input args
        # and set class attributes `n_inputs`, `argv` and `kwargv`
        self.prepare_input(func, n_inputs, *args, **kwargs)

        # Create an instance of `ACMEdaemon` that does the actual parallel computing work
        self.daemon = ACMEdaemon(self,
                                 n_jobs=n_jobs,
                                 write_worker_results=write_worker_results,
                                 partition=partition,
                                 mem_per_job=mem_per_job,
                                 setup_timeout=setup_timeout,
                                 setup_interactive=setup_interactive,
                                 stop_client=stop_client)

    def prepare_input(self, func, n_inputs, *args, **kwargs):

        # Ensure `func` really is a function and `n_inputs` makes sense
        if not callable(func):
            msg = "{} first input has to be a callable function, not {}"
            raise TypeError(msg.format(self.msgName, str(type(func))))
        msg = "{} `n_inputs` has to be 'auto' or an integer >= 2, not {}"
        if isinstance(n_inputs, str):
            if n_inputs != "auto":
                raise ValueError(msg.format(self.msgName, n_inputs))
            guessInputs = True
        else:
            try:
                acs._scalar_parser(n_inputs, varname="n_inputs", ntype="int_like", lims=[1, np.inf])
            except Exception as exc:
                raise exc
            guessInputs = False

        # Get `func`'s signature to extract its positional/keyword arguments
        funcSignature = inspect.signature(func)
        funcPosArgs = [name for name, value in funcSignature.parameters.items()\
            if value.default is value.empty and value.name != "kwargs"]
        funcKwargs = [name for name, value in funcSignature.parameters.items()\
            if value.default is not value.empty]

        # Compare provided `args`/`kwargs` to actually defined quantities in `func`
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

        # Prepare argument parsing: collect the the length of anything 1D-array-like
        # in `argLens` and check the size of all provided positional and keyword args
        argLens = []
        wrnMsg = "argument size {0:4.2f} MB exceeds recommended limit of {1} MB. " +\
            "Distributing large variables across workers may result in poor performance. "

        # Cycle through positional args
        args = list(args)
        for k, arg in enumerate(args):
            if isinstance(arg, range):
                arg = list(arg)
                args[k] = arg
            acs.callCount = 0
            argsize = acs.sizeOf(arg, "positional arguments")
            if argsize > self._maxArgSize:
                self.log.warning(wrnMsg.format(argsize, self._maxArgSize))
            if isinstance(arg, (list, tuple)):
                argLens.append(len(arg))
            elif isinstance(arg, np.ndarray):
                if len(arg.squeeze().shape) == 1:
                    argLens.append(arg.squeeze().size)

        # More complicated for keyword args: if provided value is 1D-array-like and
        # default value is not, interpret is as worker-arg list, and track its length
        for name, value in kwargs.items():
            defaultValue = funcSignature.parameters[name].default
            if isinstance(value, range):
                value = list(value)
                kwargs[name] = value
            acs.callCount = 0
            valsize = acs.sizeOf(value, "keyword arguments")
            if valsize > self._maxArgSize:
                self.log.warning(wrnMsg.format(valsize, self._maxArgSize))
            if isinstance(value, (list, tuple)):
                if isinstance(defaultValue, (list, tuple)):
                    if len(defaultValue) != len(value):
                        argLens.append(len(value))
                else:
                    argLens.append(len(value))
            elif isinstance(value, np.ndarray) and not isinstance(defaultValue, np.ndarray):
                if len(value.squeeze().shape) == 1:
                    argLens.append(value.squeeze().size)

        # If `n_input` is `"auto"`, make an educated guess as to how many parallel
        # executions of `func` are intended; if input args contained multiple
        # 1D-array-likes, pump the brakes. If `n_input` was explicitly provided,
        # ensure at least one input argument actually contains `n_input` elements
        # for distribution across parallel workers
        if guessInputs:
            if len(set(argLens)) > 1:
                msg = "{} automatic input distribution failed: found {} objects " +\
                    "containing {} to {} elements. Please specify `n_inputs` manually. "
                raise ValueError(msg.format(self.msgName, len(argLens), min(argLens), max(argLens)))
            n_inputs = argLens[0]
        else:
            if n_inputs not in set(argLens):
                msg = "{} No object has required length of {} matching `n_inputs`. "
                raise ValueError(msg.format(self.msgName, n_inputs))
        self.n_inputs = int(n_inputs)

        # Anything that does not contain `n_input` elements is duplicated for
        # distribution across workers, e.g., ``args = [3, [0, 1, 1]]`` then
        # ``self.argv = [[3, 3, 3], [0, 1, 1]]``
        self.argv = list(args)
        for ak, arg in enumerate(args):
            if isinstance(arg, (list, tuple)):
                if len(arg) == self.n_inputs:
                    continue
            elif isinstance(arg, np.ndarray):
                if len(arg.squeeze().shape) == 1 and arg.squeeze().size == self.n_inputs:
                    continue
            self.argv[ak] = [arg] * self.n_inputs

        # Same for keyword arguments with the caveat that default values have to
        # be taken into account (cf. above)
        self.kwargv = dict(kwargs)
        for name, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                if len(value) == self.n_inputs:
                    continue
            elif isinstance(value, np.ndarray) and \
                not isinstance(funcSignature.parameters[name].default, np.ndarray):
                if len(value.squeeze().shape) == 1 and value.squeeze().size == self.n_inputs:
                    continue
            self.kwargv[name] = [value] * self.n_inputs

        # Finally, attach user-provided function to class instance
        self.func = func

    def __enter__(self):
        return self.daemon

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.daemon.cleanup()
