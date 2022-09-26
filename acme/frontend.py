# -*- coding: utf-8 -*-
#
# User-exposed interface of acme
#

# Builtin/3rd party package imports
import inspect
import sys
import numpy as np
import dask.array as da

# Local imports
from .backend import ACMEdaemon
from . import shared as acs
isSpyModule = False
if "syncopy" in sys.modules:
    isSpyModule = True

__all__ = ["ParallelMap"]


# Main context manager for parallel execution of user-defined functions
class ParallelMap(object):

    msgName = "{pre:s}<{pkg:s}ParallelMap>".format(pre="Syncopy " if isSpyModule else "",
                                                   pkg="ACME: " if isSpyModule else "")
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
        output_dir=None,
        result_shape=None,
        result_dtype="float",
        single_file=False,
        write_pickle=False,
        partition="auto",
        n_jobs="auto",
        mem_per_job="auto",
        setup_timeout=60,
        setup_interactive=True,
        stop_client="auto",
        verbose=None,
        dryrun=False,
        logfile=None,
        **kwargs):
        """
        Context manager that executes user-defined functions in parallel

        Parameters
        ----------
        func : callable
            User-defined function to be executed concurrently. Input arguments
            and return values should be "simple" (i.e., regular Python objects or
            NumPy arrays). See Notes for more information and Examples for
            details.
        args : arguments
            Positional arguments of `func`. Should be regular Python objects
            (lists, tuples, scalars, strings etc.) or NumPy arrays. See Notes
            for more information and Examples for details.
        kwargs : keyword arguments
            Keyword arguments of `func` (if any). Should be regular Python objects
            (lists, tuples, scalars, strings etc.) or NumPy arrays. See Notes
            for more information and Examples for details.
        n_inputs : int or "auto"
            Number of times `func` is supposed to be called in parallel. Usually,
            `n_inputs` does not have to be provided explicitly. If `n_inputs` is
            `"auto"` (default) this quantity is inferred from provided `args` and
            `kwargs`. This estimation may fail due to ambiguous input arguments
            (e.g., `args` and/or `kwargs` contain lists of differing lengths)
            triggering a `ValueError`. Only then is it required to set `n_input`
            manually. See Examples for details.
        write_worker_results : bool
            If `True`, the return value(s) of `func` is/are saved on disk (one
            HDF5 file per parallel worker). If `False`, the output of all parallel calls
            of `func` is collected in memory. See Examples and Notes for details.
        write_pickle : bool
            If `True`, the return value(s) of `func` is/are pickled to disk (one
            `'.pickle'`-file per parallel worker). Only effective if `write_worker_results`
            is `True`.
        partition : str
            Name of SLURM partition to use. If `"auto"` (default), the memory footprint
            of `func` is estimated using dry-run stubs based on randomly sampling
            provided `args` and `kwargs`. Estimated memory usage dictates queue
            auto-selection under the assumption of short run-times (**currently only
            supported on the ESI HPC cluster**). For instance, on a predicted memory
            footprint of 6 GB causes the `"8GBXS"` partition to be selected (minimal
            but sufficient memory and shortest runtime).
            To override auto-selection, provide name of SLURM queue explicitly. See, e.g.,
            :func:`~acme.esi_cluster_setup` for details.
        n_jobs : int or "auto"
            Number of SLURM jobs (=workers) to spawn. If `"auto"` (default), then
            ``n_jobs = n_inputs``, i.e., every SLURM worker performs a single
            call of `func`.
            If `n_inputs` is large and executing `func` is fast, setting
            ``n_jobs = int(n_inputs / 2)`` might be beneficial. See Notes for details.
        mem_per_job : str
            Memory booking for each SLURM worker. If `"auto"` (default), the standard
            value is inferred from the used partition (if possible). See, e.g.,
            :func:`~acme.esi_cluster_setup` for details.
        setup_timeout : int
            Timeout period (in seconds) for SLURM workers to come online. See, e.g.,
            :func:`~acme.esi_cluster_setup` for details.
        setup_interactive : bool
            If `True` (default), user input is queried in case not enough SLURM
            workers could be started within `setup_timeout` seconds. If no input
            is provided, the current number of spawned workers is used (even if
            smaller than the amount requested by `n_jobs`). If `False`, no user
            choice is requested.
        stop_client : bool or "auto"
            If `"auto"` (default), automatically started distributed computing clients
            are shut down at the end of computation, while user-provided clients
            are left untouched. If `False`, automatically started clients are
            left running after completion, user-provided clients are left untouched.
            If `True`, auto-generated clients *and* user-provided clients are
            shut down at the end of the computation.
        verbose : None or bool
            If `None` (default), general run-time information as well as warnings
            and errors are shown. If `True`, additionally debug information is
            shown. If `False`, only warnings and errors are propagated.
        dryrun : bool
            If `True` the user-provided function `func` is executed once using
            one of the input argument tuples prepared for the parallel workers (picked
            at random). If `setup_interactive` is `True`, a prompt asks if the
            actual parallel execution of `func` is supposed to be launched after the
            dry-run. The `dryrun` keyword is intended to to estimate memory consumption
            as well as runtime of worker jobs prior to the actual concurrent
            computation.
        logfile : None or bool or str
            If `None` (default) or `False`, all run-time information as well as errors and
            warnings are printed to the command line only. If `True`, an auto-generated
            log-file is set up that records run-time progress. Alternatively, the
            name of a custom log-file can be provided (must not exist). The verbosity
            of recorded information can be controlled via setting `verbose`.

        Returns
        -------
        results : list
            If `write_worker_results` is `True`, `results` is a list of HDF5 file-names
            containing computed results. If `write_worker_results` is `False`,
            results is a list comprising the actual return values of `func`.

        Examples
        --------
        Call `f` with four different values of `x` while setting `y` to 4:

        .. code-block:: python

            from acme import ParallelMap

            with ParallelMap(f, [2, 4, 6, 8], 4) as pmap:
                results = pmap.compute()

        Collect results in memory (can be slow due to network traffic and may cause
        memory overflow in parent caller):

        .. code-block:: python

            with ParallelMap(f, [2, 4, 6, 8], 4, write_worker_results=False) as pmap:
                results = pmap.compute()

        Manually set `n_inputs` in case of argument distribution cannot be determined
        automatically:

        .. code-block:: python

            with ParallelMap(f, [2, 4, 6, 8], y, n_inputs=4, write_worker_results=False) as pmap:
                results = pmap.compute()

        More examples and tutorials are available in the
        `ACME online documentation <https://esi-acme.readthedocs.io>`_.

        See also
        --------
        esi_cluster_setup : spawn custom SLURM worker clients on the ESI HPC cluster
        local_cluster_setup : start a local Dask multi-processing cluster on the host machine
        ACMEdaemon : Manager class performing the actual concurrent processing
        """

        # First and foremost, set up logging system (unless logger is already present)
        self.log = acs.prepare_log(func, caller=self.msgName, logfile=logfile, verbose=verbose)

        # Either guess `n_inputs` or use provided value to duplicate input args
        # and set class attributes `n_inputs`, `argv` and `kwargv`
        self.prepare_input(func, n_inputs, *args, **kwargs)

        # Create an instance of `ACMEdaemon` that does the actual parallel computing work
        self.daemon = ACMEdaemon(self,
                                 n_jobs=n_jobs,
                                 write_worker_results=write_worker_results,
                                 output_dir=output_dir,
                                 result_shape=result_shape,
                                 result_dtype=result_dtype,
                                 single_file=single_file,
                                 write_pickle=write_pickle,
                                 dryrun=dryrun,
                                 partition=partition,
                                 mem_per_job=mem_per_job,
                                 setup_timeout=setup_timeout,
                                 setup_interactive=setup_interactive,
                                 stop_client=stop_client)

    def prepare_input(self, func, n_inputs, *args, **kwargs):
        """
        User input parser

        Ensure `func` can actually process provided arguments. If `n_inputs` was
        not set, attempt to infer the number of required concurrent function
        calls from `args` and `kwargs`. In addition, ensure the size of each
        argument is "reasonable" for propagation across multiple workers.
        """

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

        # Account for positional args that were specified by name (keyword-like)
        args = list(args)
        posArgNames = []
        for name, value in kwargs.items():
            if name in funcPosArgs:
                args.insert(funcPosArgs.index(name), value)
                posArgNames.append(name)
        for name in posArgNames:
            kwargs.pop(name)

        # If "taskID" is a keyword arg, include/overwrite it in `kwargs` - the rest
        # is done by `ACMEdaemon`
        if "taskID" in funcKwargs:
            kwargs["taskID"] = None

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

        # Anything that does not contain `n_input` elements is converted to a one-element list
        wrnMsg = "Found a single callable object in positional arguments. " +\
            "It will be executed just once and shared by all workers"
        self.argv = list(args)
        for ak, arg in enumerate(args):
            if isinstance(arg, (list, tuple)):
                if len(arg) == self.n_inputs:
                    continue
            elif isinstance(arg, np.ndarray):
                if len(arg.squeeze().shape) == 1 and arg.squeeze().size == self.n_inputs:
                    continue
            elif callable(arg):
                self.log.warning(wrnMsg)
            self.argv[ak] = [arg]

        # Same for keyword arguments with the caveat that default values have to
        # be taken into account (cf. above)
        wrnMsg = "Found a single callable object in keyword arguments: {}. " +\
            "It will be executed just once and shared by all workers"
        self.kwargv = dict(kwargs)
        for name, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                if len(value) == self.n_inputs:
                    continue
            elif isinstance(value, np.ndarray) and \
                not isinstance(funcSignature.parameters[name].default, np.ndarray):
                if len(value.squeeze().shape) == 1 and value.squeeze().size == self.n_inputs:
                    continue
            elif callable(value):
                self.log.warning(wrnMsg.format(name))
            self.kwargv[name] = [value]

        # Finally, attach user-provided function to class instance
        self.func = func

    def compute(self):
        """
        Shortcut to launch parallel computation via `ACMEdaemon`
        """
        if hasattr(self, "daemon"):
            self.daemon.compute()

    def cleanup(self):
        """
        Shortcut to corresponding cleanup-routine provided by `ACMEdaemon`
        """
        if hasattr(self, "daemon"):
            self.daemon.cleanup

    def __enter__(self):
        """
        If `ParallelMap` is used as context manager, launch `ACMEdaemon`
        """
        return self.daemon

    def __exit__(self, exception_type, exception_value, exception_traceback):
        """
        If `ParallelMap` is used as context manager, close any ad-hoc computing
        clients created by `ACMEdaemon`
        """
        self.daemon.cleanup()
