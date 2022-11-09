# -*- coding: utf-8 -*-
#
# User-exposed interface of acme
#

# Builtin/3rd party package imports
import inspect
import sys
import numpy as np

# Local imports
from acme import __deprecated__, __deprecation_wrng__
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
        n_workers="auto",
        mem_per_worker="auto",
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
            NumPy arrays). See Examples and [1]_ for more information.
        args : arguments
            Positional arguments of `func`. Should be regular Python objects
            (lists, tuples, scalars, strings etc.) or NumPy arrays. See
            Examples and [1]_ for more information.
        kwargs : keyword arguments
            Keyword arguments of `func` (if any). Should be regular Python objects
            (lists, tuples, scalars, strings etc.) or NumPy arrays. See Examples
            and [1]_ for more information.
        n_inputs : int or "auto"
            Number of times `func` is supposed to be called in parallel. Usually,
            `n_inputs` does not have to be provided explicitly. If `n_inputs` is
            `"auto"` (default) this quantity is inferred from provided `args` and
            `kwargs`. This estimation may fail due to ambiguous input arguments
            (e.g., `args` and/or `kwargs` contain lists of differing lengths)
            triggering a `ValueError`. Only then is it required to set `n_input`
            manually. See Examples and [1]_ for more information.
        write_worker_results : bool
            If `True`, the return value(s) of `func` is/are saved on disk.
            If `False`, the output of all parallel calls of `func` is collected
            in memory. See Examples as well as [1]_ and [2]_ for more information.
        output_dir : str or None
            Only relevant if `write_worker_results` is `True`. If `output_dir` is `None`
            (default) and `write_worker_results` is `True`, all files auto-generated
            by `ParallelMap` are stored in a directory `'ACME_YYYYMMDD-hhmmss-ffffff'`
            (encoding the current time as YearMonthDay-HourMinuteSecond-Microsecond).
            The path to a custom output directory can be specified via providing
            `output_dir`. See Examples and [1]_ for more information.
        result_shape : tuple or None
            Only relevant if `write_pickle` is `False`. If provided, return
            values of `func` are slotted into a (virtual) dataset (if
            `write_worker_results` is True) or array (otherwise) of shape
            `result_shape`, where a single `None` entry designates the stacking
            dimension. For instance, ``result_shape = (None, 100)`` implies
            that `func` returns a 100-element array which is to be stacked
            along the first dimension for each concurrent call of `func`
            resulting in a ``(n_inputs, 100)`` dataset or array. See Notes
            and Examples for details. See Examples as well as [1]_ and [2]_
            for more information.
        result_dtype : str or None
            Only relevant if `result_shape` is not `None`. If provided, determines
            the numerical datatype of the dataset laid out by `result_shape`.
            By default, results are stored in `float64` format. See [2]_ for
            more details.
        single_file : bool
            Only relevant if `write_worker_results` is `True` and `write_pickle`
            is `False`. If `single_file` is `False` (default), the results of each parallel
            call of `func` are stored in dedicated HDF5 files, such that the auto-
            generated HDF5 results-container is a collection of symbolic links
            pointing to these files.
            Conversely, if `single_file` is `True`, all parallel workers
            write to the same results container (using a distributed file-locking
            mechanism). See [2]_ for more details.
        write_pickle : bool
            Only relevant if `write_worker_results` is `True`. If `True`,
            the return value(s) of `func` is/are pickled to disk (one
            `'.pickle'`-file per parallel worker). See Examples as well as
            [1]_ and [2]_ for more information.
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
        n_workers : int or "auto"
            Number of SLURM workers (=jobs) to spawn. If `"auto"` (default), then
            ``n_workers = n_inputs``, i.e., every SLURM worker performs a single
            call of `func`.
            If `n_inputs` is large and executing `func` is fast, setting
            ``n_workers = int(n_inputs / 2)`` might be beneficial. See Examples
            as well as [1]_ and [2]_ for more information.
        mem_per_worker : str
            Memory booking for each SLURM worker. If `"auto"` (default), the standard
            value is inferred from the used partition (if possible). See
            :func:`~acme.slurm_cluster_setup` for details.
        setup_timeout : int
            Timeout period (in seconds) for SLURM workers to come online. Refer to
            keyword `timeout` in :func:`~acme.slurm_cluster_setup` for details.
        setup_interactive : bool
            If `True` (default), user input is queried in case not enough SLURM
            workers could be started within `setup_timeout` seconds. If no input
            is provided, the current number of spawned workers is used (even if
            smaller than the amount requested by `n_workers`). If `False`, no user
            choice is requested. Refer to keyword `interactive` in :func:`~acme.slurm_cluster_setup`
        stop_client : bool or "auto"
            If `"auto"` (default), automatically started distributed computing clients
            are shut down at the end of computation, while user-provided clients
            are left untouched. If `False`, automatically started clients are
            left running after completion, user-provided clients are left untouched.
            If `True`, auto-generated clients *and* user-provided clients are
            shut down at the end of the computation. See Examples as well
            as [1]_ and [2]_ for more information.
        verbose : None or bool
            If `None` (default), general run-time information as well as warnings
            and errors are shown. If `True`, additionally debug information is
            shown. If `False`, only warnings and errors are propagated.
            See [2]_ for more details.
        dryrun : bool
            If `True` the user-provided function `func` is executed once using
            one of the input argument tuples prepared for the parallel workers (picked
            at random). If `setup_interactive` is `True`, a prompt asks if the
            actual parallel execution of `func` is supposed to be launched after the
            dry-run. The `dryrun` keyword is intended to to estimate memory consumption
            as well as runtime of worker jobs prior to the actual concurrent
            computation. See [1]_ and [2]_ for more information.
        logfile : None or bool or str
            If `None` (default) or `False`, all run-time information as well as errors and
            warnings are printed to the command line only. If `True`, an auto-generated
            log-file is set up that records run-time progress. Alternatively, the
            name of a custom log-file can be provided (must not exist). The verbosity
            of recorded information can be controlled via setting `verbose`.
            See [2]_ for more details.

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

        Notes
        -----
        Please consult [1]_ for detailed usage information.

        See also
        --------
        esi_cluster_setup : spawn custom SLURM worker clients on the ESI HPC cluster
        local_cluster_setup : start a local Dask multi-processing cluster on the host machine
        ACMEdaemon : Manager class performing the actual concurrent processing

        References
        ----------
        .. [1] https://esi-acme.readthedocs.io/en/latest/userguide.html
        .. [2] https://esi-acme.readthedocs.io/en/latest/advanced_usage.html
        """

        # First and foremost, set up logging system (unless logger is already present)
        self.log = acs.prepare_log(func, caller=self.msgName, logfile=logfile, verbose=verbose)

        # Backwards compatibility: legacy keywords are converted to new nomenclature
        if any(kw in kwargs for kw in __deprecated__):
            self.log.warning(__deprecation_wrng__)
            n_workers = kwargs.pop("n_jobs", n_workers)
            mem_per_worker = kwargs.pop("mem_per_job", mem_per_worker)

        # Either guess `n_inputs` or use provided value to duplicate input args
        # and set class attributes `n_inputs`, `argv` and `kwargv`
        self.prepare_input(func, n_inputs, *args, **kwargs)

        # Create an instance of `ACMEdaemon` that does the actual parallel computing work
        self.daemon = ACMEdaemon(self,
                                 n_workers=n_workers,
                                 write_worker_results=write_worker_results,
                                 output_dir=output_dir,
                                 result_shape=result_shape,
                                 result_dtype=result_dtype,
                                 single_file=single_file,
                                 write_pickle=write_pickle,
                                 dryrun=dryrun,
                                 partition=partition,
                                 mem_per_worker=mem_per_worker,
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
        # either all input arguments must have unit length (or are nd-arrays)
        # or at at least one input argument actually contains `n_input` elements
        if guessInputs:
            if len(set(argLens)) > 1 or len(argLens) == 0:
                msg = "{} automatic input distribution failed: found {} objects " +\
                    "containing {} to {} elements. Please specify `n_inputs` manually. "
                raise ValueError(msg.format(self.msgName,
                                            len(argLens),
                                            min(argLens, default=0),
                                            max(argLens, default=0)))
            n_inputs = argLens[0]
        else:
            if n_inputs not in set(argLens) and not all(arglen == 1 for arglen in argLens):
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
