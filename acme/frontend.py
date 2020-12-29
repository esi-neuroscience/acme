# -*- coding: utf-8 -*-
#
# User-exposed interface of acme
#

# Builtin/3rd party package imports
import inspect
import numpy as np

# Local imports
from .backend import ACMEdaemon
from . import shared as acs

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
        setup_timeout=60,
        setup_interactive=True,
        stop_client="auto",
        verbose=None,
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
        partition : str
            Name of SLURM partition to use. If `"auto"` (default), the memory footprint
            of `func` is estimated using dry-run stubs based on randomly sampling
            provided `args` and `kwargs`. Estimated memory usage dictates queue
            auto-selection under the assumption of short run-times. For instance,
            with a predicted memory footprint of 6 GB the `"8GBXS"` queue is selected
            (minimal but sufficient memory and shortest runtime). To override
            auto-selection, provide name of SLURM queue explicitly. See
            :func:`~acme.esi_cluster_setup` for details.
        n_jobs : int or "auto"
            Number of SLURM jobs (=workers) to spawn. If `"auto"` (default), then
            ``n_jobs = n_inputs``, i.e., every SLURM worker performs a single
            call of `func`.
            If `n_inputs` is large and executing `func` is fast, setting
            ``n_jobs = int(n_inputs / 2)`` might be beneficial. See Notes for details.
        mem_per_job : str
            Memory booking for each SLURM worker. If `"auto"` (default), the standard
            value is inferred from the used partition (if possible). See
            :func:`~acme.esi_cluster_setup` for details.
        setup_timeout : int
            Timeout period (in seconds) for SLURM workers to come online. See
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
            results is a list of lists comprising the actual return values of
            `func`.

        Examples
        --------
        Assume the function defined below is supposed to be run multiple times
        via SLURM for different values of `x`, `y` and `z`

        .. code-block:: python

            def f(x, y, z=3):
                return (x + y) * z

        The following code calls the function `f` with four different values of
        `x` (namely 2, 4, 6 and 8) setting `y` to 4 and leaving `z` at its default
        value of 3:

        .. code-block:: python

            from acme import ParallelMap

            with ParallelMap(f, [2, 4, 6, 8], 4) as pmap:
                results = pmap.compute()

        By default results are saved to disk in HDF5 format and `results` is a list
        of the corresponding filenames:

        >>> results
        ['/mnt/hpx/home/username/ACME_20201217-135011-448825/f_0.h5',
         '/mnt/hpx/home/username/ACME_20201217-135011-448825/f_1.h5',
         '/mnt/hpx/home/username/ACME_20201217-135011-448825/f_2.h5',
         '/mnt/hpx/home/username/ACME_20201217-135011-448825/f_3.h5']

        The contents of the containers can be accessed using `h5py`, e.g.,

        .. code-block:: python

            out = np.zeros((4, ))
            import h5py
            for ii, fname in enumerate(results):
                with h5py.File(fname, 'r') as f:
                    out[ii] = np.array(f['result_0'])

        which yields

        >>> out
        array([18., 24., 30., 36.])

        Alternatively, results may be collected directly in memory by setting
        `write_worker_results` to `False`. This is **not** recommended, since
        values have to be gathered from compute nodes via ethernet (slow) and
        are accumulated in the local memory of the interactive node you are using
        (potential memory overflow):

        .. code-block:: python

            with ParallelMap(f, [2, 4, 6, 8], 4, write_worker_results=False) as pmap:
                results = pmap.compute()

        Now `results` is a list of lists:

        >>> results
        [[18], [24], [30], [36]]

        To extract values into a NumPy array one may use

        >>> out = np.array([xi[0][0] for xi in results])

        Next, suppose `f` has to be evaluated for the same values of `x` (again
        2, 4, 6 and 8), but `y` is not a number but a NumPy array:

        .. code-block:: python

            y = np.ones((3,)) * 4
            with ParallelMap(f, [2, 4, 6, 8], y) as pmap:
                results = pmap.compute()

        This fails, because it is not clear which input is to be split up and distributed
        across workers for parallel execution:

        >>> ValueError: <ParallelMap> automatic input distribution failed: found 2 objects containing 3 to 4 elements. Please specify `n_inputs` manually.

        In this case, `n_inputs` has to be provided explicitly (`write_worker_results`
        is set to `False` for illustrative purposes only)

        .. code-block:: python

            with ParallelMap(f, [2, 4, 6, 8], y, n_inputs=4, write_worker_results=False) as pmap:
                results = pmap.compute()

        yielding

        >>> results
        [[array([18., 18., 18.])],
         [array([24., 24., 24.])],
         [array([30., 30., 30.])],
         [array([36., 36., 36.])]]

        Now suppose `f` needs to be evaluated for fixed values of `x` and `y`
        with `z` varying randomly 500 times between 1 and 10. Since `f` is a
        very simple function, it is not necessary to spawn 500 SLURM jobs for this.
        Instead, allocate only 50 workers in the smallest available queue "8GBXS",
        i.e., each worker has to perform 10 evaluations of `f`. Additionally, keep the workers
        alive for re-use afterwards

        .. code-block:: python

            import numpy as np
            x = 2
            y = 4
            rng = np.random.default_rng()
            z = rng.integers(low=1, high=10, size=500, endpoint=True)

            with ParallelMap(f, x, y, z=z, n_jobs=50, partition="8GBXS", stop_client=False) as pmap:
                results = pmap.compute()

        This yields

        >>> len(results)
        500

        In a subsequent computation `f` needs to be evaluated for 1000 samples of
        `z`. In the previous call, `stop_client` was `False`, thus the next
        invocation of `ParallelMap` re-uses the existing SLURM worker swarm:

        .. code-block:: python

            z = rng.integers(low=1, high=10, size=1000, endpoint=True)

            with ParallelMap(f, x, y, z=z) as pmap:
                results = pmap.compute()

        Note the info message:

        >>> <ParallelMap> INFO: Attaching to global parallel computing client <Client: 'tcp://10.100.32.5:39747' processes=50 threads=50, memory=400.00 GB>

        Finally, suppose `f` has to be called for 20000 different values of `z`.
        Under the assumption that this computation takes a while, any run-time
        messages are to be written to a an auto-generated log-file:

        .. code-block:: python

            z = rng.integers(low=1, high=10, size=20000, endpoint=True)

            with ParallelMap(f, x, y, z=z, logfile=True) as pmap:
                results = pmap.compute()

        Alternatively, logging information may be written to a file my_log.txt instead

        .. code-block:: python

            z = rng.integers(low=1, high=10, size=20000, endpoint=True)

            with ParallelMap(f, x, y, z=z, logfile="my_log.txt") as pmap:
                results = pmap.compute()

        Note that debugging programs running in parallel can be quite tricky.
        For instance, assume the function `f` is (erroneously) called with `z`
        set to `None`. In a regular sequential setting, identifying the problem
        is (relatively) straight-forward:

        >>> f(2, 4, z=None)
        TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'

        However, when executing `f` in parallel using SLURM

        .. code-block:: python

            with ParallelMap(f, [2, 4, 6, 8], 4, z=None) as pmap:
                results = pmap.compute()

        the resulting error message can be somewhat overwhelming

        .. code-block:: python

            Function:  execute_task
            args:      ((<function reify at 0x7f425c25b0d0>, (<function map_chunk at 0x7f425c25b4c0>,
            <function ACMEdaemon.func_wrapper at 0x7f42569f1e50>, [[2], [4], [None], ['/mnt/hpx/home/fuertingers/ACME_20201217-160137-984430'],
            ['f_0.h5'], [0], [<function f at 0x7f425c34bee0>]], ['z', 'outDir', 'outFile', 'taskID', 'userFunc'], {})))
            kwargs:    {}
            Exception: TypeError("unsupported operand type(s) for *: 'int' and 'NoneType'")
            slurmstepd: error: *** JOB 1873974 ON esi-svhpc18 CANCELLED AT 2020-12-17T16:01:43 ***

        To narrow down problems with parallel execution, the `compute` method
        of `ParallelMap` offers the `debug` keyword. If enabled, all function calls
        are performed in the local thread of the active Python interpreter. Thus, the execution
        is **not** actually performed in parallel. This allows regular error progration
        and even permits the use of tools like `pdb <https://docs.python.org/3/library/pdb.html>`_
        or ``%debug`` `iPython magics <https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-debug>`_.

        .. code-block:: python

            with ParallelMap(f, [2, 4, 6, 8], 4, z=None) as pmap:
                results = pmap.compute(debug=True)

        which results in

        .. code-block:: python

            <ipython-input-2-47feb885f020> in f(x, y, z)
                1 def f(x, y, z=3):
            ----> 2     return (x + y) * z

            TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'

        In general it is strongly recommended to make sure any function supplied
        to `ParallelMap` works as intended in a sequential setting prior to running
        it in parallel.

        More examples and usage notes can be found in the package
        `README <https://github.com/esi-neuroscience/acme#acme-asynchronous-computing-made-easy>`_.

        Notes
        -----
        This code is solely intended for executing user-provided functions multiple
        times in parallel. Thus, only problems that can be split up into
        independent tasks can be processed with `ParallelMap` ("embarassingly parallel workloads").
        Inter-process communication, worker-synchronization or shared memory
        problems **will not work**.

        **User-Function Requirements**

        The user-provided function `func` has to meet some basic requirements to
        permit parallel execution with `ParallelMap`:

        * **input arguments of `func`** should be regular Python objects (lists, tuples,
          scalars, strings etc.) or NumPy arrays. Custom user-defined classes
          may or may not work. In general, anything that can be serialized via
          `cloudpickle <https://pypi.org/project/cloudpickle/>`_ should work out of the box.
        * if automatic result saving is used (`write_worker_results` is `True`),
          the **return value(s) of `func`** have to be suitable for storage in HDF5
          containers. Thus, anything returned by `func` should be either purely
          numeric (scalars or NumPy arrays) or purely lexical (strings). Hybrid
          text/numeric data-types (e.g., Pandas dataframes), custom class instances,
          functions, generators or complex objects (like matplotlib figures)
          **will not work**.

        **Auto-Generated HDF5-Files**

        All HDF5 files auto-generated by `ParallelMap` are stored in a directory
        *ACME_YYYYMMDD-hhmmss-ffffff* (encoding the current time as
        *YearMonthDay-HourMinuteSecond-Microsecond*) that is created in the user's
        home directory on ``hpx`` (if ACME is running on the ESI cluster) or the
        current working directory (if running locally). The HDF5 files themselves
        are named *funcname_workerid.h5*, where `funcname` is the name of the user-provided
        function and `workerid` encodes the number of the worker that generated
        the file (cf. Examples).

        The internal structure of all HDF5 files is kept as simple as possible:
        each return value of the user-provided function `func` is saved in a
        separate dataset in the file's root group. For instance, processing
        the following user-provided function

        .. code-block:: python

            def this_func(a, b, c):
                # ...some complicated calculations...
                return r0, r1, r2

        with 50 workers using ``write_worker_results = True`` yields 50 HDF5
        files *this_func_0.h5*, *this_func_1.h5*, ..., *this_func_49.h5* each
        containing three datasets `"result_0"` (holding `r0`), `"result_1"`
        (holding `r1`) and `"result_2"` (holding `r2`). User-provided functions
        with only a single return value correspondingly yield HDF5 files that
        only contain one dataset (`"result_0"`) in their respective root group.

        See also
        --------
        esi_cluster_setup : spawn custom SLURM worker clients
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

    def compute(self):
        if hasattr(self, "daemon"):
            self.daemon.compute()

    def cleanup(self):
        if hasattr(self, "daemon"):
            self.daemon.cleanup

    def __enter__(self):
        return self.daemon

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.daemon.cleanup()
