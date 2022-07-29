ACME User Guide
===============
Learn how to get the most out of ACME for your own work by running through a
quick (but hopefully) illustrative example that starts simple and subsequently
turns on all of ACME's bells and whistles.

.. contents:: Quick Links
    :depth: 3

Walkthrough
-----------
Assume the function defined below is supposed to be run multiple times
via SLURM for different values of `x`, `y` and `z`

.. code-block:: python

    def f(x, y, z=3):
        return (x + y) * z

The following code calls `f` with four different values of `x` (namely 2, 4, 6 and 8)
sets `y` to 4 and leaves `z` at its default value of 3:

.. code-block:: python

    from acme import ParallelMap
    with ParallelMap(f, [2, 4, 6, 8], 4) as pmap:
        results = pmap.compute()

Where Are My Results?
^^^^^^^^^^^^^^^^^^^^^
By default results are saved to disk in HDF5 format and `results` is a list
of the corresponding filenames:

.. code-block:: python

    >>> results
    ['/cs/home/username/ACME_20201217-135011-448825/f_0.h5',
     '/cs/home/username/ACME_20201217-135011-448825/f_1.h5',
     '/cs/home/username/ACME_20201217-135011-448825/f_2.h5',
     '/cs/home/username/ACME_20201217-135011-448825/f_3.h5']

The contents of the containers can be accessed using `h5py`:

.. code-block:: python

    out = np.zeros((4, ))
    import h5py
    for ii, fname in enumerate(results):
        with h5py.File(fname, 'r') as f:
            out[ii] = np.array(f['result_0'])

which yields

.. code-block:: python

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

Now `results` is a list of integers:

.. code-block:: python

    >>> results
    [18, 24, 30, 36]

Override Automatic Input Argument Distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, suppose `f` has to be evaluated for the same values of `x` (again
2, 4, 6 and 8), but `y` is not a number but a NumPy array:

.. code-block:: python

    y = np.ones((3,)) * 4
    with ParallelMap(f, [2, 4, 6, 8], y) as pmap:
        results = pmap.compute()

This fails, because it is not clear which input is to be split up and distributed
across workers for parallel execution:

.. code-block:: python

    >>> ValueError: <ParallelMap> automatic input distribution failed: found 2 objects containing 3 to 4 elements. Please specify `n_inputs` manually.

In this case, `n_inputs` has to be provided explicitly (`write_worker_results`
is set to `False` for illustrative purposes only)

.. code-block:: python

    with ParallelMap(f, [2, 4, 6, 8], y, n_inputs=4, write_worker_results=False) as pmap:
        results = pmap.compute()

yielding

.. code-block:: python

    >>> results
    [array([18., 18., 18.]),
     array([24., 24., 24.]),
     array([30., 30., 30.]),
     array([36., 36., 36.])]


Reuse Worker Clients
^^^^^^^^^^^^^^^^^^^^^
Now suppose `f` needs to be evaluated for fixed values of `x` and `y`
with `z` varying randomly 500 times between 1 and 10. Since `f` is a
very simple function, it is not necessary to spawn 500 SLURM jobs for this.
Instead, allocate only 50 jobs in the smallest available queue ("8GBXS" on the ESI HPC cluster),
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

.. code-block:: python

    >>> len(results)
    500

In a subsequent computation `f` needs to be evaluated for 1000 samples of
`z`. In the previous call, `stop_client` was `False`, thus the next
invocation of :class:`~acme.ParallelMap` re-uses the existing SLURM worker swarm:

.. code-block:: python

    z = rng.integers(low=1, high=10, size=1000, endpoint=True)
    with ParallelMap(f, x, y, z=z) as pmap:
        results = pmap.compute()

Note the info message:

.. code-block:: python

    >>> <ParallelMap> INFO: Attaching to global parallel computing client <Client: 'tcp://10.100.32.5:39747' processes=50 threads=50, memory=400.00 GB>

Non-Interactive Logging
^^^^^^^^^^^^^^^^^^^^^^^
Finally, suppose `f` has to be called for 20000 different values of `z`.
Under the assumption that this computation takes a while, any run-time
messages are to be written to a an auto-generated log-file:

.. code-block:: python

    z = rng.integers(low=1, high=10, size=20000, endpoint=True)
    with ParallelMap(f, x, y, z=z, logfile=True) as pmap:
        results = pmap.compute()

Alternatively, logging information may be written to a file "my_log.txt" instead

.. code-block:: python

    z = rng.integers(low=1, high=10, size=20000, endpoint=True)
    with ParallelMap(f, x, y, z=z, logfile="my_log.txt") as pmap:
        results = pmap.compute()

Alternative Storage Format: Pickle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In some cases it might be necessary to work with objects that are not
HDF5 compatible, e.g., sparse matrices created by `scipy.sparse`. Consider

.. code-block:: python

    from scipy.sparse import spdiags
    ndim = 4
    x = spdiags(np.ones((ndim,)), 0, ndim, ndim)
    y = spdiags(3 * np.ones((ndim,)), 0, ndim, ndim)

Then

.. code-block:: python

    >>> x
    <4x4 sparse matrix of type '<class 'numpy.float64'>'
        with 4 stored elements (1 diagonals) in DIAgonal format>
    >>> y
    <4x4 sparse matrix of type '<class 'numpy.float64'>'
        with 4 stored elements (1 diagonals) in DIAgonal format>
    >>> x.toarray()
    array([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
    >>> y.toarray()
    array([[3., 0., 0., 0.],
        [0., 3., 0., 0.],
        [0., 0., 3., 0.],
        [0., 0., 0., 3.]])
    >>> f(x, y)
    <4x4 sparse matrix of type '<class 'numpy.float64'>'
        with 4 stored elements (1 diagonals) in DIAgonal format>

In this case, the default HDF5 storage format can be overridden using the
keyword `write_pickle`

.. code-block:: python

    with ParallelMap(f, x, y, n_inputs=5, write_pickle=True) as pmap:
        results = pmap.compute()

which yields

.. code-block:: python

    >>> results
    ['/cs/home/username/ACME_20201217-135011-448825/f_0.pickle',
    '/cs/home/username/ACME_20201217-135011-448825/f_1.pickle',
    '/cs/home/username/ACME_20201217-135011-448825/f_2.pickle',
    '/cs/home/username/ACME_20201217-135011-448825/f_3.pickle']

Debugging
^^^^^^^^^
Debugging programs running in parallel can be quite tricky.
For instance, assume the function `f` is (erroneously) called with `z`
set to `None`. In a regular sequential setting, identifying the problem
is (relatively) straight-forward:

.. code-block:: python

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
    <function ACMEdaemon.func_wrapper at 0x7f42569f1e50>, [[2], [4], [None], ['/cs/home/fuertingers/ACME_20201217-160137-984430'],
    ['f_0.h5'], [0], [<function f at 0x7f425c34bee0>]], ['z', 'outDir', 'outFile', 'taskID', 'userFunc'], {})))
    kwargs:    {}
    Exception: TypeError("unsupported operand type(s) for *: 'int' and 'NoneType'")
    slurmstepd: error: *** JOB 1873974 ON esi-svhpc18 CANCELLED AT 2020-12-17T16:01:43 ***

To narrow down problems with parallel execution, the `compute` method
of :class:`~acme.ParallelMap` offers the `debug` keyword. If enabled, all function calls
are performed in the local thread of the active Python interpreter. Thus, the execution
is **not** actually performed in parallel. This allows regular error propagation
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

In addition, the automatically generated argument distribution to user-provided
functions can be tested via the `dryrun` keyword. This permits to test-drive
ACME's automatically generated argument lists prior to the actual concurrent
computation, e.g.,

.. code-block:: python

    >>> with ParallelMap(f, [2, 4, 6, 8], 4, dryrun=True) as pmap:
    >>>     results = pmap.compute()
    <ParallelMap> INFO: Performing a single dry-run of f simulating randomly picked worker #1 with automatically distributed arguments
    <ParallelMap> INFO: Dry-run completed. Elapsed time is 0.004725 seconds, estimated memory consumption was 0.01 MB.
    Do you want to continue executing f with the provided arguments? [Y/n] n

In general it is strongly recommended to make sure any function supplied
to :class:`~acme.ParallelMap` works as intended in a sequential setting prior to running
it in parallel.

Wait, There's More...
---------------------
ACME attempts to be as agnostic of the functions it is wrapping as possible. However,
there are some technical limitations that impose medium to hard boundaries as to
what a user-provided function can and should do.

User-Function Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^
The user-provided function `func` has to meet some basic requirements to
permit parallel execution with :class:`~acme.ParallelMap`:

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

Auto-Generated HDF5-Files
^^^^^^^^^^^^^^^^^^^^^^^^^
All HDF5 files auto-generated by :class:`~acme.ParallelMap` are stored in a directory
*ACME_YYYYMMDD-hhmmss-ffffff* (encoding the current time as
*YearMonthDay-HourMinuteSecond-Microsecond*) that is created in the user's
home directory on ``/cs`` (if ACME is running on the ESI HPC cluster) or the
current working directory (if running locally). The HDF5 files themselves
are named *funcname_workerid.h5*, where `funcname` is the name of the user-provided
function and `workerid` encodes the number of the worker that generated
the file (see `Walkthrough`_ above for examples).
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
