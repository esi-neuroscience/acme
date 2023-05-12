.. Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
.. in Cooperation with Max Planck Society

.. SPDX-License-Identifier: CC-BY-NC-SA-1.0

Troubleshooting + FAQ
======================
ACME is solely intended for executing user-provided functions multiple
times in parallel. Thus, only problems that can be split up into
independent tasks can be processed with :class:`~acme.ParallelMap` ("embarassingly parallel workloads").
Inter-process communication, worker-synchronization or shared memory
problems **will not work**.

.. contents:: Quick Links
    :depth: 2

If Things Go Wrong
^^^^^^^^^^^^^^^^^^
First and foremost, ensure that the function you want to execute in parallel
runs fine on its own. Consider

.. code-block:: python

    def f(x, y, z=3, w=np.arange(3)):
        return (x.sum() + y) * z * w.max()

Then

.. code-block:: python

    >>> f(np.ones((3,)), 4)
    42.0

works as expected. However,

.. code-block:: python

    >>> f([1,1,1], 4)
    AttributeError                            Traceback (most recent call last)
    Input In [10], in <cell line: 1>()
    ----> 1 f([1,1,1], 4)

    Input In [8], in f(x, y, z, w, **kwargs)
        1 def f(x, y, z=3, w=np.arange(3), **kwargs):
    ----> 2     return (x.sum() + y) * z * w.max()

    AttributeError: 'list' object has no attribute 'sum'

fails, since the Python list ``[1,1,1]`` does not support summing its components via
the method ``.sum()`` as NumPy arrays do. Thus, invoking :class:`~acme.ParallelMap` like this:

.. code-block:: python

    with ParallelMap(f, [[1, 1, 1], [2, 2, 2]], 4) as pmap:
        results = pmap.compute()

will similarly make any spawned distributed workers crash with ``Compute Failed``
errors. The root problem is of course completely unrelated to actual parallel
execution of `f` but is instead actually caused by using the wrong input type for ``x``.
*Nail It Before You Scale It*.

Once you have ensured that your function works fine in a sequential setting,
you can try increasing ACME's logging verbosity to get a better understanding
of what's happening under the hood:

.. code-block:: python

    with ParallelMap(myfunc, ..., logfile=True, verbose=True) as pmap:
        results = pmap.compute()

If your function works fine, but you think something's wrong with ACME,
please let us know by opening a bug report in our
`GitHub Issue Tracker <https://github.com/esi-neuroscience/acme/issues>`_.

FAQ
^^^

Q: My function runs fine sequentially. But with ACME I'm getting lots of ``distributed.core - INFO - Event loop was unresponsive in Worker for 3.34s.  This is often caused by long-running GIL-holding functions or moving large chunks of data. This can cause timeouts and instability.`` Errors
***************************************************************************************************************************************************************************************************************************************************************************************************
This may happen if the function you're wrapping with :class:`~acme.ParallelMap` is defined
in the same script you also invoke :class:`~acme.ParallelMap`. See :ref:`Best Practices <mainblock>` for
a strategy to circumvent this problem.

Q: I try to run ACME locally on my machine but I always get a ``RuntimeError``
******************************************************************************
If you call :class:`~acme.ParallelMap` (or :func:`~acme.local_cluster_setup`) inside a script that does not
contain a ``if __name__ == "__main__"`` block, starting parallel workers results
in an infinite recursion triggered by new processes being started before the calling
process can finish its bootstrapping phase. Thus, try wrapping :class:`~acme.ParallelMap`
(or :func:`~acme.local_cluster_setup`) inside a main module block, i.e.,

.. code-block:: python

    if __name__ == "__main__":
        ...
        local_cluster_setup()

        with ParallelMap(...) ...
        ...

If you still encounter problems, try migrating the function you are :class:`~acme.ParallelMap`-ping
to a separate `.py` file (see :ref:`Best Practices <isolation>`).

Q: I started a parallel computing client manually - how can I shut it down?
***************************************************************************
You can simply use the client's ``.close()`` class method (e.g., ``myclient.close()``)
or use the ACME convenience function :func:`~acme.cluster_cleanup`.


ESI-HPC Cluster Specifics
^^^^^^^^^^^^^^^^^^^^^^^^^
To make optimal use of the ESI HPC infrastructure, please make sure to first
pick the right SLURM partition for your task (ACME's default falls back to
our "smallest" partition `8GBXS`).

Instead of letting ACME automatically spawn and kill SLURM jobs, you can use
the helper function :func:`~acme.esi_cluster_setup` to start a parallel computing client
using a set number of SLURM jobs (=workers). This client can be re-used across
several invocations of :class:`~acme.ParallelMap`. Specifically, if you plan to call :class:`~acme.ParallelMap`
multiple times in your analysis script, overall runtime performance can be greatly improved
by creating a client once at the beginning and subsequently "recycling" it for every
:class:`~acme.ParallelMap` call (otherwise ACME spawns and kills workers every time you invoke
:class:`~acme.ParallelMap`). In practice, a single line at the top of your script is sufficient
to put this strategy into action (ACME picks up any existing client automatically,
you don't have to tell it beforehand):

.. code-block:: python

    from acme import ParallelMap, cluster_cleanup

    if __name__ == "__main__":
        myclient = esi_cluster_setup(partition="16GBXL", n_workers=10)

        ...
        with ParallelMap(...) as pmap:
            pmap.compute()

        ...
        with ParallelMap(...) as pmap:
            pmap.compute()

        ...
        with ParallelMap(...) as pmap:
            pmap.compute()

        cluster_cleanup(myclient)
