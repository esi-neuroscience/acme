Getting started with ACME
=========================

Installing ACME
---------------

ACME can be installed using `Pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

    pip install esi-acme

and soon using conda as well. ACME is also part of the
`Syncopy package <https://pypi.org/project/esi-syncopy/>`_.
If you're working on the ESI HPC cluster installing ACME and/or SyNCoPy is only necessary if
you create your own Conda environment.

Setting Up Your Python Environment
----------------------------------

On the ESI cluster, all current pre-configured reference environments
(`ESI-202xa/b`) provide the respective most recent ACME version. These environments
can be easily started using the `ESI JupyterHub <https://jupyterhub.esi.local>`_

Using ACME
----------
In contrast to many Python packages, ACME is built around a single key element,
the :class:`~acme.ParallelMap` context manager. Thus, for most use-cases importing
:class:`~acme.ParallelMap` is likely sufficient to start ACME parallelization:

.. code-block:: python

    from acme import ParallelMap

    def f(x, y, z=3):
        return (x + y) * z

    with ParallelMap(f, [2, 4, 6, 8], 4) as pmap:
        pmap.compute()

More fine-grained control over allocated resources and load-balancer options is available
via the routine :func:`~acme.esi_cluster_setup`. It permits to launch a custom-tailored
"cluster" of parallel workers (corresponding to CPU cores if run on a single machine, i.e.,
laptop or workstation, or compute jobs if run on a cluster computing manager such as SLURM).
Thus, instead of letting ACME automatically allocate a worker swarm, more fine-grained
control over resource allocation and management can be achieved via running
:func:`~acme.esi_cluster_setup` **before** launching the actual calculation.
For example,

.. code-block:: python

    slurmClient = esi_cluster_setup(partition="16GBXL", n_jobs=10)

starts 10 concurrent SLURM workers in the `16GBXL` queue if run on the ESI HPC
cluster. Any subsequent invocation of :class:`~acme.ParallelMap` will automatically
pick up ``slurmClient`` and distribute any occurring computational payload across
the workers collected in ``slurmClient``.

More Information
^^^^^^^^^^^^^^^^

Details about optional arguments of :class:`~acme.ParallelMap` are summarized in
its constructor :meth:`~acme.ParallelMap.__init__` method.
Additional :class:`~acme.ParallelMap` usage examples are provided in the repository's
`README <https://github.com/esi-neuroscience/acme#acme-asynchronous-computing-made-easy>`_.
