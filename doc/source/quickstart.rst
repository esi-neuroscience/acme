.. Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
.. in Cooperation with Max Planck Society

.. SPDX-License-Identifier: CC-BY-NC-SA-1.0

Quickstart
==========

Installing ACME
---------------

For Users
^^^^^^^^^

ACME can be installed using ``pip`` or ``conda``:

.. tabs::

    .. tab:: conda

        .. code-block:: shell

            conda install -c conda-forge esi-acme

    .. tab:: pip

        .. code-block:: shell

            pip install esi-acme

If you're working on the ESI or CoBIC HPC clusters installing ACME is only
necessary if you create your own conda environment.
On the ESI cluster, all current pre-configured reference environments
(`ESI-202xa/b`) provide the respective most recent ACME version. These environments
can be easily started using the `ESI JupyterHub <https://jupyterhub.esi.local>`_.
On the CoBIC cluster, ACME is pre-installed in the pre-configured
`neuro-conda <https://github.com/neuro-conda/neuro-conda>`_ environments (`neuro-conda-202xa/b`).

For Developers
^^^^^^^^^^^^^^

To install the latest development version of ACME in a fresh conda environment
(called `acme` by default), please follow these steps:

.. code-block:: bash

    git clone https://github.com/esi-neuroscience/acme.git
    cd acme/
    conda env create -f acme.yml
    pip install -e .

(note the dot "." at the end of the ``pip install`` command). More details
are provided in :doc:`Contributing to ACME <contributing>`.

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

For more fine-grained control over resource allocation and load-balancer options,
ACME offers "cluster setup" convenience functions to launch a custom-tailored
"client" of parallel workers (corresponding to CPU cores if run on a single
machine, i.e., laptop or workstation, or compute jobs if run on a cluster
computing manager such as SLURM). These helper functions are mere wrappers
around :class:`distributed.LocalCluster` and :class:`dask_jobqueue.SLURMCluster`
which perform the actual heavy lifting.
Thus, instead of letting ACME automatically allocate a worker swarm, more
fine-grained control over resource allocation and management can be achieved
by running :func:`~acme.slurm_cluster_setup` (on any HPC cluster managed by the
`SLURM Workload Manager <https://slurm.schedmd.com/documentation.html>`_) or
:func:`~acme.local_cluster_setup` (on local multi-processing hardware)
**before** launching the actual calculation. For example,

.. code-block:: python

    from acme import slurm_cluster_setup

    slurmClient = slurm_cluster_setup(partition="some_partition",
                                      cores_per_worker=2,
                                      mem_per_worker="2GB",
                                      n_workers=10)

starts 10 concurrent SLURM workers each provisioned with two CPU cores and
2 GB of RAM in a queue named `"some_partition"`. Any subsequent invocation
of :class:`~acme.ParallelMap` will automatically pick up ``slurmClient``
and distribute any occurring computational payload across the workers collected
in ``slurmClient``.

On the ESI HPC cluster the routine :func:`~acme.esi_cluster_setup` provides
some sane defaults tailored to the specifics of the SLURM layout of the cluster.
For instance,

.. code-block:: python

    esiClient = esi_cluster_setup(partition="16GBXL", n_workers=10)

starts 10 concurrent SLURM workers in the `16GBXL` partition (no need to further
specify CPU core count or memory requirements).

Analogously, on the CoBIC HPC cluster, the routine :func:`~acme.bic_cluster_setup`
provides similar functionality, e.g.,

.. code-block:: python

    bicClient = bic_cluster_setup(partition="16GBSppc", n_workers=10)

starts 10 concurrent SLURM workers in the `16GBSppc` partition (similarly, CPU
core count and memory requirements are set automatically).

.. note::
    Since ACME internally relies on `distributed <https://distributed.dask.org/en/stable/>`_
    and `dask_jobqueue <https://jobqueue.dask.org/en/latest/>`_ it can leverage
    any HPC infrastructure (CPU nodes, GPU nodes etc.) managed by SLURM, PBS,
    SGE, Moab etc. For users of the ESI and CoBIC HPC clusters ACME offers the above
    presented convenience functions :func:`~acme.esi_cluster_setup` and
    :func:`~acme.bic_cluster_setup`. However, the underlying general purpose
    setup routine :func:`acme.slurm_cluster_setup` (which is invoked by these
    convenience functions) can be used on its own to start a parallel worker
    cluster on any distributed system controlled by SLURM.
    If you are interested in having a `*_cluster_setup` routine for your institution's
    HPC infrastructure being included in ACME, please open an issue in our
    `GitHub Issue Tracker <https://github.com/esi-neuroscience/acme/issues>`_.


More Information
^^^^^^^^^^^^^^^^

Details about optional arguments of :class:`~acme.ParallelMap` are summarized in
its constructor :meth:`~acme.ParallelMap.__init__` method. Additional usage examples
and tutorials are provided in the :doc:`User Walkthrough <userguide>`.
