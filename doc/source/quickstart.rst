.. Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
.. in Cooperation with Max Planck Society

.. SPDX-License-Identifier: CC-BY-NC-SA-1.0

Quickstart
==========

Installing ACME
---------------

For Users
^^^^^^^^^

ACME can be installed using `pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

    pip install esi-acme

or via `conda <https://www.anaconda.com/products/individual>`_

.. code-block:: bash

    conda install -c conda-forge esi-acme

ACME is the parallelization engine of `Syncopy <https://pypi.org/project/esi-syncopy/>`_.
If you're working on the ESI HPC cluster installing ACME and/or SyNCoPy is only necessary if
you create your own conda environment.
On the ESI cluster, all current pre-configured reference environments
(`ESI-202xa/b`) provide the respective most recent ACME version. These environments
can be easily started using the `ESI JupyterHub <https://jupyterhub.esi.local>`_

For Developers
^^^^^^^^^^^^^^

To install the latest development version of ACME in a fresh conda environment
(called `acme` by default), please follow these steps:

.. code-block:: bash

    git clone https://github.com/esi-neuroscience/acme.git
    cd acme/
    conda env create -f acme.yml
    pip install -e .

(note the dot "." at the end of the ``pip install`` command).

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
by running :func:`~acme.slurm_cluster_setup` (on an HPC cluster managed by the
`SLURM Workload Manager <https://slurm.schedmd.com/documentation.html>`_) or
:func:`~acme.local_cluster_setup` (on local multi-processing hardware)
**before** launching the actual calculation. For example,

.. code-block:: python

    from acme import slurm_cluster_setup

    slurmClient = slurm_cluster_setup(partition="some_partition",
                                      n_cores=2,
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

starts 10 concurrent SLURM workers in the `16GBXL` queue (no need to further
specify CPU core count or memory requirements).

.. note::
    Since ACME internally relies on `distributed <https://distributed.dask.org/en/stable/>`_
    and `dask_jobqueue <https://jobqueue.dask.org/en/latest/>`_ it can leverage
    any HPC infrastructure (CPU nodes, GPU nodes etc.) managed by SLURM, PBS,
    SGE, Moab etc. For users of the ESI HPC cluster ACME offers the above
    presented convenience function :func:`~acme.esi_cluster_setup`, however,
    the underlying general purpose setup routine :func:`acme.slurm_cluster_setup`
    (which is invoked by :func:`~acme.esi_cluster_setup`) can be used to
    start a parallel worker cluster on any distributed system controlled by SLURM.
    If you are interested in having a `*_cluster_setup` routine for your institution's
    HPC infrastructure being included in ACME, please open an issue in our
    `GitHub Issue Tracker <https://github.com/esi-neuroscience/acme/issues>`_.


More Information
^^^^^^^^^^^^^^^^

Details about optional arguments of :class:`~acme.ParallelMap` are summarized in
its constructor :meth:`~acme.ParallelMap.__init__` method. Additional usage examples
and tutorials are provided in the :doc:`User Walkthrough <userguide>`.
