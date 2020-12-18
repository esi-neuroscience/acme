.. ACME documentation master file, created by
   sphinx-quickstart on Fri Dec 18 12:23:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: ACME Documentation

Welcome to the Documentation of ACME
====================================
ACME (**A**\synchronous **C**\omputing **M**\ade **E**\asy, pronounced "ak-mee")
is a framework for calling Python functions in parallel using SLURM on the ESI
HPC cluster (local multi-processor hardware is supported as well). ACME is based
on the parallelization engine used in `SyNCoPy <http://www.syncopy.org/>`_ and
is itself part of the SyNCoPy project.

Getting Started
---------------
The :doc:`Quickstart Guide <quickstart>` covers installation and basic usage of ACME.
More information can be found in the docstrings of the respective classes and
functions, specifically, `ParallelMap` and `ACMEdaemon`.

Contact
-------
To report bugs or ask questions please use our `GitHub issue tracker <https://github.com/esi-neuroscience/acme/issues>`_.

Navigation
----------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Any sections to be included in the Documentation dropdown menu have to be in the toctree

.. toctree::
   :hidden:

   quickstart
   user/users.rst
   developer/developers.rst
