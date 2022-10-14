.. ACME documentation master file, created by
   sphinx-quickstart on Fri Dec 18 12:23:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: ACME Documentation

.. image:: https://raw.githubusercontent.com/esi-neuroscience/acme/master/doc/source/_static/acme_logo.png
   :alt: ACME-Logo

Welcome to the Documentation of ACME
====================================
ACME (**A**\synchronous **C**\omputing **M**\ade **E**\asy, pronounced *"ak-mee"*)
is a framework for calling Python functions in parallel, primarily intended for
using SLURM on the ESI HPC cluster (local multi-processor hardware is supported as well).
ACME is built on top of the parallel computing library `dask <https://www.dask.org/>`_
and is used by `SyNCoPy <http://www.syncopy.org/>`_.

Getting Started
---------------
Please have a look at the :doc:`Quickstart Guide <quickstart>`.

Examples
--------
Simple to advanced usage examples our collected on our `GitHub Page <https://github.com/esi-neuroscience/acme#usage>`_.

User Guide
----------
A more elaborate tour illustrating how to use ACME in various scenarios is
provided in the :doc:`User Walkthrough <userguide>`. A short intro for how to use
ACME in your processing pipeline is provided in :doc:`Best Practices <best_practices>`.
Things don't go as planned? Take a look at :doc:`Troubleshooting + FAQ <troubleshooting_faq>`.

API Documentation
-----------------
The full source code documentation can be found in the :doc:`API Docs <api>`.

Contact
-------
To report bugs or ask questions please use our `GitHub Issue Tracker <https://github.com/esi-neuroscience/acme/issues>`_.

Navigation
----------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Any sections to be included in the Documentation dropdown menu have to be in the toctree

.. toctree::
   :hidden:

   quickstart
   userguide
   best_practices
   troubleshooting_faq
   advanced_usage
   api
