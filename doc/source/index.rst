.. ACME documentation master file, created by
   sphinx-quickstart on Fri Dec 18 12:23:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
.. in Cooperation with Max Planck Society

.. SPDX-License-Identifier: CC-BY-NC-SA-1.0

.. title:: ACME Documentation

.. image:: https://raw.githubusercontent.com/esi-neuroscience/acme/master/doc/source/_static/acme_logo.png
   :alt: ACME-Logo

|

.. image:: https://img.shields.io/conda/vn/conda-forge/esi-acme.svg
   :target: https://anaconda.org/conda-forge/esi-acme
   :alt: conda-version

.. image:: https://badge.fury.io/py/esi-acme.svg
   :target: https://badge.fury.io/py/esi-acme
   :alt: pypi-version

.. image:: https://img.shields.io/github/license/esi-neuroscience/acme
   :target: https://github.com/esi-neuroscience/acme/blob/main/LICENSE
   :alt: license

.. image:: https://bestpractices.coreinfrastructure.org/projects/7144/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/7144
   :alt: openssf


Welcome to the Documentation of ACME
====================================
ACME (**A**\synchronous **C**\omputing **M**\ade **E**\SI, pronounced *"ak-mee"*)
is a framework for calling Python functions in parallel, originally intended for
using SLURM on the ESI HPC cluster (local multi-processor hardware is supported as well).
ACME is built on top of the parallel computing library `dask <https://www.dask.org/>`_
and is used by `SyNCoPy <http://www.syncopy.org/>`_.

Getting Started
---------------
Please have a look at the :doc:`Quickstart Guide <quickstart>`.

Examples
--------
Simple to advanced usage examples are collected on our `GitHub Page <https://github.com/esi-neuroscience/acme#usage>`_.

User Guide
----------
Looking for a more elaborate tour illustrating how to use ACME in various scenarios? ⟶ :doc:`User Walkthrough <userguide>`

Want to include ACME in your processing pipeline? ⟶ :doc:`Best Practices <best_practices>`

Things don't go as planned? ⟶ :doc:`Troubleshooting + FAQ <troubleshooting_faq>`.

API Documentation
-----------------
The full source code documentation can be found in the :doc:`API Docs <api>`.

Contributing
------------
Contributions to ACME are always welcome! Please see :doc:`Contributing to ACME <contributing>`
for details.

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
   contributing
   api
