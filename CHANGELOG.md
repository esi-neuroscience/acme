# Changelog of ACME
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### CHANGED
- Changed job submission system: instead of using dask bags, input arguments
  are directly propagated using dask-client methods. This has the side-effect
  that the layout of in-memory results changed: instead of returning a nested
  lists of lists, the user namespace is populated with a plain list of objects
  (simplifying result handling in the process)
### DEPRECATED
- In-memory list-of-list returns are not supported anymore; `ParallelMap` now
  returns plain (non-nested) lists.
### FIXED
- User-provided functions in custom modules are now correctly propagated
  by inheriting `sys.path` from the parent client
- Argument distribution is more memory efficient: input arguments are not
  held in memory by the scheduler and then propagated to workers anymore.
  Instead, arguments shared by all workers are broadcast to the cluster and
  referenced by the workers.

## [v0.1b] - 2020-01-15
### NEW
- This CHANGELOG file

### CHANGED
- Modified dependencies to not include Jupyter-related packages

### FIXED
- Fixed markdown syntax and URLs
- Fixed CI pipelines and repaired `h5py` version mismatch in dependencies
- Pin ACME to Python 3.8.x due to various packages not working properly
  (yet) in Python 3.9

## [v0.1a] - 2020-12-30
### NEW
- Initial ACME pre-release on PyPI

### CHANGED
- Made ACME GitHub repository public
