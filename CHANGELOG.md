# Changelog of ACME
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2rc3] - 2021-11-26
### NEW
- Upgraded dask version used by ACME (anything below 2021.12)
- Added macOS as officially supported platform. A corresponding CI job has been
  set up as well.

### CHANGED
- Updated `email` and `homepage` tags in setup.cfg to comply with new setuptools
  packaging standard

### FIXED
- NumPy arrays with singleton dimensions triggered a nondescript `TypeError` in
  `ACMEDaemon` due to incorrect indexing before broadcasting.

## [0.2rc2] - 2021-10-26
### CHANGED
- Updated versioning scheme to be PEP 440 compliant

### FIXED
- Updated dependency setup: dask 2.25/2.30 does not work with click 8.+ in a
  SLURM cluster context
- Refined query to check for active workers in a dask client: ensure workers
  are not only attached to a client but actually have resources available

## [v0.2rc1] - 2021-10-19
### NEW
- Included function `local_cluster_setup` to launch a local distributed Dask
  multi-processing cluster running on the host machine

### CHANGED
- Refined integration with [SyNCoPy](http://www.syncopy.org)

### FIXED
- Repaired auto-generated semantic version strings (use only release number + letter,
  remove local ".dev0" suffix from official release versions)

## [v0.2b] - 2021-08-04
### NEW
- Support for custom `sbatch` arguments (thanks to @KatharineShapcott)

### FIXED
- Made ID fetching of crashed SLURM jobs more robust
- Corrected faulty override of `print`/`showwarning` in case ACME was called
  from within SyNCoPy.
- Cleaned up fetching of SLURM worker memory
- Corrected keywords in CITATION.cff

## [v0.2a] - 2021-05-18
### NEW
- Made ACME PEP 517 compliant: added pyproject.toml and modified setup.py
  accordingly
- Added IBM POWER testing pipeline (via dedicated GitLab Runner)

### CHANGED
- New default SLURM partition set to "8GBXS" in `esi_cluster_setup`

### REMOVED
- Retired tox in `slurmtest` CI pipeline in favor of a "simple" pytest testing
  session due to file-locking problems of tox environments on NFS mounts

### FIXED
- Stream-lined GitLab Runner setup: use cluster-wide conda instead of local
  installations (that differ slightly across runners) and leverage `tox-conda`
  to fetch pre-built dependencies
- Opt-in pickling was not propagated correctly in daemon-reentry situations

## [v0.2] - 2021-05-05
### NEW
- New keyword `write_pickle` can be used to override HDF5 as default storage
  format in favor of pickle
- Included code-coverage information and corresponding requirements for pull
  requests in ACME repo
- Added software citation file `CITATION.cff`

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
- If auto-saving to HDF5 fails, a new "emergency pickling" mechanic kicks in and
  attempts to pickle the offending return values instead
- User-provided functions in custom modules are now correctly propagated
  by inheriting `sys.path` from the parent client
- Argument distribution is more memory efficient: input arguments are not
  held in memory by the scheduler and then propagated to workers anymore.
  Instead, arguments shared by all workers are broadcast to the cluster and
  referenced by the workers.
- Any user-issued `KeyboardInterrupt` (`CTRL` + `C` button press) is caught and
  triggers a graceful shutdown of all worker jobs managed by the current client
  (specifically, do not leave SLURM jobs detached from the client running in the
  background)
- Fixed progress bars that were left broken after an exception was raised

## [v0.1b] - 2021-01-15
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
