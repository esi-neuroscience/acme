 <!--
 Copyright (c) 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
 in Cooperation with Max Planck Society
 SPDX-License-Identifier: CC-BY-NC-SA-1.0
 -->

# Changelog of ACME
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [2025.6]
Performance improvements, bugfixes and dependency updates. 

### NEW
- New (optional) keyword `mem_cushion` in `slurm_cluster_setup` controls how 
  much memory to take off `mem_per_worker` to stay clear of memory limits defined 
  by QoS rules or partition definitions. The helper functions `esi_cluster_setup` 
  and `bic_cluster_setup` have been modified to use default values propagated to 
  `slurm_cluster_setup` tested on the respective clusters. 

### CHANGED
- Modified heuristics to allocate CPUs given a worker's memory demands to optimize 
  multi-threading performance on both ESI and CoBIC clusters. 

### FIXED
- Made type-hint of `_probe_mem_spec` Python 3.9-compliant (cf. #64)
- Repaired blocking of cross-architecture job submission on the CoBIC HPC cluster 
- Updated tests to account for changes in SLURM setup on both ESI and CoBIC HPC 
  clusters and fixed some lurking bugs 

## [2025.3]
Included new convenience function `bic_cluster_setup` for the HPC cluster at
CoBIC Frankfurt. Analogous to the similarly named helper function built for 
the ESI HPC cluster, `bic_cluster_setup` simplifies creating a Dask parallel 
computing client. For instance, the following command transparently launches 10 
SLURM workers in the `8GBSppc` partition:

```python
client = bic_cluster_setup(n_workers=10, partition="8GBSppc")
```

Additionally, ACME's automatic partition selection has been extended to also support 
workloads running on the CoBIC HPC cluster. Similarly, all customization settings 
supported by `esi_cluster_setup` are also available in `bic_cluster_setup`, e.g., 
`cores_per_worker` can be used together with `mem_per_worker`
and `job_extra` to create specialized computing clients custom-tailored
to specific workload requirements, e.g.,

```python
client = bic_cluster_setup(n_workers=10,
                           cores_per_worker=3,
                           mem_per_worker="12GB",
                           job_extra=["--job-name='myjob'"],
                           partition="32GBSppc")
```

More information can be found in ACME's [online documentation](https://esi-acme.readthedocs.io/en/latest/)

### NEW
- New convenience function `bic_cluster_setup` to streamline managing Dask 
  parallel computing clients on the CoBIC HPC cluster. 
- Added two new (optional) keywords to `slurm_cluster_setup`: `worker_extra_args` 
  can be used to pass additional options for configuring Dask workers. Similarly, 
  `scheduler_options` propagates custom settings to the Dask scheduler. 
- New helper function `is_bic_node` determines if ACME is running on the CoBIC
  HPC cluster
- New helper function `get_interface` finds the name of the network interface 
  associated to a given IP address
- New helper function `get_free_port` finds the lowest open port in a given range 

### CHANGED
- Changed default partition-type from "XS" to "S" on the ESI HPC cluster when 
  letting ACME automatically choose a partition 
- Updated testing setup (use centralized [pytest.ini](./acme/tests/pytest.ini) 
  configuration to not pollute tests with duplicate `PYTEST_ADDOPTS` exports
- Modernized the convenience script [run_tests.sh](./acme/tests/run_tests.sh): 
  the script can now process arbitrary pytest options (run single tests, drop 
  to PDB on error etc.)

### REMOVED
- Support for the deprecated keywords `n_jobs`, `mem_per_job`, `n_jobs_startup` 
  and `workers_per_job` has been removed. Code that still uses these keywords 
  has to be modified to replace them with their corrresponding counterparts
  `n_workers`, `mem_per_worker`, `n_workers_startup` and `processes_per_worker`, 
  respectively. 

### FIXED
- Adapted helper script `run_tests.sh` to use SLURM defaults when running on 
  unknown HPC clusters
  
## [2025.1]
Implementation of user's feature request: ACME can now allocate result datasets 
with arbitrary dimensions via the `result_shape` keyword. In case it is not clear 
(or cumbersome) to determine the shape of an aggregate results dataset a-priori,
setting the appropriate dimension(s) to `np.inf` prompts ACME to create a 
[resizable HDF5 dataset](https://docs.h5py.org/en/stable/high/dataset.html#resizable-datasets). 

### NEW
- Added support for "unlimited" datasets to allow flexible dimension 
  specifications in `result_shape`. When setting the size of a dimension in 
  `result_shape` to `np.inf`, ACME allocates a resizable HDF5 dataset for the 
  results. This works for both virtual and regular datasets. 
  
### REMOVED
- As announced in the previous release the `start_client` keyword has been removed
  from `local_cluster_setup` (starting a dask `LocalCluster` always starts
  a client anyway)

### DEPRECATED
- Dropped support for Windows (ACME *should* work but is not tested any more)
- Dropped support for Python 3.7

### FIXED
- Custom resource allocations were not correctly propagated to dask
  workers (especially in the "E880" partition on the ESI cluster). This
  has been fixed (cf #60)
- A bug in `python-msgpack` under Python 3.12 triggered de-serialization
  errors; temporarily pinned `python-msgpack` to version 1.0.5 but newer 
  versions do not exhibit this problem (cf #59)

## [2023.12] - 2023-12-6
Better support for non-x86 micro-architectures. On the ESI HPC cluster,
the convenience function `esi_cluster_setup` now transparently works with the
local `"E880"` partition comprising our IBM POWER E880 servers. Similar to
the x86 nodes, a simple

```python
client = esi_cluster_setup(n_workers=10, partition="E880")
```

is enough to launch ten SLURM workers each equipped with four POWER8 cores
and 16 GB RAM by default. Similarly, ACME's automatic partition selection has been
extended to also support workloads running inside the `"E880"` partition.
Nonetheless, `esi_cluster_setup` did not only get simpler to use but now also
comes with more (still completely optional) customization settings:
the new keyword `cores_per_worker` can be used together with `mem_per_worker`
and `job_extra` to create specialized computing clients custom-tailored
to specific workload requirements, e.g.,

```python
client = esi_cluster_setup(n_workers=10,
                           cores_per_worker=3,
                           mem_per_worker="12GB",
                           job_extra=["--job-name='myjob'"],
                           partition="E880")
```

For more see [Advanced Usage and Customization](https://esi-acme.readthedocs.io/en/latest/advanced_usage.html)

### NEW
- New keyword `cores_per_worker` in `esi_cluster_setup` to explicitly set
  the core-count of SLURM workers.
- Extended functionality of ACME's partition auto-selection on the ESI
  HPC cluster to include IBM POWER machines in the "E880" partition
- Added new "Tutorials" section in documentation
- Added new tutorial on using ACME for parallel evaluation of classifier
  accuracy (Thanks to @timnaher, cf #53)
- Added new tutorial on using ACME for parallel neural net model evaluation
  (Thanks to @timnaher, cf #53)
- Added type-hints following PEP 484 to support static code analyzers
  (e.g., `mypy`) and clarify type conventions in internal functions with
  "sparse" docstrings.

### CHANGED
- To avoid dubious (and hard to debug) errors, `esi_cluster_setup` now
  checks the micro-architecture of the submitting host against the chosen
  partition. This avoids accidental start attempts of ppc64le SLURM jobs
  from inside an x86_64 Python interpreter and vice versa.

### REMOVED
- The `partition` keyword in `esi_cluster_setup` does not have a default
  value any more (the old default of "8GBXS" was inappropriate most of
  the time)
- The (undocumented) "anonymous" keyword `n_cores` of `esi_cluster_setup`
  has been removed in favor of the explicit `cores_per_worker` (now also
  visible in the API). Just like `n_cores`, setting the new `cores_per_worker`
  parameter is still optional: by default, `esi_cluster_setup` derives
  core-count from `DefMemPerCPU` and the chosen value of `mem_per_worker`.
- In `slurm_cluster_setup`, do not use `DefMemPerCPU` as fallback substitute
  in case `MaxMemPerCPU` is not defined for chosen partition (may be overly
  restrictive on requested memory settings)

### DEPRECATED
- Using `start_client` in `local_cluster_setup` does not have any effect
  any more: starting a dask `LocalCluster` always starts a client.

### FIXED
- fixed partition bug ``run_tests.sh`` (Thanks to @timnaher, cf #53)
- simplified and fixed interactive user queries: use the builtin `select`
  module in everything but Jupyter and rely on the `input` module inside
  notebooks.
- clarified docstring discussing `result_dtype`: must not be `None` but
  `str` (still defaults to "float")
- numerous corrections of errata/outdated information in docstrings

## [2023.4] - 2023-04-14
Re-designed ACME's logs and command line output.

### NEW
- Created templates for filing issues and opening Pull Requests for ACME
  on GitHub.
- Enabled private security reporting in ACME's GitHub repository and
  added a security policy for ACME (in compliance with the OpenSSF Best
  Practices Badge)

### CHANGED
- Overhauled ACME's logging facilities: many print messages have been
  marked `"DEBUG"` to make ACME's default output less "noisy". To this
  effect the Python `logging` module is now used more extensively than
  before. The canonical name of ACME's logger is simply "ACME".
- By default, ACME now creates a log-file alongside any auto-generated
  output files to keep a record of file creation and attribution.
- Reworked ACME's SyNCoPy interface: a dedicated module `spy_interface.py`
  is now managing ACME's I/O direction if ACME is called by SyNCoPy. This
  allows for (much) cleaner exception handling in ACME's cluster helpers
  (`esi_cluster_setup`, `cluster_cleanup` etc.) which ultimately permits
  a more streamlined extension of ACME to more HPC infrastructure.
- Redesigned ACME's online documentation: increased font-size to enhance
  readability, included a contribution guide and reworked the overall page
  navigation + visual layout.

### FIXED
- Prevented ACME from accidentally using a dysfunctional client (e.g., a
  SLURM client with workers whose jobs have been externally cancelled).
  Thanks to @KatharineShapcott, cf #47

## [2022.12] - 2022-12-15
Bugfix release.

### CHANGED
- If not provided, a new lower default value of one is used for `n_workers_startup`

### FIXED
- Updated memory estimation logic on the ESI HPC cluster: if ACME does not
  handle result output distribution but memory estimation is still requested
  do not perform `memEstRun` keyword injection.

## [2022.11] - 2022-11-11
Major changes in managing auto-generated files
- If `write_worker_results` is `True`, ACME now creates an aggregate results
  container comprised of external links that point to actual data in HDF5
  payload files generated by parallel workers.
- Optionally, results can be slotted into a single dataset/array (via the
  `result_shape` keyword).
- If `single_file` is `True`, ACME stores results of parallel compute runs
  not in dedicated payload files but all workers write to a single aggregate
  results container.
- By providing `output_dir`, the location of auto-generated HDF5/pickle files can
  be customized
- Entities in a distributed computing client that concurrently process tasks
  are now consistently called "workers" (in line with dask terminology).
  Accordingly the keywords `n_jobs`, `mem_per_job`, `n_jobs_startup` and
  `workers_per_job` have been renamed `n_workers`, `mem_per_worker`,
  `n_workers_startup` and `processes_per_worker`, respectively. To ensure
  compatibility  with existing code, the former names have been marked
  deprecated but were not removed and are still functional.

A full list of changes is provided below

### NEW
- Included keyword `output_dir` in `ParallelMap` that allows to customize the
  storage location of files auto-generated by ACME (HDF5 and pickle). Only
  effective if `write_worker_results` is `True`.
- Added keyword `result_shape` in `ParallelMap` to permit specifying the
  shape of an aggregate dataset/array that results from all computational runs
  are slotted into. In conjunction with the shape specification, the new keyword
  `result_dtype` offers the option to control the numerical type (set to
  "float64" by default) of the resulting dataset (if `write_worker_results = True`)
  or array (`write_worker_results = False`). On-disk dataset results collection
  is only available for auto-generated HDF5 containers (i.e, `write_pickle = False`)
- Introduced keyword `single_file` in `ParallelMap` to control, whether parallel
  workers store results of computational runs in dedicated HDF5 files (`single_file = False`,
  default) or share a single results container for saving (`single_file = True`).
  This option is only available for auto-generated HDF5 containers, pickle
  files are not supported (i.e., `write_worker_results = True` and
  `write_pickle = False`).
- Included options to specify worker count and memory consumption in `local_cluster_setup`
- Added a new section "Advanced Usage and Customization" in the online documentation
  that discusses settings and associated technical details
- Added support for Python 3.10 and updated dask dependencies

### CHANGED
- Modified employed terminology throughout the package: to clearly delineate
  the difference between compute runs and worker processes (and to minimize
  friction between the documentation of ACME and dask), the term "worker"
  is now consistently used throughout the code base. If ACME is running on a
  SLURM cluster, a dask "worker" corresponds to a SLURM "job".
- In line with the above change, the following input arguments have been
  renamed:
  - in `ParallelMap`:
    - `n_jobs` -> `n_workers`
    - `mem_per_job` -> `mem_per_worker`
  - in `esi_cluster_setup` and `slurm_cluster_setup`:
    - `n_jobs` -> `n_workers`
    - `mem_per_job` -> `mem_per_worker`
    - `n_jobs_startup` -> `n_workers_startup`
  - in `slurm_cluster_setup`:
    - `workers_per_job` -> `processes_per_worker`
- Made `esi_cluster_setup` respect already running clients so that new parallel
  computing clients are not launched on top of existing ones (thanks to @timnaher)
- Introduced support for positional/keyword arguments of unit-length in
  `ParallelMap` so that `n_inputs` can be used as scaling parameter to launch
  `n_inputs` calls of a user-provided function
- All docstrings and the online documentation have been re-written (and in
  parts clarified) to account for the newly introduced features.
- Code coverage is not computed by a GitHub action workflow but is now
  calculated by the GitLab CI job that invokes SLURM to run tests on the
  ESI HPC cluster.

### DEPRECATED
The keywords `n_jobs`, `mem_per_job`, `n_jobs_startup` and `workers_per_job`
have been renamed. Using these keywords is still supported but raises a
`DeprecationWarning`.
- The keywords `n_jobs` and `mem_per_job` in both `ParallelMap` and
  `esi_cluster_setup` are deprecated. To specify the number of parallel
  workers and their memory resources, please use `n_workers` and  `mem_per_worker`,
  respectively (see corresponding item in the Section CHANGED above)
- The keyword `n_jobs_startup` in `esi_cluster_setup` is deprecated. Please
  use `n_workers_startup` instead

### FIXED
- Updated dependency versions (pin `click` to version < 8.1) and fixed Syncopy
  compatibility (increase recursion depth of input size estimation to one
  million calls)
- Streamlined dryrun stopping logic invoked if user chooses to not continue
  with the computation after performing a dry-run
- Modified tests that are supposed to use an existing distributed computing
  client to not shut down that very client
- Updated memory estimation routine to deactivate auto-generation of results
  files to not accidentally corrupt pre-allocated containers before launching
  the actual concurrent computation

## [2022.8] - 2022-08-05
Bugfixes, new automatic ESI-HPC SLURM partition selection, expanded Python version
compatibility and updated dependencies as well as online documentation overhaul.

### NEW
- On the ESI HPC cluster, using `partition="auto"` in `ParallelMap` now launches
  a heuristic  automatic SLURM partition selection algorithm (instead of simply
  falling back to the "8GBXS" partition on the ESI HPC cluster)

### CHANGED
- Updated package dependencies (allow `h5py` ver 3.x) and expanded support for
  recent Python versions (include 3.9)
- Restructured and expanded online documentation based on suggestions from @naehert:
  moved most examples and usage notes from `ParallelMap`'s docstring to dedicated
  docu pages and added new "Troubleshooting + FAQ" site.

### FIXED
- Repeated `ParallelMap` calls ignored differing `logfile` specifications. This
  has been corrected. In addition, the logging setup routine now ensures that only
  one `FileHandler` is used (any existing non-default log-file locations are
  removed from the logger to avoid generating multiple logs and/or accidentally
  appending to existing logs from previous runs).

## [2022.7] - 2022-07-06
Bugfixes, new versioning scheme and updated dependencies.

### CHANGED
- Modified versioning scheme: use date-based version tags instead of increasing
  numbers
- Updated `dask`, `dask-jobqueue` and `scipy` dependency requirements
- Removed any mentions of "hpx" from the code after upgrading the main file-server
  of the ESI HPC cluster

### FIXED
- Repaired broken FQDN detection in `is_esi_node`

## [0.21] - 2022-03-01
Performance improvements, new `dryrun` keyword and preparations for deploying
ACME on other clusters

### NEW
- Re-designed cluster startup code: added new function `slurm_cluster_setup` that
  includes SLURM-specific (but ESI-agnostic) code for spinning up a `SLURMCluster`
- Included new `dryrun` keyword in `ParallelMap` to test-drive ACME's automatically
  generated argument lists simulating a single (randomly picked) worker call prior
  to the actual concurrent computation (addresses #39)
- Added helper function `is_esi_node` to determine if ACME is running on the ESI
  HPC cluster

### CHANGED
- Do not parse scalars using `numbers.Number`, use `numpy.number` instead to
  catch Boolean values
- Included `conda clean` in CD pipeline to avoid disk fillup by unused conda
  packages/cache

### DEPRECATED
- Retired `conda2pip` in favor of the modern setup.cfg dependency management
  system. ACME's dependencies are now listed in setup.cfg which is used to
  populate the conda environment file acme.yml at setup time.
- Retired travis CI tests since free test runs are exhausted. Migrated to GitHub
  actions (and re-included codecov)

### FIXED
- On the ESI HPC cluster set the job CPU count depending on the chosen partition
  if not explicitly provided by the user (one core per 8GB of RAM, e.g., jobs in
  a 32GB RAM partition now use 4 cores instead of just one)

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
