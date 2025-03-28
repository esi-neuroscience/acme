<!--
Copyright (c) 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
in Cooperation with Max Planck Society
SPDX-License-Identifier: CC-BY-NC-SA-1.0
-->

![ACME_logo](https://raw.githubusercontent.com/esi-neuroscience/acme/master/doc/source/_static/acme_logo.png)

# ACME: Asynchronous Computing Made ESI

[![conda](https://img.shields.io/conda/vn/conda-forge/esi-acme.svg)](https://anaconda.org/conda-forge/esi-acme)
[![pypi](https://badge.fury.io/py/esi-acme.svg)](https://badge.fury.io/py/esi-acme)
[![license](https://img.shields.io/github/license/esi-neuroscience/acme)](https://github.com/esi-neuroscience/acme/blob/main/LICENSE)
[![Open in Visual Studio Code](https://img.shields.io/badge/-Open_in_VS_Code-007ACC?logo=visual%20studio%20code&logoColor=ffffff)](https://github.dev/esi-neuroscience/acme)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/7144/badge)](https://bestpractices.coreinfrastructure.org/projects/7144)
[![REUSE status](https://api.reuse.software/badge/github.com/esi-neuroscience/acme)](https://api.reuse.software/info/github.com/esi-neuroscience/acme)

main: [![tests](https://github.com/esi-neuroscience/acme/actions/workflows/tests_workflow.yml/badge.svg?branch=main)](https://github.com/esi-neuroscience/acme/actions/workflows/tests_workflow.yml)
[![codecov](https://codecov.io/gh/esi-neuroscience/acme/branch/main/graph/badge.svg?token=LCB2RPBQJG)](https://codecov.io/gh/esi-neuroscience/acme)

dev: [![tests](https://github.com/esi-neuroscience/acme/actions/workflows/tests_workflow.yml/badge.svg?branch=dev)](https://github.com/esi-neuroscience/acme/actions/workflows/tests_workflow.yml)
[![codecov](https://codecov.io/gh/esi-neuroscience/acme/branch/dev/graph/badge.svg?token=LCB2RPBQJG)](https://codecov.io/gh/esi-neuroscience/acme)

## Table of Contents

1. [Summary](#summary)
1. [Installation](#installation)
1. [Usage](#usage)
   - [Basic Examples](#basic-examples)
   - [Intermediate Examples](#intermediate-examples)
   - [Advanced Use](#advanced-use)
1. [Handling Results](#handling-results)
   - [Load Results From Files](#load-results-from-files)
   - [Collect Results in Single HDF5 Dataset](#collect-results-in-single-hdf5-dataset)
   - [Collect Results in Local Memory](#collect-results-in-local-memory)
1. [Debugging](#debugging)
1. [Documentation and Contact](#documentation-and-contact)
   - [Resources](#resources)

## Summary

The objective of ACME (pronounced *"ak-mee"*) is to provide easy-to-use
wrappers for calling Python functions concurrently ("embarassingly parallel workloads").
ACME is developed at the
[Ernst Strüngmann Institute (ESI) gGmbH for Neuroscience in Cooperation with Max Planck Society](https://www.esi-frankfurt.de/>)
and released free of charge under the
[BSD 3-Clause "New" or "Revised" License](https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_(%22BSD_License_2.0%22,_%22Revised_BSD_License%22,_%22New_BSD_License%22,_or_%22Modified_BSD_License%22)).
ACME relies heavily on the concurrent processing library [dask](https://docs.dask.org/en/latest/>)
and was primarily designed to facilitate the use of [SLURM](https://slurm.schedmd.com/documentation.html)
on the ESI HPC cluster (although other HPC infrastructure running SLURM can be
leveraged as well). Local multi-processing hardware (i.e., multi-core CPUs)
is fully supported too. ACME is itself used as the parallelization engine of [SyNCoPy](http://www.syncopy.org/).

![](https://github.com/esi-neuroscience/acme/blob/main/doc/source/_static/acme_demo.gif)

## Installation

ACME can be installed with `pip`

```shell
pip install esi-acme
```

or via `conda`

```shell
conda install -c conda-forge esi-acme
```

To get the latest development version, simply clone our GitHub repository:

```shell
git clone https://github.com/esi-neuroscience/acme.git
cd acme/
pip install -e .
```

## Usage

### Basic Examples

Simplest use, everything is done automatically.

```python
from acme import ParallelMap

def f(x, y, z=3):
  return (x + y) * z

with ParallelMap(f, [2, 4, 6, 8], 4) as pmap:
  pmap.compute()
```

See also our [Quickstart Guide](https://esi-acme.readthedocs.io/en/latest/quickstart.html).

### Intermediate Examples

Set number of function calls via `n_inputs`

```python
import numpy as np
from acme import ParallelMap

def f(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return (sum(x) + y) * z * w.max()

pmap = ParallelMap(f, [2, 4, 6, 8], [2, 2], z=np.array([1, 2]), w=np.ones((8, 1)), n_inputs=2)

with pmap as p:
  p.compute()
```

More details in
[Override Automatic Input Argument Distribution](https://esi-acme.readthedocs.io/en/latest/userguide.html#override-automatic-input-argument-distribution)

### Advanced Use

Allocate custom `client` object and recycle it for several computations
(use `slurm_cluster_setup` on non-ESI HPC infrastructure or `local_cluster_setup`
when working on your local machine)

```python
import numpy as np
from acme import ParallelMap, esi_cluster_setup

def f(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return (sum(x) + y) * z * w.max()

def g(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return (max(x) + y) * z * w.sum()

n_workers = 200
client = esi_cluster_setup(partition="8GBXS", n_workers=n_workers)

x = [2, 4, 6, 8]
z = range(n_workers)
w = np.ones((8, 1))

pmap = ParallelMap(f, x, np.random.rand(n_workers), z=z, w=w, n_inputs=n_workers)
with pmap as p:
    p.compute()

pmap = ParallelMap(g, x, np.random.rand(n_workers), z=z, w=w, n_inputs=n_workers)
with pmap as p:
    p.compute()
```

For more information see [Reuse Worker Clients](https://esi-acme.readthedocs.io/en/latest/userguide.html#reuse-worker-clients)

## Handling Results

### Load Results From Files

By default, results are saved to disk in HDF5 format and can be accessed using
the `results_container` attribute of `ParallelMap`:

```python
def f(x, y, z=3):
  return (x + y) * z

with ParallelMap(f, [2, 4, 6, 8], 4) as pmap:
  filenames = pmap.compute()
```

Example loading code:

```python
import h5py
import numpy as np
out = np.zeros((4,))

with h5py.File(pmap.results_container, "r") as h5f:
  for k, key in enumerate(h5f.keys()):
    out[k] = h5f[key]["result_0"][()]
```

See also [Where Are My Results?](https://esi-acme.readthedocs.io/en/latest/userguide.html#where-are-my-results)

### Collect Results in Single HDF5 Dataset

If possible, results can be slotted into a single HDF5 dataset using the
`result_shape` keyword (`None` denotes the dimension for stacking results):

```python
def f(x, y, z=3):
  return (x + y) * z

with ParallelMap(f, [2, 4, 6, 8], 4, result_shape=(None,)) as pmap:
  pmap.compute()
```

Example loading code:

```python
import h5py

with h5py.File(pmap.results_container, "r") as h5f:
  out = h5f["result_0"][()] # returns a NumPy array of shape (4,)
```

Datasets support "unlimited" dimensions that do not have to be set a priori
(use `np.inf` in `result_shape` to denote a dimension of arbitrary size)

```python
# Assume only the channel count but not the number of samples is known
nChannels = 10
nSamples = 1234
mock_data = np.random.rand(nChannels, nSamples)
np.save("mock_data.npy", mock_data)

def mock_processing(val):
  data = np.load("mock_data.npy")
  return val * data

with ParallelMap(mock_processing, [2, 4, 6, 8], result_shape=(None, nChannels, np.inf)) as pmap:
  pmap.compute()

with h5py.File(pmap.results_container, "r") as h5f:
  out = h5f["result_0"][()] # returns a NumPy array of shape (4, nChannels, nSamples)
```

More examples can be found in
[Collect Results in Single Dataset](https://esi-acme.readthedocs.io/en/latest/userguide.html#collect-results-in-single-dataset)

### Collect Results in Local Memory

This is possible but not recommended.

```python
def f(x, y, z=3):
  return (x + y) * z

with ParallelMap(f, [2, 4, 6, 8], 4, write_worker_results=False) as pmap:
  result = pmap.compute() # returns a 4-element list
```

Alternatively, create an in-memory NumPy array

```python
with ParallelMap(f, [2, 4, 6, 8], 4, write_worker_results=False, result_shape=(None,)) as pmap:
  result = pmap.compute() # returns a NumPy array of shape (4,)
```

## Debugging

Use the `debug` keyword to perform all function calls in the local thread of
the active Python interpreter

```python
def f(x, y, z=3):
  return (x + y) * z

with ParallelMap(f, [2, 4, 6, 8], 4, z=None) as pmap:
    results = pmap.compute(debug=True)
```

This way tools like `pdb` or ``%debug`` IPython magics can be used.
More information can be found in the [FAQ](https://esi-acme.readthedocs.io/en/latest/troubleshooting_faq.html).

## Documentation and Contact

To report bugs or ask questions please use our
[GitHub issue tracker](https://github.com/esi-neuroscience/acme/issues).
More usage details and background information is available in our
[online documentation](https://esi-acme.readthedocs.io).

### Resources

- [ACME Presentation](https://pantaray.github.io/RSE23-talk/)
  at [deRSE23 - Conference for Research Software Engineering in Germany](https://de-rse23.sciencesconf.org/)
- [ACME Demo](https://esi-neuroscience.github.io/acme-demo/) presented at the 4th annual Data Scientist Community Meeting
- [ACME Tutorials](https://esi-acme.readthedocs.io/en/latest/tutorials.html)
- [ACME FAQ](https://esi-acme.readthedocs.io/en/latest/troubleshooting_faq.html)
