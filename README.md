# ACME: Asynchronous Computing Made Easy

main: [![Build Status](https://travis-ci.com/esi-neuroscience/acme.svg?branch=main)](https://travis-ci.com/esi-neuroscience/acme)
dev: [![Build Status](https://travis-ci.com/esi-neuroscience/acme.svg?branch=dev)](https://travis-ci.com/esi-neuroscience/acme)

## Summary

The objective of ACME (pronounced *"ak-mee"*) is to provide easy-to-use
wrappers for calling Python functions in parallel ("embarassingly parallel workloads").
ACME is developed at the
[Ernst Strüngmann Institute (ESI) gGmbH for Neuroscience in Cooperation with Max Planck Society](https://www.esi-frankfurt.de/>)
and released free of charge under the
[BSD 3-Clause "New" or "Revised" License](https://en.wikipedia.org/wiki/BSD_licenses#3-clause_license_(%22BSD_License_2.0%22,_%22Revised_BSD_License%22,_%22New_BSD_License%22,_or_%22Modified_BSD_License%22)).
ACME relies on the concurrent processing library [Dask](https://docs.dask.org/en/latest/>)
and was primarily designed to facilitate the use of [SLURM](https://slurm.schedmd.com/documentation.html)
on the ESI HPC cluster. However, local multi-processing hardware (i.e., multi-core CPUs)
is fully supported as well. ACME is based on the parallelization engine used in [SyNCoPy](http://www.syncopy.org/) and
is itself part of the SyNCoPy package.

## Installation

ACME can be installed with pip

```
pip install esi-acme
```
To get the latest development version, simply clone our GitHub repository:
```
git clone https://github.com/esi-neuroscience/acme.git
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

### Advanced Use

Allocate custom `client` object and recycle it for several computations

```python
import numpy as np
from acme import ParallelMap, esi_cluster_setup

def f(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return (sum(x) + y) * z * w.max()

def g(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return (max(x) + y) * z * w.sum()

n_jobs = 200
client = esi_cluster_setup(partition="8GBXS", n_jobs=n_jobs)

x = [2, 4, 6, 8]
z = range(n_jobs)
w = np.ones((8, 1))

pmap = ParallelMap(f, x, np.random.rand(n_jobs), z=z, w=w, n_inputs=n_jobs)
with pmap as p:
    p.compute()

pmap = ParallelMap(g, x, np.random.rand(n_jobs), z=z, w=w, n_inputs=n_jobs)
with pmap as p:
    p.compute()
```

## Handling results

### Load results from files

The results are saved to disk in HDF5 format and the filenames are returned as a list of strings.

```python
with ParallelMap(f, [2, 4, 6, 8], 4) as pmap:
  filenames = pmap.compute()
```

Example loading code:

```python
out = np.zeros((4))
import h5py
for ii, fname in enumerate(filenames):
    with h5py.File(fname, 'r') as f:
        out[ii] = np.array(f['result_0'])
```

### Collect results in local memory

This is possible but not recommended.

```python
with ParallelMap(f, [2, 4, 6, 8], 4, write_worker_results=False) as pmap:
  results = pmap.compute() # returns a list of outputs
```

## Debugging

Use the `debug` keyword to perform all function calls in the local thread of
the active Python interpreter

```python
with ParallelMap(f, [2, 4, 6, 8], 4, z=None) as pmap:
    results = pmap.compute(debug=True)
```

This way tools like `pdb` or ``%debug`` IPython magics can be used.

## Documentation and Contact

To report bugs or ask questions please use our
[GitHub issue tracker](https://github.com/esi-neuroscience/acme/issues).
More usage details and background information is available in our
[online documentation](https://esi-acme.readthedocs.io/en/latest/).
