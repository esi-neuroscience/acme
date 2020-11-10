# ACME: Asynchronous Computing Made Easy
Asynchronous Computing Made Easy

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
Set no. of function calls via `n_inputs`
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
Allocate custom `client` object and re-cycle it for several computations
```python
import numpy as np
from acme import ParallelMap, esi_cluster_setup

def f(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return (sum(x) + y) * z * w.max()

def g(x, y, z=3, w=np.zeros((3, 1)), **kwargs):
    return (max(x) + y) * z * w.sum()

n_jobs = 200
client = esi_cluster_setup(partition="16GB", n_jobs=n_jobs)

x = [2, 4, 6, 8]
z = range(n_jobs)
w = np.ones((8, 1))

pmap = ParallelMap(f, x, np.random.rand(n_jobs), z=z, , w=w, n_inputs=n_jobs)
with pmap as p:
    p.compute()

pmap = ParallelMap(g, x, np.random.rand(n_jobs), z=z, , w=w, n_inputs=n_jobs)
with pmap as p:
    p.compute()
```


