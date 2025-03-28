<!--
Copyright (c) 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
in Cooperation with Max Planck Society
SPDX-License-Identifier: CC-BY-NC-SA-1.0
-->

# ACME Test Pipeline

This directory contains ACME's test suite. Tests can be run either locally
or on HPC cluster nodes.

## Run Entire Suite

If ACME is installed in your `site-packages`, you can run all tests directly
from your Python interpreter

```python
import pytest
pytest_args = ["-v", "--pyargs", "acme"]
pytest.main(pytest_args)
```

If you're working on a local clone of ACME, the convenience script
[run_tests.sh](./run_tests.sh) can take care of setting up and
launching ACME's test-suite. Running the script without arguments shows
a brief usage summary:

```bash
cd acme/acme/tests
./run_tests.sh
```

Pick an option and start testing, e.g.,

```bash
./run_tests.sh pytest
```

## Run Single Tests

If ACME is installed in your `site-packages`, simply use the following
code snippet to run a single test, e.g., `test_simple_filter`

```python
import pytest
pytest_args = ["-v", "--pyargs", "acme", "-k", "test_simple_filter"]
pytest.main(pytest_args)
```

If ACME is **not** installed in your `site-packages` directory, you can use 
[run_tests.sh](./run_tests.sh) with appropriate arguments, e.g., 

``` bash
./run_tests.sh pytest test_pmap.py -k 'test_simple_filter'
```

Use pytest's extensive [command line arguments](https://docs.pytest.org/en/6.2.x/usage.html)
to accommodate your development workflow. For instance, to show ACME's
command line output, use the `-s` flag, to drop to
[pdb](https://docs.python.org/3/library/pdb.html) on failures, additionally
provide the `--pdb` option

```bash
./run_tests.sh pytest -s test_pmap.py -k 'test_simple_filter' --pdb
```
