# ACME Test Pipeline

This directory contains ACME's test suite. Tests can be run either locally
or on HPC cluster nodes.

## Run Entire Suite

ACME comes with the convenience script [run_tests.sh](./run_tests.sh) to set up and
launch its test-suite. Launching it without arguments shows a brief usage
summary:

```bash
cd acme/acme/tests
./run_tests.sh
```

Pick an option and start testing, e.g.,

```bash
./run_tests.sh pytest
```

## Run Single Tests

If ACME is not installed in your `site-packages` directory, it has to be
added to your Python path first:

```bash
cd acme/acme/tests/
export PYTHONPATH=$(cd ../../ && pwd)
```

Then pick a testing function and run it, e.g., to execute the `test_simple_filter`
test, you can use

```bash
pytest -v test_pmap.py -k 'test_simple_filter'
```

Use pytest's extensive [command line arguments](https://docs.pytest.org/en/6.2.x/usage.html)
to accommodate your development workflow. For instance, to show ACME's
command line output, use the `-s` flag, to drop to
[pdb](https://docs.python.org/3/library/pdb.html) on failures, additionally
provide the `--pdb` option

```bash
pytest -sv test_pmap.py -k 'test_simple_filter' --pdb
```
