# Releasing a New ACME Version

The instructions collected here are intended to help with preparing a
new ACME release. This file is mainly intended for internal use, for a
detailed guide on how to contribute to ACME, please see our
[Contributing Guide](https://esi-acme.readthedocs.io/en/latest/contributing.html)

## Prerequisites

1. On your development machine, set up a new conda environment with the
   most recent Python version intended to be supported

   ```bash
   conda create -n acme-py11 python=3.11
   ```

1. Update dependencies: open [setup.cfg](./setup.cfg) and install the
   most recent versions (not necessarily those stated) of all listed dependencies

   ```bash
   conda install dask dask-jobqueue h5py ...
   ```

1. Run the test-suite locally

   ```bash
   cd acme/tests
   ./run_tests.sh pytest
   ```

1. Export your environment, re-recreate it on the ESI HPC cluster and run
   ACME's cluster test-suite

   ```bash
   conda env export --from-history > acmepy11.yml
   scp acmepy11.yml esi-svhpc2:~/
   ssh esi-svhpc2
   module load conda
   conda env create --file acmepy11.yml
   ./run_tests.sh pytest
   ```

1. Coming soon..

## Deployment

## Post-Release Cleanup

1. Prepare next `[Unreleased]` section with pre-defined headings in
   `CHANGELOG.md` for next release:

   ```text
   ## [Unreleased]
   ### NEW
   ### CHANGED
   ### REMOVED
   ### DEPRECATED
   ### FIXED
   ```
