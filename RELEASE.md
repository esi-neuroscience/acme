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

1. Export your environment and re-recreate it on an x86 ESI HPC cluster node:

   ```bash
   conda env export --from-history > acmepy11.yml
   scp acmepy11.yml esi-svhpc2:~/
   ssh esi-svhpc2
   module load conda
   conda env create --file acmepy11.yml
   ```

   Create an identical environment (append "-ppc" to its name) on a ppc64le
   node:

   ```bash
   ssh HUB
   module load conda
   conda env create --file acmepy11.yml
   ```

1. Run ACME's test-suite on both architectures

   ```bash
   ssh {hub/esi-svhpc2}
   module load conda
   conda activate acme-py11{-ppc}
   cd acme/acme/tests
   ./run_tests.sh pytest
   ```

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
