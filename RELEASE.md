<!--
Copyright (c) 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
in Cooperation with Max Planck Society
SPDX-License-Identifier: CC-BY-NC-SA-1.0
-->

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
   conda install dask "dask-jobqueue>=0.8" h5py numpy "tqdm>=4.31" pytest-cov ipdb ipython mypy "scipy>= 1.5,<2.0" tox
   ```

1. Run the test-suite locally

   ```bash
   cd /path/to/acme-repo/acme/tests
   ./run_tests.sh pytest
   ```

1. Update testing environment in [tox.ini](./tox.ini) and run tox locally
   (**IMPORTANT** this also tests if ACME can be installed via pip!)

   ```bash
   cd /path/to/acme-repo
   tox
   ```

1. Run mypy static type checker locally

   ```bash
   cd /path/to/acme-repo
   mypy acme --allow-redefinition
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
   ssh hub
   module load conda
   conda env create --file acmepy11.yml
   ```

1. Run ACME's test-suite on both architectures

   ```bash
   ssh {hub,esi-svhpc2}
   module load conda
   conda activate acme-py11{-ppc}
   cd /path/to/acme-repo/acme/tests
   ./run_tests.sh pytest
   ```

If all tests are passing, merge changes into ``[dev]`` branch.

## Deployment

Actual deployment of a new release requires several manual steps for publishing 
to PyPI and conda-forge, respectively. **Note**: these steps have to be performed
in the order shown here!

### 1. Publish to PyPI

> Ensure you're working in ``[dev]``, not ``[main]``!

1. Update dependencies/supported Python version in [setup.cfg](./setup.cfg)
1. Update build-system requirements in [pyproject.toml](./pyproject.toml)
1. Bump version number in [setup.py](./setup.py)
1. Prepare release notes in [CHANGELOG.md](./CHANGELOG.md)
1. Force-update environment file [acme.yml](./acme.yml) as well as citation
   resource [CITATION.cff](./CITATION.cff):

   ```bash
   python setup.py --version
   ```

1. Check proper licensing of all files (errors in [setup.py](./setup.py)
   and [CITATION.cff](./CITATION.cff) can be ignored)

   ```bash
   reuse lint
   ```

1. Create a release tag

   ```bash
   git tag -a "202x.y" -m "202x.y"
   git push --tags
   ```

Finally, open a PR into ``[main]``. Once merged, wait for the CI pipeline
to finish and click the play button to publish to PyPi. 

### 2. Publish to `conda-forge`

Wait for the `regro-cf-autotick-bot` to open an PR in ACME's 
[conda-forge feedstock](https://github.com/conda-forge/esi-acme-feedstock).
Check out the bot's fork of the feedstock into a separate directory and 
switch to the branch shown in the PR:

```bash 
git clone git@github.com:regro-cf-autotick-bot/esi-acme-feedstock.git /path/to/bot-forge/
git checkout -t origin/2025.1_hf4c7f2
```
Change dependencies as needed in `meta.yaml`. Take care (not) to bump
the `build/number` as explained in
[Updating esi-acme-feedstock](https://github.com/conda-forge/esi-acme-feedstock#updating-esi-acme-feedstock).
Run the Docker-based testing pipeline:

```bash
sudo ./build-locally.py
```

If the pipeline completes successfully, commit and push your changes. Once all 
GitHub actions are done, review and merge the bot's PR. 

## 3. GitHub Release

Create a new GitHub release based on the tag created in Step #1: go to 
[ACME's GitHub page](https://github.com/esi-neuroscience/acme) and click on 
*Releases*. Choose *Draft a new release* and pick the most recently created tag. 
Choose the tag's title as *Release title* and copy the corresponding section from 
`CHANGELOG.md` in the field saying *Describe this release* (use previous releases 
as template and click *Preview* to check markdown formatting). Scroll to the 
bottom of the page and ensure the box next to *Set as the latest release* is 
ticked. Finally, click on *Publish release*. 

## Post-Release Cleanup

1. Checkout ``[dev]`` branch
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
