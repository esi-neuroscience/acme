#!/bin/bash
#
# Some quick shortcuts to quickly test ACME's proper functionality
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# First and foremost, check if `srun` is available
_useSLURM=$(command -v srun)

# Stuff only relevant in here
_self=$(basename "$BASH_SOURCE")
_selfie="${_self%.*}"
_ppname="<$_selfie>"

# Brief help message explaining script usage
usage()
{
    echo "
usage: $_selfie COMMAND

Run ACME's testing pipeline via SLURM

Arguments:
  COMMAND
    pytest        perform testing using pytest in current user environment
                  (if SLURM is available, tests are executed via `srun`)
    tox           use tox to set up a new virtual environment (as defined in tox.ini)
                  and run tests within this newly created env
    -h or --help  show this help message and exit
Example:
  $_selfie pytest
"
}

# Running this script w/no arguments displays the above help message
if [ "$1" == "" ]; then
    usage
fi

# Define default SLURM partition based on architecture we're running on
mArch=`uname -m`
if [ "${mArch}" == "x86_64" ]; then
    pytestQ="8GBL"
    pytestCPU=1
    toxQ="16GBL"
    toxCPU=2
else
    pytestQ="E880"
    pytestCPU=4
    toxQ="E880"
    toxCPU=8
fi

# Set up "global" pytest options for running test-suite (coverage is only done in local pytest runs)
export PYTEST_ADDOPTS="--color=yes --tb=short --verbose"

# The while construction allows parsing of multiple positional/optional args (future-proofing...)
while [ "$1" != "" ]; do
    case "$1" in
        pytest)
            shift
            export PYTHONPATH=$(cd ../../ && pwd)
            if [ $_useSLURM ]; then
                CMD="srun -u -n 1 -p ${pytestQ} --mem=8000m -c ${pytestCPU} pytest"
            else
                PYTEST_ADDOPTS="${PYTEST_ADDOPTS} --cov=../../acme --cov-config=../../.coveragerc"
                export PYTEST_ADDOPTS
                CMD="pytest"
            fi
            echo ">>>"
            echo ">>> Running ${CMD} ${PYTEST_ADDOPTS}"
            echo ">>>"
            ${CMD}
            ;;
        tox)
            shift
            if [ $_useSLURM ]; then
                CMD="srun -u -p ${toxQ} --mem=8000m -c ${toxCPU} tox"
            else
                CMD="tox"
            fi
            echo ">>>"
            echo ">>> Running ${CMD} "
            echo ">>>"
            ${CMD}
            ;;
        -h | --help)
            shift
            usage
            ;;
        *)
            shift
            echo "$_ppname invalid argument '$1'"
            ;;
    esac
done
