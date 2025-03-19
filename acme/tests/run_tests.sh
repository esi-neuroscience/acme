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
useSLURM=$(command -v srun)

# Display brief help message explaining script usage
usage()
{
    echo "
$(basename "$0") [--help] pytest|tox [PYTESTOPTS]

usage:

    $(basename "$0") pytest [PYTESTOPTS]
    $(basename "$0") tox [PYTESTOPTS]

Run ACME's testing pipeline via SLURM or locally.

Arguments:
    pytest       perform testing using pytest in current user environment
                 (if SLURM is available, tests are executed via `srun`)
    tox          use tox to set up a new virtual environment (as defined in tox.ini)
                 and run tests within this environment
    PYTESTOPTS   OPTIONAL: any additional options to be passed on to pytest
    --help       OPTIONAL: show this help message and exit

Examples:
    $(basename "$0") pytest
        Runs testing pipeline in current user environment using settings found
        in pytest.ini
    $(basename "$0") pytest --pdb
        Runs testing pipeline in current user environment using settings found
        in pytest.ini and the additional option --pdb so that pytest
        drops to PDB in case of errors
"
}

# Running this script w/no arguments displays the above help message
if [ "$1" == "" ]; then
    usage
    exit 0
fi

# Parse any provided options
optArray=()
testargs=()
while :; do
    case "$1" in
        "")
            break
            ;;
        pytest)
	        optArray+=("pytest")
            ;;
        tox)
	        optArray+=("tox")
            ;;
        --help)
	        optArray+=("help")
            ;;
        *)
            testargs+=("$1")
            ;;
    esac
    shift
done

# Parse mutually exclusive CLI args
if [[ "${#optArray[@]}" -gt 1 ]]; then
    echo "ERROR: Too many options provided"
    exit 1
fi
if [[ "${#optArray[@]}" -lt 1 ]]; then
    echo "ERROR: At least one valid option required"
    exit 1
fi

# Include additional coverage options
if [[ -z "${testargs}" ]]; then
    testargs=()
fi
testargs+=("--cov=../../acme")
testargs+=("--cov-config=../../.coveragerc")

# On ESI and CoBIC HPC clusters, explicitly define default SLURM partitions to use
if [ "${useSLURM}" ]; then
    mArch=`uname -m`
    if [[ "${HOSTNAME}" == "esi-sv"* ]]; then
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
    elif [[ "${HOSTNAME}" == "bic-sv"* ]]; then
        if [ "${mArch}" == "x86_64" ]; then
            pytestQ="8GBMx86"
            pytestCPU=1
            toxQ="16GBMx86"
            toxCPU=2
        else
            pytestQ="-p 16GBLppc"
            pytestCPU=2
            toxQ="-p 16GBLppc"
            toxCPU=2
        fi
    else
        echo "WARNING: Unknown HPC environment ${HOSTNAME} - falling back to SLURM defaults"
    fi
fi

# (Re)set PYTHONPATH to make local import of ACME possible
# (attempt to) preserve already set PYTHONPATH
if [ -n "${PYTHONPATH+x}" ]; then
    ptmp="${PYTHONPATH}"
fi
export PYTHONPATH=$(cd ../../ && pwd)

# Execute appropriate testing command
for option in "${optArray[@]}"; do
    if [[ "${option}" == "help" ]]; then
	    usage
    elif [[ "${option}" == "pytest" ]]; then
        cmd="pytest"
        if [ "${useSLURM}" ]; then
            if [ -n "${pytestQ+x}" ]; then
                cmd="srun -u -n 1 -p ${pytestQ} --mem=8000m -c ${pytestCPU} ${cmd}"
            else
                cmd="srun -u -n 1 ${cmd}"
            fi
        fi
        cmd="${cmd} ${testargs[@]}"
        echo ">>>"
        echo ">>> Running ${cmd}"
        echo ">>>"
        ${cmd}
    elif [[ "${option}" == "tox" ]]; then
        cmd="tox"
        if [ $_useSLURM ]; then
            if [ -n "${toxQ+x}" ]; then
                cmd="srun -u -p ${toxQ} --mem=8000m -c ${toxCPU} ${cmd}"
            else
                cmd="srun -u ${cmd}"
            fi
        fi
        cmd="${cmd} ${testargs[@]}"
        echo ">>>"
        echo ">>> Running ${cmd} "
        echo ">>>"
        ${cmd}
    fi
done

# Reset PYTHONPATH if it was set before
if [ -n "${ptmp+x}" ]; then
    PYTHONPATH="${ptmp}"
fi

exit 0
