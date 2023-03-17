::
:: Copyright (c) 2023 Ernst Strüngmann Institute (ESI) for Neuroscience in Cooperation with Max Planck Society
::
:: SPDX-License-Identifier: CC0-1.0
::

@echo off
for %%I in ("%cd%\..\..") do set "PYTHONPATH=%%~fI"

set PYTEST_ADDOPTS="-v"

if "%1" == "" goto usage

for %%a in (%*) do (
    if "%%a" == "tox" (
        tox
        goto end
    )
    if "%%a" == "pytest" (
        pytest
        goto end
    ) else (goto usage)
)

:end
exit /B 1

:usage
echo "Run ACME's testing pipeline on Windows"
echo " "
echo "Arguments:"
echo "  pytest  perform testing using pytest in current user environment"
echo "  tox     use tox to set up a new virtual environment (as defined in tox.ini)"
echo "          and run tests within this newly created env"
echo " "
echo "Example: run_tests.cmd pytest "
