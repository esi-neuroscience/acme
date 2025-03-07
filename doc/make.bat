REM
REM Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
REM in Cooperation with Max Planck Society
REM
REM SPDX-License-Identifier: BSD-3-Clause
REM

@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help
if "%1" == "clean" goto clean

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

REM Custom directive to clean up build dir api files
:clean
del /Q /S %BUILDDIR%"\*" > nul
rmdir /Q /S %BUILDDIR% > nul
del /Q /S %SOURCEDIR%"\api\*" > nul
goto end

:end
popd
