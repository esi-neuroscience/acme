#
# Main package initializer
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import subprocess
import warnings
import sys
from importlib.metadata import version, PackageNotFoundError
from typing import List

# Get package version: either via meta-information from egg or via latest git commit
try:
    __version__ = version("esi-acme")
except PackageNotFoundError:
    proc = subprocess.Popen("git describe --always",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    out, err = proc.communicate()
    if proc.returncode != 0:                                            # pragma: no cover
        proc = subprocess.Popen("git rev-parse HEAD:acme/__init__.py",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True)
        out, err = proc.communicate()
        if proc.returncode != 0:
            msg = "<ACME> Package is not installed in site-packages nor cloned via git. " +\
                "Please consider obtaining ACME sources from supported channels. "
            warnings.warn(msg)
            out = "-999"
    __version__ = out.rstrip("\n")

# Remove dask-jobqueue's FutureWarnings about tmpfile (which we don't use)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import local modules
from . import frontend, backend, shared, dask_helpers
from .frontend import *
from .backend import *
from .shared import *
from .dask_helpers import *
from .logger import prepare_log

# Set up module-wide logging
prepare_log(logname="ACME")

# Override default exception handler (take care of Jupyter's Exception handling)
from .shared import ctrlc_catcher
try:
    ipy = get_ipython()                                                             # type: ignore
    import IPython
    ipy.ipyTBshower = IPython.core.interactiveshell.InteractiveShell.showtraceback
    IPython.core.interactiveshell.InteractiveShell.showtraceback = ctrlc_catcher    # type: ignore
except NameError:
    sys.excepthook = ctrlc_catcher

# Manage user-exposed namespace imports
__all__: List[str] = []
__all__.extend(frontend.__all__)
__all__.extend(backend.__all__)
__all__.extend(shared.__all__)
__all__.extend(dask_helpers.__all__)
