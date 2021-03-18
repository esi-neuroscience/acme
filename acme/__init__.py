# -*- coding: utf-8 -*-
#
# Main package initializer
#

# Builtin/3rd party package imports
import subprocess
import warnings
import inspect
import dask.distributed as dd
from importlib.metadata import version, PackageNotFoundError

# Get package version: either via meta-information from egg or via latest git commit
try:
    __version__ = version("esi-acme")
except PackageNotFoundError:
    proc = subprocess.Popen("git describe --always",
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, shell=True)
    out, err = proc.communicate()
    if proc.returncode != 0:
        proc = subprocess.Popen("git rev-parse HEAD:acme/__init__.py",
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, shell=True)
        out, err = proc.communicate()
        if proc.returncode != 0:
            msg = "<ACME> Package is not installed in site-packages nor cloned via git. " +\
                "Please consider obtaining ACME sources from supported channels. "
            warnings.showwarning(msg, ImportWarning, __file__, inspect.currentframe().f_lineno)
            out = "-999"
    __version__ = out.rstrip("\n")

# Import local modules
from . import frontend, backend, shared, dask_helpers
from .frontend import *
from .backend import *
from .shared import *
from .dask_helpers import *

# Manage user-exposed namespace imports
__all__ = []
__all__.extend(frontend.__all__)
__all__.extend(backend.__all__)
__all__.extend(shared.__all__)
__all__.extend(dask_helpers.__all__)
