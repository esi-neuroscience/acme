# -*- coding: utf-8 -*-
#
# Main package initializer
#

# Builtin/3rd party package imports
import subprocess
import warnings
import inspect
import sys
from pkg_resources import get_distribution, DistributionNotFound

# Get package version: either via meta-information from egg or via latest git commit
try:
    __version__ = get_distribution("esi-acme").version
except DistributionNotFound:
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

# Central collection of deprecated keywords/canonical warning message
__deprecated__ = ["n_jobs", "mem_per_job", "n_jobs_startup" "workers_per_job"]
__deprecation_wrng__ = \
    "The keywords `n_jobs`, `mem_per_job`, `n_jobs_startup` and " +\
    "`workers_per_job` are DEPRECATED. Please use `n_workers`, `mem_per_worker`, " +\
    "`n_workers_startup` and `processes_per_worker`, respectively."

# Import local modules
from . import frontend, backend, shared, dask_helpers
from .frontend import *
from .backend import *
from .shared import *
from .dask_helpers import *

# Override default exception handler (take care of Jupyter's Exception handling)
from .shared import ctrlc_catcher
try:
    ipy = get_ipython()
    import IPython
    ipy.ipyTBshower = IPython.core.interactiveshell.InteractiveShell.showtraceback
    IPython.core.interactiveshell.InteractiveShell.showtraceback = ctrlc_catcher
except NameError:
    sys.excepthook = ctrlc_catcher

# Manage user-exposed namespace imports
__all__ = []
__all__.extend(frontend.__all__)
__all__.extend(backend.__all__)
__all__.extend(shared.__all__)
__all__.extend(dask_helpers.__all__)
