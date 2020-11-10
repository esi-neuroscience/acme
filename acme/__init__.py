# -*- coding: utf-8 -*-
#
# Main package initializer
#

# Builtin/3rd party package imports
import dask.distributed as dd

# Global version number
__version__ = "0.1a0"

# Check if we're being imported by a parallel worker process
try:
    dd.get_worker()
    __worker__ = True
except ValueError:
    __worker__ = False

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
