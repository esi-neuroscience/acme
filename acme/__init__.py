# -*- coding: utf-8 -*-
# 
# Main package initializer
# 

# Builtin/3rd party package imports
import dask.distributed as dd

# Global version number
__version__ = "0.1a"

# Check if we're being imported by a parallel worker process
try: 
    dd.get_worker()
    __worker__ = True
except ValueError:
    __worker__ = False
