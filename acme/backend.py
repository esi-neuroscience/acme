# -*- coding: utf-8 -*-
# 
# Computational scaffolding for user-interface
# 

# Builtin/3rd party package imports
import os

# Main context manager for parallel execution of user-defined functions
class ACMEdaemon(object):
    
    def __init__(
        self, 
        n_calls,
        n_jobs="auto", 
        write_worker_results=True, 
        partition="auto", 
        mem_per_job="auto",
        setup_timeout=180, 
        setup_interactive=True, 
        start_client=True):
        pass
