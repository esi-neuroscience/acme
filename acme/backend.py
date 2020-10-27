# -*- coding: utf-8 -*-
# 
# Computational scaffolding for user-interface
# 

# Builtin/3rd party package imports
import os

# Main context manager for parallel execution of user-defined functions
class ACMEdaemon(object):
    
    def __init__(self, file_name, method):
        self.file_obj = open(file_name, method)
