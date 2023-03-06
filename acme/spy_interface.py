# -*- coding: utf-8 -*-
#
# Interface for streamlining output to Syncopy
#

try:
    import syncopy as spy
    isSpyModule = True
except ImportError:
    isSpyModule = False

if isSpyModule:
    customIOError = lambda msg : spy.shared.errors.SPYIOError(msg)
    customValueError = lambda legal=None, varname=None, actual=None : \
        spy.shared.errors.SPYValueError(legal=legal, varname=varname, actual=actual)
    customTypeError = lambda val, varname=None, expected=None : \
        spy.shared.errors.SPYTypeError(val, varname=varname, expected=expected)
    scalar_parser = lambda var, varname="", ntype=None, lims=None : \
        spy.shared.parsers.scalar_parser(var, varname=varname, ntype=ntype, lims=lims)
else:
    isSpyModule = False
    from warnings import showwarning
    import logging
    from .shared import _scalar_parser as scalar_parser
    customIOError = IOError
    customValueError = lambda legal=None, varname=None, actual=None : ValueError(legal)
    customTypeError = lambda msg, varname=None, expected=None : TypeError(msg)
