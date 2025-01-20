#
# Interface for integrating with Syncopy
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import sys
import logging

# Differentiate b/w being imported as Syncopy sub-package or standalone
# ACME module: do not attempt to import syncopy but instead see if it has
# already been imported; then ACME should use Syncopy's built-in logging
if "syncopy" in sys.modules:                                            # pragma: no cover
    import syncopy as spy                                               # type: ignore
    log = logging.getLogger("syncopy")
    scalar_parser = lambda var, varname="", ntype=None, lims=None : \
        spy.shared.parsers.scalar_parser(var, varname=varname, ntype=ntype, lims=lims)
else:
    from .shared import _scalar_parser as scalar_parser
    log = logging.getLogger("ACME")
