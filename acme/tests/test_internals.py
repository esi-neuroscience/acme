#
# Testing module for non-user exposed functionality in ACME
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import os
import shutil
import pytest
import numpy as np
import logging
from logging import handlers

# Local imports
import acme
from acme.shared import sizeOf, callMax, _scalar_parser
from acme.logger import prepare_log

def test_sizeof():

    # Ensure self-referencing dict does not lock up ACME (decrease callMax
    # to not trigger segfaults)
    infDict = {}; infDict["key"] = infDict
    acme.shared.callMax = 100
    with pytest.raises(RecursionError) as recerr:
        sizeOf(infDict, "infdict")
        assert f"maximum recursion depth {acme.shared.callMax} exceeded" in str(recerr.value)
    acme.shared.callMax = callMax

    # Ensure well-behaved dicts are processed correctly (`arrsize` denotes array size in MB)
    arrsize = 100
    arr = np.ones((int(arrsize * 1024**2 / np.dtype("float").itemsize), ))
    normDict = {"arr": arr, "b": "string"}
    assert sizeOf(normDict, "normDict") > arrsize

def test_scalarparser():

    # Ensure int-likes are properly recognized
    _scalar_parser(3, ntype="int_like")
    _scalar_parser(3.0, ntype="int_like")
    with pytest.raises(ValueError) as valerr:
        _scalar_parser(3.14, ntype="int_like")
        assert "`varname` has to be an integer between -inf and inf, not 3.14" in str(valerr.value)

    # Ensure limits are inclusive and respected
    _scalar_parser(3, lims=[-4, 4])
    _scalar_parser(3, lims=[3, 3])
    with pytest.raises(ValueError) as valerr:
        _scalar_parser(3, lims=[0, 2])
        assert "`varname` has to be an integer between 0 and 2, not 3" in str(valerr.value)

    # Anything not-int-like should parse fine too
    _scalar_parser(3, ntype="something", lims=[3, 3])

    # Anything not number-like should not
    with pytest.raises(TypeError) as tperr:
        _scalar_parser("notAnumber")
        assert "`varname` has to be a scalar, not <class 'str'>" in str(tperr.value)

def test_logger():

    # Basic input parsing
    with pytest.raises(TypeError):
        prepare_log(logname=3)
    with pytest.raises(TypeError):
        prepare_log(verbose="invalid")

    # Create tmp directory for logfile
    tempDir = os.path.join(os.path.abspath(os.path.expanduser("~")), "acme_tmp")
    os.makedirs(tempDir, exist_ok=True)
    customLog = os.path.join(tempDir, "logger_test.txt")

    # Ensure basic logging functionality
    prepare_log(logname="custom", logfile=customLog)
    log = logging.getLogger("custom")
    logMsg = "this was test_logger"
    dbgMsg = "this is invisible"
    log.info(logMsg)
    log.debug(dbgMsg)
    memHandlers = [h for h in log.handlers if isinstance(h, handlers.MemoryHandler)]
    assert len(memHandlers) == 1
    memHandler = memHandlers[0]
    memHandler.flush()  # important: flush memory to write its contents to file
    assert memHandler.target is not None
    assert memHandler.target.baseFilename == customLog
    with open(customLog, "r", encoding="utf8") as fl:
        logTxt = fl.read()
    assert logMsg in logTxt
    assert dbgMsg not in logTxt

    # Ensure logger creation works w/existing file
    prepare_log(logname="custom", logfile=customLog)
    logMsg2 = "another test_logger"
    log.info(logMsg2)
    memHandlers = [h for h in log.handlers if isinstance(h, handlers.MemoryHandler)]
    assert len(memHandlers) == 1
    memHandler = memHandlers[0]
    memHandler.flush()  # important: flush memory to write its contents to file
    assert memHandler.target is not None
    assert memHandler.target.baseFilename == customLog
    with open(customLog, "r", encoding="utf8") as fl:
        logTxt = fl.read()
    assert logMsg2 in logTxt

    # Ensure chosen log levels are respected
    prepare_log(logname="custom", logfile=customLog, verbose=False)
    logWrn = "this must show up"
    logMsg3 = "this should not show up"
    log.warning(logWrn)
    log.info(logMsg3)
    memHandlers = [h for h in log.handlers if isinstance(h, handlers.MemoryHandler)]
    assert len(memHandlers) == 1
    memHandler = memHandlers[0]
    memHandler.flush()  # important: flush memory to write its contents to file
    assert memHandler.target is not None
    assert memHandler.target.baseFilename == customLog
    with open(customLog, "r", encoding="utf8") as fl:
        logTxt = fl.read()
    assert logWrn in logTxt
    assert logMsg3 not in logTxt

    # Clean up
    os.unlink(customLog)
    shutil.rmtree(tempDir, ignore_errors=True)
