#
# ACME's logging facilities
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import warnings
import inspect
import logging
import os
from logging import handlers
from typing import Optional, Any, List

__all__: List[str] = []


def prepare_log(
        logname: str = "ACME",
        logfile: Optional[str] = None,
        verbose: Optional[bool] = None) -> None:
    """
    Convenience function to set up ACME logger

    Parameters
    ----------
    logname : str
        Name of the logger to set up
    logfile : None or str
        If `None`, logging information is streamed to stdout only, otherwise
        `logfile` is interpreted as path to a file used for logging.
    verbose : bool or None
        If `None`, the logging-level only contains messages of `'INFO'` priority and
        higher (`'WARNING'` and `'ERROR'`). If `verbose` is `True`, logging is
        performed on ``DEBUG`', `'INFO`', `'WARNING'` and `'ERROR'` levels. If
        `verbose` is `False` only `'WARNING'` and `'ERROR'` messages are propagated.

    Returns
    -------
    log : logger object
        A Python :class:`logging.Logger` instance
    """

    # For later reference: dynamically fetch name of current function
    funcName = f"<{inspect.currentframe().f_code.co_name}>"     # type: ignore

    # Ensure `logname` can be processed
    if not isinstance(logname, str):
        msg = "%s `caller` has to be a string, not %s"
        raise TypeError(msg%(funcName, str(type(logname))))

    # Basal sanity check for Boolean flag
    if verbose is not None and not isinstance(verbose, bool):
        msg = "%s `verbose` has to be `True`, `False` or `None`, not %s"
        raise TypeError(msg%(funcName, str(type(verbose))))

    # Either parse provided `logfile` or set up an auto-generated file
    if logfile is not None and os.path.isfile(logfile):
        msg = "%s log-file %s already exists, appending to it"
        warnings.warn(msg%(logname, logfile))

    # Add custom "announce" level to logger
    announceLvl = logging.INFO + 5
    logging.addLevelName(announceLvl, "ANNOUNCE")
    logging.ANNOUNCE = announceLvl                                      # type: ignore

    def announce(
            self,
            msg: str,
            *args: Any,
            **kwargs: Optional[Any]) -> None:
        """
        Log 'msg % args' with severity 'ANNOUNCE'.

        To pass exception information, use the keyword argument exc_info with
        a true value, e.g.

        logger.announce("Houston, we have a %s", "thorny problem", exc_info=1)
        """
        if self.isEnabledFor(announceLvl):
            self._log(announceLvl, msg, args, **kwargs)

    logging.getLoggerClass().announce = announce                        # type: ignore
    logging.announce = announce                                         # type: ignore

    # Fetch/set up ACME logger
    log = logging.getLogger(logname)

    # Set logging verbosity based on `verbose` flag
    if verbose is None:
        loglevel = logging.INFO
    elif verbose is True:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.WARNING
    log.setLevel(loglevel)

    # Create logging formatters
    acmeFmt = "%(name)s %(levelname)s <%(funcName)s> %(message)s"
    streamFrmt = AcmeFormatter(acmeFmt, color=True)
    fileFrmt = AcmeFormatter(acmeFmt, color=False)

    # Upon package import, create stdout handler + memory handler for
    # temporary buffering of all log messages
    if len(log.handlers) == 0:
        stdoutHandler = logging.StreamHandler()
        stdoutHandler.setFormatter(streamFrmt)
        log.addHandler(stdoutHandler)
        memHandler = handlers.MemoryHandler(1000,
                                            flushLevel=logging.ERROR,
                                            target=None,
                                            flushOnClose=True)
        memHandler.setFormatter(streamFrmt)
        log.addHandler(memHandler)

    # If log-file creation was requested, add a target to the initially
    # created `MemoryHandler` (initiated by `ACMEdaemon`)
    if logfile is not None:
        memHandler = [h for h in log.handlers if isinstance(h, handlers.MemoryHandler)][0]
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setLevel(loglevel)
        fileHandler.setFormatter(fileFrmt)
        memHandler.setTarget(fileHandler)

    # No matter if called for the first or n-th time, (re)set log-level
    # and formatter
    for h in log.handlers:
        h.setLevel(loglevel)

    return


class AcmeFormatter(logging.Formatter):
    """
    Adapted from https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
    """

    def __init__(
            self,
            fmt: str,
            color: bool = True):

        super().__init__()

        if color:
            green = "\x1b[92m"
            gray = "\x1b[90m"
            blue = "\x1b[38;5;39m"
            magenta = "\x1b[35m"
            red = "\x1b[38;5;196m"
            bold = "\x1b[1m"
            reset = "\x1b[0m"
        else:
            green = ""
            gray = ""
            blue = ""
            magenta = ""
            red = ""
            bold = ""
            reset = ""

        fmtName = fmt.partition("%(name)s")
        fmtName = fmtName[0] + bold + fmtName[1] + reset + fmtName[2]   # type: ignore
        fmt = "".join(fmtName)

        fmtLvl = fmt.partition("%(levelname)s")
        fmtDebug = fmtLvl[0] + bold + green + \
            "# " + fmtLvl[1] + " #" + reset + gray + fmtLvl[2] + reset
        fmtInfo = fmtLvl[0] + bold + blue + \
            "- " + fmtLvl[1] + " -" + reset + fmtLvl[2]
        fmtAnnounce = fmtLvl[0] + bold + blue + \
            "> " + fmtLvl[1] + " <" + reset + bold + fmtLvl[2] + reset
        fmtWarn = fmtLvl[0] + bold + magenta + \
            "! " + fmtLvl[1] + " !" + reset + fmtLvl[2]
        fmtError = fmtLvl[0] + bold + red + \
            "| " + fmtLvl[1] + " |" + reset + red + fmtLvl[2] + reset

        fmtAnnounce = "".join(fmtAnnounce).replace("<%(funcName)s>", "")
        fmtInfo = "".join(fmtInfo).replace("<%(funcName)s>", "")

        self.FORMATS = {
            logging.DEBUG: "".join(fmtDebug),
            logging.INFO: "".join(fmtInfo),
            logging.ANNOUNCE: "".join(fmtAnnounce),                     # type: ignore
            logging.WARNING: "".join(fmtWarn),
            logging.ERROR: "".join(fmtError),
            logging.CRITICAL: "".join(fmtError),
        }

    def format(
            self,
            record: logging.LogRecord):
        logFmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(logFmt)
        return formatter.format(record)
