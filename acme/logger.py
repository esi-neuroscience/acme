#
# ACME's logging facilities
#
# Copyright © 2023 Ernst Strüngmann Institute (ESI) for Neuroscience in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import warnings
import datetime
import inspect
import logging
import os

__all__ = []


def prepare_log(caller=None, logfile=False, func=None, verbose=None):
    """
    Convenience function to set up ACME logger

    Parameters
    ----------
    caller : None or str
        Routine/class that initiated logging (presumable :class:~`acme.ParallelMap`
        or :class:~`acme.ACMEDaemon`)
    logfile : None or bool or str
        If `True` an auto-generated log-file is set up. If `logfile` is a string
        it is interpreted as file-name for a new log-file (must not exist). If
        `False` or `None` logging information is streamed to stdout only.
    func : None or callable
        User-provided function to be called concurrently by ACME (optional)
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
    funcName = "<{}>".format(inspect.currentframe().f_code.co_name)

    # If not provided, get name of calling method/function
    if caller is None:
        caller = "<{}>".format(inspect.currentframe().f_back.f_code.co_name)
    elif not isinstance(caller, str):
        msg = "%s `caller` has to be a string, not %s"
        raise TypeError(msg%(funcName, str(type(caller))))

    # Basal sanity check for Boolean flag
    if verbose is not None and not isinstance(verbose, bool):
        msg = "%s `verbose` has to be `True`, `False` or `None`, not %s"
        raise TypeError(msg%(funcName, str(type(verbose))))

    # Either parse provided `logfile` or set up an auto-generated file
    msg = "%s `logfile` has to be `None`, `True`, `False` or a valid file-name, not %s"
    if logfile is None or isinstance(logfile, bool):
        if logfile is True:
            if func is None:
                msg = "%s cannot auto-create log-file if `func` is `None`. Skipping"
                warnings.showwarning(msg%(caller), RuntimeWarning,
                                    __file__, inspect.currentframe().f_lineno)
                logfile = None
            else:
                logfile = os.path.dirname(os.path.abspath(inspect.getfile(func)))
                logfile = os.path.join(logfile, "ACME_{func:s}_{date:s}.log")
                logfile = logfile.format(func=func.__name__,
                                        date=datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        else:
            logfile = None
    elif isinstance(logfile, str):
        if os.path.isdir(logfile):
            raise IOError(msg%(funcName, "a directory"))
        logfile = os.path.abspath(os.path.expanduser(logfile))
    else:
        raise TypeError(msg%(funcName, str(type(logfile))))
    if logfile is not None and os.path.isfile(logfile):
        msg = "%s log-file %s already exists, appending to it"
        warnings.showwarning(msg%(caller, logfile), RuntimeWarning,
                             __file__, inspect.currentframe().f_lineno)

    # Set logging verbosity based on `verbose` flag
    if verbose is None:
        loglevel = logging.INFO
    elif verbose is True:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.WARNING
    log = logging.getLogger(caller)
    log.setLevel(loglevel)

    # Create logging formatter
    formatter = AcmeFormatter("%(name)s %(levelname)s %(message)s")

    # Output handlers: print log messages via `StreamHandler` as well
    # as to a provided text file `logfile using a `FileHandler`.
    # Note: at import time (when logger is initially set up) no `logfile`
    # specification is provided, so `fileHandler`` can only be set up upon
    # successive calls to `prepare_log`
    if len(log.handlers) == 0:
        initialRun = True
        stdoutHandler = logging.StreamHandler()
    else:
        # Note: avoid adding the same log-file location as distinct handlers to the logger
        # in case `ParallelMap` is executed repeatedly; also remove existing non-default
        # logfile handlers to avoid generating multiple logs (and accidental writes to existing logs)
        initialRun = False
        stdoutHandler = [h for h in log.handlers if isinstance(h, logging.StreamHandler)][0]
        if logfile is not None:
            fileHandler = None
            fHandlers = [h for h in log.handlers if isinstance(h, logging.FileHandler)]
            for handler in fHandlers:
                if handler.baseFilename == logfile:
                    fileHandler = handler
                    break
                log.handlers.remove(handler)
            # No file-handler configured yet, create a new one
            if fileHandler is None:
                fileHandler = logging.FileHandler(logfile)
                log.addHandler(fileHandler)

    # Apply formatting to existing loggers as well as newly created ones
    stdoutHandler.setLevel(loglevel)
    stdoutHandler.setFormatter(formatter)
    if logfile is not None:
        fileHandler.setLevel(loglevel)
        fileHandler.setFormatter(formatter)

    # If this is round one, add stdout handler
    if initialRun:
        log.addHandler(stdoutHandler)

    return log


class AcmeFormatter(logging.Formatter):
    """
    Adapted from https://alexandra-zaharia.github.io/posts/make-your-own-custom-color-formatter-with-python-logging/
    """

    green = "\x1b[92m"
    gray = "\x1b[90m"
    blue = "\x1b[38;5;39m"
    magenta = "\x1b[35m"
    red = "\x1b[38;5;196m"
    bold = "\x1b[1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()

        fmtName = fmt.partition("%(name)s")
        fmtName = fmtName[0] + self.bold + fmtName[1] + self.reset + fmtName[2]
        fmt = "".join(fmtName)

        fmtLvl = fmt.partition("%(levelname)s")
        fmtDebug = fmtLvl[0] + self.bold + self.green + \
            "# " + fmtLvl[1] + " #" + self.reset + self.gray + fmtLvl[2] + self.reset
        fmtInfo = fmtLvl[0] + self.bold + self.blue + \
            "- " + fmtLvl[1] + " -" + self.reset + fmtLvl[2]
        fmtWarn = fmtLvl[0] + self.bold + self.magenta + \
            "! " + fmtLvl[1] + " !" + self.reset + fmtLvl[2]
        fmtError = fmtLvl[0] + self.bold + self.red + \
            "| " + fmtLvl[1] + " |" + self.reset + self.red + fmtLvl[2] + self.reset

        self.FORMATS = {
            logging.DEBUG: "".join(fmtDebug),
            logging.INFO: "".join(fmtInfo),
            logging.WARNING: "".join(fmtWarn),
            logging.ERROR: "".join(fmtError),
            logging.CRITICAL: "".join(fmtError),
        }

    def format(self, record):
        logFmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(logFmt)
        return formatter.format(record)
