#
# Result handling utilities for ACME
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import os
import pickle
import logging
import h5py
import numbers
import weakref
import dask.distributed as dd
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Tuple

# Fetch logger
log = logging.getLogger("ACME")


class ResultHandler(ABC):
    """Abstract base for result storage strategies"""

    @abstractmethod
    def write_result(self, **kwargs) -> None:
        """Write result to storage"""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize after all results written"""
        pass


class HDF5ResultHandler(ResultHandler):
    """Write results to HDF5 containers"""

    def __init__(
        self,
        fname: str,
        result: Any,
        task_id: int,
        single_file: bool,
        stacking_dim: Optional[int],
    ):
        self.fname = fname
        self.result = result
        self.task_id = task_id
        self.single_file = single_file
        self.stacking_dim = stacking_dim
        self.grpName = ""
        self.locks = {}
        self.finalizer = weakref.finalize(self, self.finalize)

    def write_result(self, **kwargs) -> None:
        """Write result to HDF5 file with locking"""

        if not isinstance(self.result, (list, tuple)):
            self.result = [self.result]

        if self.single_file:
            self._write_single_file()
        else:
            self._write_multiple_files()

    def _write_single_file(self) -> None:
        """Write to single shared HDF5 file with distributed lock"""

        lock_name = os.path.basename(self.fname)
        if lock_name not in self.locks:
            self.locks[lock_name] = dd.lock.Lock(name=lock_name)
        lock = self.locks[lock_name]
        lock.acquire()
        self.grpName = f"comp_{taskID}/"
        try:
            with h5py.File(self.fname, "a") as h5f:
                if self.stacking_dim is None:
                    self._write_no_shape(h5f)
                else:
                    self._write_with_shape(h5f)
        except TypeError as exc:
            err = "Could not write to %s. File potentially corrupted. Original error message: %s"
            log.error(err, self.fname, str(exc))
            lock.release()
            raise exc
        except Exception as exc:
            lock.release()
            log.error(str(exc))
            raise exc
        finally:
            lock.release()

    def _write_no_shape(self, h5f: h5py.File) -> None:
        """Create datasets in container"""
        if not all(isinstance(value, (numbers.Number, str)) for value in self.result):
            for rk, res in enumerate(self.result):
                h5f.create_dataset(f"{self.grpName}result_{rk}", data=res)
            else:
                h5f.create_dataset(self.grpName + "result_0", data=self.result)

    def _write_with_shape(self, h5f: h5py.File) -> None:
        """Write to pre-allocated dataset with specific shape"""
        dset = h5f["result_0"]
        idx = [slice(None)] * len(dset.shape)
        idx[self.stacking_dim] = self.task_id

        # Handle resizable datasets
        if None in dset.maxshape:
            actShape = self._calculate_actual_shape(dset)
            dset.resize(actShape)

        dset[tuple(idx)] = self.result[0]

        # Handle additional return values
        for rk, res in enumerate(self.result[1:]):
            h5f.create_dataset(f"comp_{self.task_id}/result_{rk + 1}", data=res)

    def _calculate_actual_shape(self, dset: h5py.Dataset) -> tuple:
        """Calculate actual shape for resizable dataset"""
        if len(self.result[0].shape) < len(dset.maxshape):
            lenDim = list(set(self.result[0].shape).difference(dset.maxshape))
            return tuple(
                (
                    spec
                    if spec is not None
                    else (lenDim[0] if lenDim else self.result[0].shape[0])
                )
                for spec in dset.maxshape
            )
        else:
            actShape = list(self.result[0].shape)
            actShape[self.stacking_dim] = dset.maxshape[self.stacking_dim]
            return tuple(actShape)

    def _write_multiple_files(self) -> None:
        """Write to separate HDF5 file per task"""
        self.grpName = ""
        try:
            with h5py.File(self.fname, "a") as h5f:
                if self.stacking_dim is None:
                    self._write_no_shape(h5f)
                else:
                    for rk, res in enumerate(self.result):
                        h5f.create_dataset(f"result_{rk}", data=res)
        except TypeError as exc:
            if "has no native HDF5 equivalent" in str(
                exc
            ) or "One of data, shape or dtype must be specified" in str(exc):
                os.unlink(self.fname)  # type: ignore
                err = f"Unable to write {self.fname}, successive attempts to pickle results failed too: %s"
                pickle_handler = PickleResultHandler(
                    fname=self.fname, result=self.result, task_id=self.task_id
                )
                pickle_handler.write_result(original_exception=exc, err_msg=err)
                msg = (
                    "Could not write %s results have been pickled instead. Return values are most likely "
                    + "not suitable for storage in HDF5 containers. Original error message: %s"
                )
                log.warning(msg, self.fname, str(exc))
            else:
                err = "Could not access %s. Original error message: %s"
                log.error(err, self.fname, str(exc))
                raise exc
        except Exception as exc:
            log.error(str(exc))
            raise exc

    def finalize(self) -> None:
        """Clean up any remaining locks"""
        for lock in self.locks.values():
            if lock._held:
                lock.release()


class PickleResultHandler(ResultHandler):
    """Handle emergency pickling when HDF5 fails"""

    def __init__(
        self,
        fname: str,
        result: Any,
        task_id: int,
    ):
        self.fname = fname
        self.result = result
        self.task_id = task_id
        self.finalizer = weakref.finalize(self, self.finalize)

    def write_result(self, **kwargs) -> None:
        """Pickle result to file"""
        if self.fname.endswith(".h5") and isinstance(
            kwargs.get("original_exception"), TypeError
        ):
            self.fname = self.fname.rstrip(".h5") + ".pickle"

        try:
            with open(self.fname, "wb") as pkf:
                pickle.dump(self.result, pkf)
        except pickle.PicklingError as pexc:
            err = f"Could not pickle results to file {self.fname}. Original error message: %s"
            log.error(kwargs.get("err_msg", err))
            log.error(err, str(pexc))

    def finalize(self) -> None:
        pass


class ResultStorageManager:
    """High-level manager for result storage operations"""

    @staticmethod
    def create_handler(
        fname: str,
        result: Any,
        task_id: int,
        write_pickle: bool,
        single_file: bool,
        stacking_dim: Optional[int],
    ) -> ResultHandler:
        """Factory method to create appropriate result handler"""
        if write_pickle:
            return PickleResultHandler(fname=fname, result=result, task_id=task_id)
        else:
            return HDF5ResultHandler(
                fname=fname,
                result=result,
                task_id=task_id,
                single_file=single_file,
                stacking_dim=stacking_dim,
            )
