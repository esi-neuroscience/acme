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
import dask.distributed as dd
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Tuple

# Fetch logger
log = logging.getLogger("ACME")


class ResultHandler(ABC):
    """Abstract base for result storage strategies"""

    @abstractmethod
    def write_result(self, result: Any, task_id: int, **kwargs) -> None:
        """Write result to storage"""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize after all results written"""
        pass


class MemoryResultHandler(ResultHandler):
    """Collect results in memory"""

    def write_result(self, result: Any, task_id: int, **kwargs) -> None:
        """Simply return result (no storage)"""
        return result

    def finalize(self) -> None:
        pass


class HDF5ResultHandler(ResultHandler):
    """Write results to HDF5 containers"""

    def __init__(self, base_filename: str, result_shape: Optional[tuple] = None,
                 stacking_dim: Optional[int] = None, single_file: bool = False,
                 result_dtype: str = "float"):
        self.base_filename = base_filename
        self.result_shape = result_shape
        self.stacking_dim = stacking_dim
        self.single_file = single_file
        self.result_dtype = result_dtype
        self.locks = {}

    def write_result(self, result: Any, task_id: int, **kwargs) -> None:
        """Write result to HDF5 file with locking"""
        fname = kwargs['outFile']
        single_file = kwargs.get('singleFile', False)
        stacking_dim = kwargs.get('stackingDim', None)

        if isinstance(result, (list, tuple)):
            data = result
        else:
            data = [result]

        if single_file:
            self._write_single_file(data, task_id, **kwargs)
        else:
            self._write_separate_files(data, task_id, **kwargs)

    def _write_single_file(self, data: list, task_id: int, **kwargs) -> None:
        """Write to single shared HDF5 file with distributed lock"""
        fname = kwargs['outFile']
        lock_name = os.path.basename(fname)

        if lock_name not in self.locks:
            self.locks[lock_name] = dd.lock.Lock(name=lock_name)

        lock = self.locks[lock_name]
        lock.acquire()
        try:
            with h5py.File(fname, "a") as h5f:
                if self.stacking_dim is None:
                    for rk, res in enumerate(data):
                        h5f.create_dataset(f"comp_{task_id}/result_{rk}", data=res)
                else:
                    self._write_with_shape(h5f, data, task_id)
        finally:
            lock.release()

    def _write_separate_files(self, data: list, task_id: int, **kwargs) -> None:
        """Write to separate HDF5 file per task"""
        fname = kwargs['outFile']
        with h5py.File(fname, "w") as h5f:
            for rk, res in enumerate(data):
                h5f.create_dataset(f"result_{rk}", data=res)

    def _write_with_shape(self, h5f: h5py.File, data: list, task_id: int) -> None:
        """Write to pre-allocated dataset with specific shape"""
        dset = h5f["result_0"]
        idx = [slice(None)] * len(dset.shape)
        idx[self.stacking_dim] = task_id

        # Handle resizable datasets
        if None in dset.maxshape:
            actShape = self._calculate_actual_shape(dset, data[0])
            dset.resize(actShape)

        dset[tuple(idx)] = data[0]

        # Handle additional return values
        for rk, res in enumerate(data[1:]):
            h5f.create_dataset(f"comp_{task_id}/result_{rk + 1}", data=res)

    def _calculate_actual_shape(self, dset: h5py.Dataset, first_result: Any) -> tuple:
        """Calculate actual shape for resizable dataset"""
        if len(first_result.shape) < len(dset.maxshape):
            lenDim = list(set(first_result.shape).difference(dset.maxshape))
            return tuple(spec if spec is not None else (lenDim[0] if lenDim else first_result.shape[0]) for spec in dset.maxshape)
        else:
            actShape = list(first_result.shape)
            actShape[self.stacking_dim] = dset.maxshape[self.stacking_dim]
            return tuple(actShape)

    def finalize(self) -> None:
        """Clean up any remaining locks"""
        for lock in self.locks.values():
            if lock._held:
                lock.release()


class PickleResultHandler(ResultHandler):
    """Handle emergency pickling when HDF5 fails"""

    def __init__(self, base_filename: str):
        self.base_filename = base_filename

    def write_result(self, result: Any, task_id: int, **kwargs) -> None:
        """Pickle result to file"""
        fname = kwargs['outFile']
        if fname.endswith('.h5') and isinstance(kwargs.get('original_exception'), TypeError):
            fname = fname.rstrip('.h5') + '.pickle'

        with open(fname, "wb") as pkf:
            pickle.dump(result, pkf)

    def finalize(self) -> None:
        pass


class ResultStorageManager:
    """High-level manager for result storage operations"""

    @staticmethod
    def create_handler(
        write_pickle: bool,
        write_worker_results: bool,
        single_file: bool,
        result_shape: Optional[tuple],
        result_dtype: str,
        outfile_pattern: str
    ) -> ResultHandler:
        """Factory method to create appropriate result handler"""
        if not write_worker_results:
            return MemoryResultHandler()
        elif write_pickle:
            return PickleResultHandler(outfile_pattern)
        else:
            return HDF5ResultHandler(
                base_filename=outfile_pattern,
                result_shape=result_shape,
                stacking_dim=result_shape.index(None) if result_shape else None,
                single_file=single_file,
                result_dtype=result_dtype
            )