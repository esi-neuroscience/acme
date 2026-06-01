#
# Tests for result_handler module
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

import os
import tempfile
import pytest
import h5py
import pickle
import numpy as np
from unittest.mock import Mock, patch
from acme.results.result_handler import (
    ResultHandler, MemoryResultHandler, HDF5ResultHandler, 
    PickleResultHandler, ResultStorageManager
)


class TestMemoryResultHandler:
    """Test memory result handler functionality"""

    def test_write_result(self):
        """Test memory result writing"""
        handler = MemoryResultHandler()
        result = {"test": "data"}
        task_id = 0
        
        returned_result = handler.write_result(result, task_id)
        assert returned_result == result

    def test_finalize(self):
        """Test memory result finalization"""
        handler = MemoryResultHandler()
        # Should not raise any exceptions
        handler.finalize()


class TestHDF5ResultHandler:
    """Test HDF5 result handler functionality"""

    def test_write_result_separate_files(self, tmp_path):
        """Test HDF5 result writing to separate files"""
        handler = HDF5ResultHandler(
            base_filename=str(tmp_path),
            single_file=False
        )
        
        result = np.array([[1, 2, 3], [4, 5, 6]])
        task_id = 0
        fname = os.path.join(str(tmp_path), "test_result.h5")
        
        handler.write_result(
            result, task_id, 
            outFile=fname,
            singleFile=False
        )
        
        # Verify file was created
        assert os.path.exists(fname)
        
        # Verify data was written correctly
        with h5py.File(fname, "r") as h5f:
            assert "result_0" in h5f
            assert np.array_equal(h5f["result_0"][()], result)

    def test_write_result_single_file(self, tmp_path):
        """Test HDF5 result writing to single file"""
        handler = HDF5ResultHandler(
            base_filename=str(tmp_path),
            single_file=True
        )
        
        result = np.array([[1, 2, 3], [4, 5, 6]])
        task_id = 0
        fname = os.path.join(str(tmp_path), "test_container.h5")
        
        handler.write_result(
            result, task_id,
            outFile=fname,
            singleFile=True
        )
        
        # Verify file was created
        assert os.path.exists(fname)
        
        # Verify data was written to group
        with h5py.File(fname, "r") as h5f:
            assert f"comp_{task_id}" in h5f
            assert "result_0" in h5f[f"comp_{task_id}"]
            assert np.array_equal(h5f[f"comp_{task_id}/result_0"][()], result)

    def test_finalize(self, tmp_path):
        """Test HDF5 result handler finalization"""
        handler = HDF5ResultHandler(
            base_filename=str(tmp_path),
            single_file=True
        )
        
        # Create a lock to test cleanup
        fname = os.path.join(str(tmp_path), "test_container.h5")
        handler.write_result(
            np.array([1, 2, 3]), 0,
            outFile=fname,
            singleFile=True
        )
        
        # Finalize should clean up locks
        handler.finalize()
        # Should not raise any exceptions


class TestPickleResultHandler:
    """Test pickle result handler functionality"""

    def test_write_result(self, tmp_path):
        """Test pickle result writing"""
        handler = PickleResultHandler(
            base_filename=str(tmp_path)
        )
        
        result = {"test": "data", "array": np.array([1, 2, 3])}
        task_id = 0
        fname = os.path.join(str(tmp_path), "test_result.pickle")
        
        handler.write_result(
            result, task_id,
            outFile=fname
        )
        
        # Verify file was created
        assert os.path.exists(fname)
        
        # Verify data was written correctly
        with open(fname, "rb") as f:
            loaded_result = pickle.load(f)
            assert loaded_result == result
            assert np.array_equal(loaded_result["array"], result["array"])

    def test_finalize(self):
        """Test pickle result handler finalization"""
        handler = PickleResultHandler("test_base")
        # Should not raise any exceptions
        handler.finalize()


class TestResultStorageManager:
    """Test result storage manager functionality"""

    def test_create_memory_handler(self):
        """Test creation of memory result handler"""
        handler = ResultStorageManager.create_handler(
            write_pickle=False,
            write_worker_results=False,
            single_file=False,
            result_shape=None,
            result_dtype="float",
            outfile_pattern="test"
        )
        
        assert isinstance(handler, MemoryResultHandler)

    def test_create_pickle_handler(self):
        """Test creation of pickle result handler"""
        handler = ResultStorageManager.create_handler(
            write_pickle=True,
            write_worker_results=True,
            single_file=False,
            result_shape=None,
            result_dtype="float",
            outfile_pattern="test"
        )
        
        assert isinstance(handler, PickleResultHandler)

    def test_create_hdf5_handler(self):
        """Test creation of HDF5 result handler"""
        handler = ResultStorageManager.create_handler(
            write_pickle=False,
            write_worker_results=True,
            single_file=False,
            result_shape=(3, 4, 5),
            result_dtype="float",
            outfile_pattern="test"
        )
        
        assert isinstance(handler, HDF5ResultHandler)
        assert handler.single_file == False
        assert handler.result_shape == (3, 4, 5)

    def test_create_hdf5_single_file_handler(self):
        """Test creation of HDF5 single file result handler"""
        handler = ResultStorageManager.create_handler(
            write_pickle=False,
            write_worker_results=True,
            single_file=True,
            result_shape=(3, 4, 5),
            result_dtype="float",
            outfile_pattern="test"
        )
        
        assert isinstance(handler, HDF5ResultHandler)
        assert handler.single_file == True
        assert handler.result_shape == (3, 4, 5)


class TestResultHandlerIntegration:
    """Integration tests for result handler functionality"""

    def test_complete_workflow_memory(self):
        """Test complete workflow with memory handler"""
        handler = ResultStorageManager.create_handler(
            write_pickle=False,
            write_worker_results=False,
            single_file=False,
            result_shape=None,
            result_dtype="float",
            outfile_pattern="test"
        )
        
        result = {"data": np.array([1, 2, 3])}
        task_id = 0
        
        returned_result = handler.write_result(result, task_id)
        assert returned_result == result
        
        handler.finalize()

    def test_complete_workflow_hdf5(self, tmp_path):
        """Test complete workflow with HDF5 handler"""
        handler = ResultStorageManager.create_handler(
            write_pickle=False,
            write_worker_results=True,
            single_file=False,
            result_shape=(2, 3),
            result_dtype="float",
            outfile_pattern=str(tmp_path)
        )
        
        result = np.array([[1, 2, 3], [4, 5, 6]])
        task_id = 0
        fname = os.path.join(str(tmp_path), "test_result.h5")
        
        handler.write_result(
            result, task_id,
            outFile=fname,
            singleFile=False
        )
        
        # Verify file was created and contains correct data
        assert os.path.exists(fname)
        with h5py.File(fname, "r") as h5f:
            assert "result_0" in h5f
            assert np.array_equal(h5f["result_0"][()], result)
        
        handler.finalize()