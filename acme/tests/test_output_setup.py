#
# Tests for output_setup module
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
import numpy as np
from acme.results.output_setup import OutputDirectoryManager, HDF5ContainerFactory, OutputSetupError


class TestOutputDirectoryManager:
    """Test output directory management functionality"""

    def test_create_payload_directory(self, tmp_path):
        """Test payload directory creation"""
        func_name = "test_function"
        payload_dir = HDF5ContainerFactory.create_payload_directory(str(tmp_path), func_name)
        
        expected_dir = os.path.join(str(tmp_path), f"{func_name}_payload")
        assert payload_dir == expected_dir
        assert os.path.exists(payload_dir)
        assert os.path.isdir(payload_dir)

    def test_create_payload_directory_existing(self, tmp_path):
        """Test payload directory creation when directory already exists"""
        func_name = "test_function"
        payload_dir = HDF5ContainerFactory.create_payload_directory(str(tmp_path), func_name)
        
        # Call again - should not raise error due to exist_ok=True
        payload_dir2 = HDF5ContainerFactory.create_payload_directory(str(tmp_path), func_name)
        assert payload_dir == payload_dir2


class TestHDF5ContainerFactory:
    """Test HDF5 container creation functionality"""

    def test_create_single_file_container_no_shape(self, tmp_path):
        """Test single file container creation without result shape"""
        filename = os.path.join(str(tmp_path), "test_container.h5")
        task_ids = [0, 1, 2]
        
        HDF5ContainerFactory.create_single_file_container(
            filename, task_ids, None, "float"
        )
        
        # Verify container was created
        assert os.path.exists(filename)
        
        # Verify groups were created
        with h5py.File(filename, "r") as h5f:
            for task_id in task_ids:
                assert f"comp_{task_id}" in h5f
                assert isinstance(h5f[f"comp_{task_id}"], h5py.Group)

    def test_create_single_file_container_with_shape(self, tmp_path):
        """Test single file container creation with result shape"""
        filename = os.path.join(str(tmp_path), "test_container.h5")
        task_ids = [0, 1, 2]
        result_shape = (3, 4, 5)
        
        HDF5ContainerFactory.create_single_file_container(
            filename, task_ids, result_shape, "float"
        )
        
        # Verify container was created
        assert os.path.exists(filename)
        
        # Verify dataset was created with correct shape
        with h5py.File(filename, "r") as h5f:
            assert "result_0" in h5f
            dataset = h5f["result_0"]
            assert dataset.shape == result_shape
            assert dataset.dtype == np.dtype('float64')

    def test_create_single_file_container_with_inf_shape(self, tmp_path):
        """Test single file container creation with infinite dimensions"""
        filename = os.path.join(str(tmp_path), "test_container.h5")
        task_ids = [0, 1, 2]
        result_shape = (np.inf, 4, 5)
        
        HDF5ContainerFactory.create_single_file_container(
            filename, task_ids, result_shape, "float"
        )
        
        # Verify container was created
        assert os.path.exists(filename)
        
        # Verify dataset was created with resizable dimensions
        with h5py.File(filename, "r") as h5f:
            assert "result_0" in h5f
            dataset = h5f["result_0"]
            assert dataset.shape == (1, 4, 5)  # np.inf becomes 1
            assert dataset.maxshape == (None, 4, 5)  # Resizable first dimension

    def test_create_virtual_dataset_container(self, tmp_path):
        """Test virtual dataset container creation"""
        filename = os.path.join(str(tmp_path), "test_container.h5")
        task_ids = [0, 1, 2]
        worker_filenames = [
            os.path.join(str(tmp_path), f"worker_{i}.h5") for i in task_ids
        ]
        result_shape = (3, 4, 5)
        stacking_dim = 0
        payload_dir = str(tmp_path)
        
        # Create worker files first
        for worker_file in worker_filenames:
            with h5py.File(worker_file, "w") as h5f:
                h5f.create_dataset("result_0", shape=(4, 5), data=np.zeros((4, 5)))
        
        HDF5ContainerFactory.create_virtual_dataset_container(
            filename, task_ids, worker_filenames, result_shape, 
            stacking_dim, "float", payload_dir
        )
        
        # Verify container was created
        assert os.path.exists(filename)
        
        # Verify virtual dataset was created
        with h5py.File(filename, "r") as h5f:
            assert "result_0" in h5f
            dataset = h5f["result_0"]
            assert dataset.shape == result_shape
            assert dataset.dtype == np.dtype('float64')
            # Verify it's a virtual dataset by checking if it has virtual layout attributes
            # Note: h5py.VirtualDataset may not be available in all versions


class TestOutputSetupIntegration:
    """Integration tests for output setup functionality"""

    def test_complete_setup_workflow(self, tmp_path):
        """Test complete output setup workflow"""
        # Create output directory
        output_dir = HDF5ContainerFactory.create_payload_directory(str(tmp_path), "test_func")
        
        # Create worker filenames
        task_ids = [0, 1, 2]
        worker_filenames = [
            os.path.join(output_dir, f"test_func_{task_id}.h5") for task_id in task_ids
        ]
        
        # Create worker files
        for worker_file in worker_filenames:
            with h5py.File(worker_file, "w") as h5f:
                h5f.create_dataset("result_0", shape=(4, 5), data=np.zeros((4, 5)))
        
        # Create virtual container
        container_file = os.path.join(str(tmp_path), "test_func.h5")
        result_shape = (3, 4, 5)
        stacking_dim = 0
        
        HDF5ContainerFactory.create_virtual_dataset_container(
            container_file, task_ids, worker_filenames, result_shape,
            stacking_dim, "float", output_dir
        )
        
        # Verify everything was created correctly
        assert os.path.exists(container_file)
        for worker_file in worker_filenames:
            assert os.path.exists(worker_file)