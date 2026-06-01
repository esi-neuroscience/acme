#
# Tests for post_processor module
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
from unittest.mock import Mock, patch, MagicMock
from acme.results.post_processor import ResultPostProcessor


class TestResultPostProcessor:
    """Test result post-processor functionality"""

    def test_process_futures_memory_collection(self):
        """Test processing futures with in-memory collection"""
        # Create mock client and futures
        mock_client = Mock()
        mock_futures = [Mock(), Mock(), Mock()]
        
        # Mock gather to return test results
        test_results = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9])
        ]
        mock_client.gather.return_value = test_results
        
        # Create post-processor
        post_processor = ResultPostProcessor(mock_client, None)
        
        # Test memory collection
        result = post_processor.process_futures(
            futures=mock_futures,
            collect_results=True,
            result_shape=None,
            stacking_dim=0,
            result_dtype="float",
            acme_func=Mock(),
            original_func=Mock(),
            kwargv={}
        )
        
        # Should return collected results
        assert result == test_results
        mock_client.gather.assert_called_once_with(mock_futures)

    def test_process_futures_memory_collection_with_shape(self):
        """Test processing futures with in-memory collection and result shape"""
        # Create mock client and futures
        mock_client = Mock()
        mock_futures = [Mock(), Mock(), Mock()]
        
        # Mock gather to return test results
        test_results = [
            [np.array([1, 2]), np.array([10])],
            [np.array([3, 4]), np.array([20])],
            [np.array([5, 6]), np.array([30])]
        ]
        mock_client.gather.return_value = test_results
        
        # Create post-processor
        post_processor = ResultPostProcessor(mock_client, None)
        
        # Test memory collection with shape
        result = post_processor.process_futures(
            futures=mock_futures,
            collect_results=True,
            result_shape=(3, 2),
            stacking_dim=0,
            result_dtype="float",
            acme_func=Mock(),
            original_func=Mock(),
            kwargv={}
        )
        
        # Should return stacked array and additional values
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert result[0].shape == (3, 2)
        assert np.array_equal(result[0], np.array([[1, 2], [3, 4], [5, 6]]))
        assert result[1] == [10, 20, 30]

    def test_process_futures_file_results_pickle(self, tmp_path):
        """Test processing futures with file results (pickle)"""
        # Create mock client and futures
        mock_client = Mock()
        mock_futures = [Mock(), Mock()]
        
        # Create post-processor
        post_processor = ResultPostProcessor(mock_client, str(tmp_path))
        
        # Mock kwargv for pickle results
        kwargv = {
            "outFile": [
                os.path.join(str(tmp_path), "result1.pickle"),
                os.path.join(str(tmp_path), "result2.pickle")
            ]
        }
        
        # Create some dummy pickle files
        test_data = [{"result": 1}, {"result": 2}]
        for i, fname in enumerate(kwargv["outFile"]):
            with open(fname, "wb") as f:
                pickle.dump(test_data[i], f)
        
        # Test file results processing (pickle)
        result = post_processor.process_futures(
            futures=mock_futures,
            collect_results=False,
            result_shape=None,
            stacking_dim=0,
            result_dtype="float",
            acme_func=Mock(),
            original_func=Mock(),
            kwargv=kwargv
        )
        
        # Should return success message
        assert isinstance(result, str)
        assert "Results have been saved to" in result

    def test_process_futures_file_results_single_file(self, tmp_path):
        """Test processing futures with file results (single file)"""
        # Create mock client and futures
        mock_client = Mock()
        mock_futures = [Mock(), Mock()]
        
        # Create a container file
        container_file = os.path.join(str(tmp_path), "results.h5")
        with h5py.File(container_file, "w") as h5f:
            h5f.create_dataset("result_0", shape=(2, 3), data=np.zeros((2, 3)))
        
        # Create post-processor
        post_processor = ResultPostProcessor(mock_client, container_file)
        
        # Mock kwargv for single file results
        kwargv = {
            "singleFile": [True],
            "outFile": [container_file]
        }
        
        # Test file results processing (single file)
        result = post_processor.process_futures(
            futures=mock_futures,
            collect_results=False,
            result_shape=None,
            stacking_dim=0,
            result_dtype="float",
            acme_func=Mock(),
            original_func=Mock(),
            kwargv=kwargv
        )
        
        # Should return success message
        assert isinstance(result, str)
        assert "Results have been saved to" in result

    def test_process_futures_no_worker_results(self):
        """Test processing futures when no worker results are written"""
        # Create mock client and futures
        mock_client = Mock()
        mock_futures = [Mock(), Mock()]
        
        # Create post-processor
        post_processor = ResultPostProcessor(mock_client, None)
        
        # Test when no worker results are written
        result = post_processor.process_futures(
            futures=mock_futures,
            collect_results=False,
            result_shape=None,
            stacking_dim=0,
            result_dtype="float",
            acme_func=Mock(),  # Same function - no worker results
            original_func=Mock(),
            kwargv={}
        )
        
        # Should return None
        assert result is None


class TestResultPostProcessorLogging:
    """Test logging functionality in result post-processor"""

    @patch('acme.results.post_processor.log')
    def test_log_output_mode(self, mock_log):
        """Test output mode logging"""
        post_processor = ResultPostProcessor(Mock(), None)
        
        kwargv = {
            "singleFile": [True]
        }
        
        post_processor._log_output_mode(True, kwargv)
        
        # Should have logged the output mode
        mock_log.debug.assert_called_once()
        call_args = mock_log.debug.call_args[0]
        assert "write_worker_results = True" in call_args[0]
        assert "single_file = True" in call_args[0]
        assert "write_pickle = False" in call_args[0]


class TestResultPostProcessorIntegration:
    """Integration tests for result post-processor"""

    def test_complete_workflow_memory(self):
        """Test complete workflow with memory collection"""
        # Create mock client and futures
        mock_client = Mock()
        mock_futures = [Mock(), Mock(), Mock()]
        
        # Mock gather to return test results
        test_results = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([7, 8, 9])
        ]
        mock_client.gather.return_value = test_results
        
        # Create post-processor
        post_processor = ResultPostProcessor(mock_client, None)
        
        # Test complete workflow
        result = post_processor.process_futures(
            futures=mock_futures,
            collect_results=True,
            result_shape=None,
            stacking_dim=0,
            result_dtype="float",
            acme_func=Mock(),
            original_func=Mock(),
            kwargv={}
        )
        
        # Should return collected results
        assert result == test_results
        mock_client.gather.assert_called_once_with(mock_futures)

    def test_complete_workflow_file_results(self, tmp_path):
        """Test complete workflow with file results"""
        # Create mock client and futures
        mock_client = Mock()
        mock_futures = [Mock(), Mock()]
        
        # Create post-processor
        post_processor = ResultPostProcessor(mock_client, str(tmp_path))
        
        # Mock kwargv for pickle results
        kwargv = {
            "outFile": [
                os.path.join(str(tmp_path), "result1.pickle"),
                os.path.join(str(tmp_path), "result2.pickle")
            ]
        }
        
        # Create some dummy pickle files
        test_data = [{"result": 1}, {"result": 2}]
        for i, fname in enumerate(kwargv["outFile"]):
            with open(fname, "wb") as f:
                pickle.dump(test_data[i], f)
        
        # Test complete workflow with file results
        result = post_processor.process_futures(
            futures=mock_futures,
            collect_results=False,
            result_shape=None,
            stacking_dim=0,
            result_dtype="float",
            acme_func=Mock(),
            original_func=Mock(),
            kwargv=kwargv
        )
        
        # Should return success message
        assert isinstance(result, str)
        assert "Results have been saved to" in result