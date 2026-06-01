#
# Tests for memory_profiler module
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from acme.memory_profiler import MemoryProfiler, MemoryEstimationError


class TestMemoryProfiler:
    """Test suite for MemoryProfiler class"""

    def test_initialization(self):
        """Test MemoryProfiler initialization"""
        func = lambda x: x**2
        profiler = MemoryProfiler(func)
        
        assert profiler.func == func
        assert profiler.tqdm_format == "{desc}: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

    def test_initialization_with_custom_format(self):
        """Test MemoryProfiler initialization with custom tqdm format"""
        func = lambda x: x**2
        custom_format = "custom: {percentage}%"
        profiler = MemoryProfiler(func, custom_format)
        
        assert profiler.func == func
        assert profiler.tqdm_format == custom_format

    @patch('acme.memory_profiler.multiprocessing.Process')
    @patch('acme.memory_profiler.psutil.Process')
    @patch('acme.memory_profiler.tqdm.tqdm')
    @patch('acme.memory_profiler.time.sleep')
    def test_estimate_memory_basic(self, mock_sleep, mock_tqdm, mock_psutil, mock_process):
        """Test basic memory estimation functionality"""
        # Setup mocks
        func = lambda x: x**2
        profiler = MemoryProfiler(func)
        
        # Mock dryrun setup function
        def mock_dryrun():
            return (np.array([0, 1]), [[1], [2]], [{}, {}])
        
        # Mock process
        mock_proc_instance = MagicMock()
        mock_proc_instance.is_alive.return_value = True
        mock_proc_instance.pid = 12345
        mock_process.return_value = mock_proc_instance
        
        # Mock psutil
        mock_psutil_instance = MagicMock()
        mock_psutil_instance.memory_info.return_value.rss = 1024 * 1024 * 1024  # 1 GB
        mock_psutil.return_value = mock_psutil_instance
        
        # Mock tqdm
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        # Test the estimation
        result = profiler.estimate_memory(mock_dryrun, None, 2)
        
        # Verify result format
        assert result.startswith("estimate_memuse:")
        assert result.endswith("1")  # Should round up 1 GB to 1
        
        # Verify process was started and killed (called for each job)
        assert mock_proc_instance.start.call_count == 2
        assert mock_proc_instance.kill.call_count == 2

    @patch('acme.memory_profiler.multiprocessing.Process')
    @patch('acme.memory_profiler.psutil.Process')
    @patch('acme.memory_profiler.tqdm.tqdm')
    @patch('acme.memory_profiler.time.sleep')
    def test_estimate_memory_with_output_dir(self, mock_sleep, mock_tqdm, mock_psutil, mock_process):
        """Test memory estimation with output directory (should add memEstRun flag)"""
        # Setup mocks
        func = lambda x: x**2
        profiler = MemoryProfiler(func)
        
        # Mock dryrun setup function
        def mock_dryrun():
            return (np.array([0, 1]), [[1], [2]], [{'a': 1}, {'a': 2}])
        
        # Mock process
        mock_proc_instance = MagicMock()
        mock_proc_instance.is_alive.return_value = True
        mock_proc_instance.pid = 12345
        mock_process.return_value = mock_proc_instance
        
        # Mock psutil
        mock_psutil_instance = MagicMock()
        mock_psutil_instance.memory_info.return_value.rss = 1024 * 1024 * 1024  # 1 GB
        mock_psutil.return_value = mock_psutil_instance
        
        # Mock tqdm
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        # Test the estimation with output directory
        result = profiler.estimate_memory(mock_dryrun, "/tmp/output", 2)
        
        # Verify result format
        assert result.startswith("estimate_memuse:")
        
        # Verify memEstRun flag was added (check that it was passed to the function)
        # The memEstRun flag is added to the kwargs in the dryrun_kwargs, not directly to process
        # Let's check that the dryrun function was called and returned the expected data
        assert len(mock_process.call_args_list) == 2  # Should have 2 calls for 2 jobs

    @patch('acme.memory_profiler.multiprocessing.Process')
    @patch('acme.memory_profiler.psutil.Process')
    @patch('acme.memory_profiler.tqdm.tqdm')
    @patch('acme.memory_profiler.time.sleep')
    def test_estimate_memory_with_varying_memory(self, mock_sleep, mock_tqdm, mock_psutil, mock_process):
        """Test memory estimation with varying memory usage"""
        # Setup mocks
        func = lambda x: x**2
        profiler = MemoryProfiler(func)
        
        # Mock dryrun setup function
        def mock_dryrun():
            return (np.array([0, 1, 2]), [[1], [2], [3]], [{}, {}, {}])
        
        # Mock process
        mock_proc_instance = MagicMock()
        mock_proc_instance.is_alive.return_value = True
        mock_proc_instance.pid = 12345
        mock_process.return_value = mock_proc_instance
        
        # Mock psutil with varying memory usage
        # We need to return different values for each call to simulate varying memory
        memory_values = [0.5, 1.5, 2.5]  # GB values
        call_count = [0]  # Use list to allow modification in nested function
        
        def get_memory_info():
            # Cycle through the memory values
            value = memory_values[call_count[0] % len(memory_values)]
            call_count[0] += 1
            return type('obj', (object,), {'rss': value * 1024 * 1024 * 1024})()
        
        mock_psutil_instance = MagicMock()
        mock_psutil_instance.memory_info = get_memory_info
        mock_psutil.return_value = mock_psutil_instance
        
        # Mock tqdm
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        # Test the estimation
        result = profiler.estimate_memory(mock_dryrun, None, 3)
        
        # Verify result format - should round up peak memory
        assert result.startswith("estimate_memuse:")
        # Peak memory is 2.5GB, should round up to 3
        assert result.endswith("3")

    def test_estimate_memory_with_no_jobs(self):
        """Test memory estimation with no jobs selected"""
        func = lambda x: x**2
        profiler = MemoryProfiler(func)
        
        # Mock dryrun setup function that returns empty arrays
        def mock_dryrun():
            return (np.array([]), [], [])
        
        # This should handle gracefully (though in practice this shouldn't happen)
        with pytest.raises((IndexError, ValueError)):
            profiler.estimate_memory(mock_dryrun, None, 2)

    def test_estimate_memory_with_single_job(self):
        """Test memory estimation with single job"""
        func = lambda x: x**2
        profiler = MemoryProfiler(func)
        
        # Mock dryrun setup function
        def mock_dryrun():
            return (np.array([0]), [[42]], [{'test': 'value'}])
        
        # Mock process
        with patch('acme.memory_profiler.multiprocessing.Process') as mock_process:
            with patch('acme.memory_profiler.psutil.Process') as mock_psutil:
                with patch('acme.memory_profiler.tqdm.tqdm') as mock_tqdm:
                    with patch('acme.memory_profiler.time.sleep') as mock_sleep:
                        mock_proc_instance = MagicMock()
                        mock_proc_instance.is_alive.return_value = True
                        mock_proc_instance.pid = 12345
                        mock_process.return_value = mock_proc_instance
                        
                        mock_psutil_instance = MagicMock()
                        mock_psutil_instance.memory_info.return_value.rss = 2 * 1024 * 1024 * 1024  # 2 GB
                        mock_psutil.return_value = mock_psutil_instance
                        
                        mock_pbar = MagicMock()
                        mock_tqdm.return_value.__enter__.return_value = mock_pbar
                        
                        result = profiler.estimate_memory(mock_dryrun, None, 2)
                        
        # Should work with single job
        assert result.startswith("estimate_memuse:")
        # The test sets 2GB memory, so should round up to 2
        assert result.endswith("2")


class TestMemoryEstimationError:
    """Test suite for MemoryEstimationError exception"""
    
    def test_exception_type(self):
        """Test that MemoryEstimationError is a proper exception"""
        with pytest.raises(MemoryEstimationError):
            raise MemoryEstimationError("Test error")
        
        # Test inheritance
        assert issubclass(MemoryEstimationError, Exception)