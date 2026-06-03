#
# Tests for argument_processor module
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

from unittest.mock import patch, MagicMock
from acme.argument_processor import ArgumentProcessor


class TestArgumentProcessor:
    """Test suite for ArgumentProcessor class"""

    def test_dryrun_setup_basic(self):
        """Test basic dryrun setup functionality"""
        # Test data
        argv = [[1, 2, 3], [4, 5, 6]]
        kwargv = {"a": [7, 8, 9], "b": [10, 11, 12]}
        n_calls = 3
        processor = ArgumentProcessor(argv, kwargv, n_calls)

        # Call method
        indices, args, kwargs = processor.dryrun_setup(n_runs=2)

        # Verify results
        assert len(indices) == 2
        assert len(args) == 2
        assert len(kwargs) == 2
        assert all(0 <= idx < 3 for idx in indices)
        assert len(set(indices)) == 2  # Should be unique

        # Verify each result has correct structure
        for i in range(2):
            assert len(args[i]) == 2  # Should have 2 positional args
            assert len(kwargs[i]) == 2  # Should have 2 keyword args
            assert "a" in kwargs[i]
            assert "b" in kwargs[i]

    def test_dryrun_setup_with_single_elements(self):
        """Test dryrun setup with single element arguments"""
        # Test data with single elements
        argv = [[1], [4]]  # Single element lists
        kwargv = {"a": [7], "b": [10]}  # Single element lists
        n_calls = 1
        processor = ArgumentProcessor(argv, kwargv, n_calls)

        # Call method
        indices, args, kwargs = processor.dryrun_setup(n_runs=1)

        # Verify results
        assert len(indices) == 1
        assert indices[0] == 0
        assert args[0] == [1, 4]
        assert kwargs[0] == {"a": 7, "b": 10}

    def test_dryrun_setup_with_mixed_elements(self):
        """Test dryrun setup with mixed single and multiple elements"""
        # Test data with mixed elements
        argv = [[1, 2, 3], [4]]  # One multi-element, one single-element
        kwargv = {"a": [7, 8, 9], "b": [10]}  # One multi-element, one single-element
        n_calls = 3
        processor = ArgumentProcessor(argv, kwargv, n_calls)

        # Call method
        indices, args, kwargs = processor.dryrun_setup(n_runs=2)

        # Verify results
        assert len(indices) == 2
        for i in range(2):
            assert len(args[i]) == 2
            assert len(kwargs[i]) == 2
            # Single element args should be repeated
            assert args[i][1] == 4  # Single element repeated
            assert kwargs[i]["b"] == 10  # Single element repeated

    def test_dryrun_setup_auto_n_runs(self):
        """Test automatic n_runs calculation"""
        # Test with different n_calls values
        test_cases = [
            (100, 5),  # 5% of 100 = 5, min(5, 5) = 5
            (20, 5),  # 5% of 20 = 1, max(5, 1) = 5
            (5, 5),  # 5% of 5 = 0.25, max(5, 1) = 5, but min(5, 5) = 5
            (1, 1),  # Edge case: min(1, max(5, min(1, 0.05))) = 1
        ]

        for n_calls, expected_min_runs in test_cases:
            argv = [[i for i in range(n_calls)]]
            kwargv = {"a": [i for i in range(n_calls)]}
            processor = ArgumentProcessor(argv, kwargv, n_calls)

            indices, args, kwargs = processor.dryrun_setup(n_runs=None)

            # Should pick at least the minimum expected runs
            assert len(indices) >= expected_min_runs
            assert len(indices) <= n_calls

    def test_dryrun_setup_edge_cases(self):
        """Test edge cases for dryrun setup"""
        # Test with n_runs = 0
        argv = [[1, 2, 3]]
        kwargv = {"a": [4, 5, 6]}
        n_calls = 3
        processor = ArgumentProcessor(argv, kwargv, n_calls)

        indices, args, kwargs = processor.dryrun_setup(n_runs=0)
        assert len(indices) == 0
        assert len(args) == 0
        assert len(kwargs) == 0

        # Test with n_runs > n_calls
        indices, args, kwargs = processor.dryrun_setup(n_runs=10)
        assert len(indices) == 3  # Should be capped at n_calls

    @patch("acme.argument_processor.collections.abc.Sized")
    def test_broadcast_arguments_basic(self, mock_sized):
        """Test basic argument broadcasting"""
        # Test data
        argv = [[1], [4, 5, 6]]  # One single-element, one multi-element
        kwargv = {"a": [7], "b": [10, 11, 12]}  # One single-element, one multi-element
        n_calls = 3
        processor = ArgumentProcessor(argv, kwargv, n_calls)

        # Mock client
        mock_client = MagicMock()
        mock_future = MagicMock()
        mock_client.scatter.return_value = mock_future

        # Mock Sized check
        mock_sized.return_value = True

        # Call method
        new_argv, new_kwargv = processor.broadcast_arguments(mock_client)

        # Verify single-element args were broadcasted
        assert len(new_argv[0]) == 3  # Single element should be repeated 3 times
        # Should be the future (or future[0] if it was Sized)
        assert new_argv[0][0] == mock_future or new_argv[0][0] == mock_future[0]
        assert new_argv[1] == [4, 5, 6]  # Multi-element should be unchanged

        # Verify single-element kwargs were broadcasted
        assert len(new_kwargv["a"]) == 3  # Single element should be repeated 3 times
        # Should be the future (or future[0] if it was Sized)
        assert new_kwargv["a"][0] == mock_future or new_kwargv["a"][0] == mock_future[0]
        assert new_kwargv["b"] == [10, 11, 12]  # Multi-element should be unchanged

        # Verify scatter was called correctly
        mock_client.scatter.assert_any_call([1], broadcast=True)
        mock_client.scatter.assert_any_call([7], broadcast=True)

    @patch("acme.argument_processor.collections.abc.Sized")
    def test_broadcast_arguments_no_single_elements(self, mock_sized):
        """Test argument broadcasting with no single elements"""
        # Test data with all multi-element
        argv = [[1, 2, 3], [4, 5, 6]]
        kwargv = {"a": [7, 8, 9], "b": [10, 11, 12]}
        n_calls = 3
        processor = ArgumentProcessor(argv, kwargv, n_calls)

        # Mock client
        mock_client = MagicMock()

        # Call method
        new_argv, new_kwargv = processor.broadcast_arguments(mock_client)

        # Should be unchanged since no single elements
        assert new_argv == argv
        assert new_kwargv == kwargv

        # Scatter should not be called
        mock_client.scatter.assert_not_called()

    def test_format_kwarg_list_basic(self):
        """Test basic kwarg list formatting"""
        # Test data
        argv = [[1, 2, 3], [4, 5, 6]]
        kwargv = {"a": [1, 2, 3], "b": [4, 5, 6]}
        n_calls = 3
        processor = ArgumentProcessor(argv, kwargv, n_calls)

        # Call method
        kwarg_list = processor.format_kwarg_list()

        # Verify results
        assert len(kwarg_list) == 3
        assert kwarg_list[0] == {"a": 1, "b": 4}
        assert kwarg_list[1] == {"a": 2, "b": 5}
        assert kwarg_list[2] == {"a": 3, "b": 6}

    def test_format_kwarg_list_with_single_elements(self):
        """Test kwarg list formatting with single elements"""
        # Test data with single elements
        argv = [[1, 2, 3], [4]]
        kwargv = {"a": [1], "b": [4, 5, 6]}
        n_calls = 3
        processor = ArgumentProcessor(argv, kwargv, n_calls)

        # Call method
        kwarg_list = processor.format_kwarg_list()

        # Verify results - single element should be repeated
        assert len(kwarg_list) == 3
        assert kwarg_list[0] == {"a": 1, "b": 4}
        assert kwarg_list[1] == {"a": 1, "b": 5}
        assert kwarg_list[2] == {"a": 1, "b": 6}

    def test_format_kwarg_list_empty(self):
        """Test kwarg list formatting with empty kwargv"""
        # Test data
        argv = []
        kwargv = {}
        n_calls = 3
        processor = ArgumentProcessor(argv, kwargv, n_calls)

        # Call method
        kwarg_list = processor.format_kwarg_list()

        # Verify results
        assert len(kwarg_list) == 3
        assert kwarg_list[0] == {}
        assert kwarg_list[1] == {}
        assert kwarg_list[2] == {}

    def test_format_kwarg_list_single_call(self):
        """Test kwarg list formatting with single call"""
        # Test data
        argv = [[1], [2]]
        kwargv = {"a": [42], "b": [99]}
        n_calls = 1
        processor = ArgumentProcessor(argv, kwargv, n_calls)

        # Call method
        kwarg_list = processor.format_kwarg_list()

        # Verify results
        assert len(kwarg_list) == 1
        assert kwarg_list[0] == {"a": 42, "b": 99}


class TestIntegration:
    """Integration tests for ArgumentProcessor"""

    def test_full_workflow(self):
        """Test the full argument processing workflow"""
        # Original data
        original_argv = [[1], [4, 5, 6]]
        original_kwargv = {"a": [7], "b": [10, 11, 12]}
        n_calls = 3
        processor = ArgumentProcessor(original_argv, original_kwargv, n_calls)

        # Mock client
        mock_client = MagicMock()
        mock_future = MagicMock()
        mock_client.scatter.return_value = mock_future

        # Step 1: Dryrun setup
        indices, dryrun_args, dryrun_kwargs = processor.dryrun_setup(n_runs=2)

        # Verify dryrun results
        assert len(indices) == 2
        assert len(dryrun_args) == 2
        assert len(dryrun_kwargs) == 2

        # Step 2: Broadcast arguments
        with patch("acme.argument_processor.collections.abc.Sized") as mock_sized:
            mock_sized.return_value = True
            broadcasted_argv, broadcasted_kwargv = processor.broadcast_arguments(
                mock_client
            )

        # Verify broadcasting
        assert len(broadcasted_argv[0]) == 3
        # Should be the future (or future[0] if it was Sized)
        assert (
            broadcasted_argv[0][0] == mock_future
            or broadcasted_argv[0][0] == mock_future[0]
        )

        # Step 3: Format kwarg list
        kwarg_list = processor.format_kwarg_list()

        # Verify final format
        assert len(kwarg_list) == 3
        for kwarg_dict in kwarg_list:
            assert "a" in kwarg_dict
            assert "b" in kwarg_dict
            # Broadcasted future (or future[0] if it was Sized)
            assert kwarg_dict["a"] == mock_future or kwarg_dict["a"] == mock_future[0]
