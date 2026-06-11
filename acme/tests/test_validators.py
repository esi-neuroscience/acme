#
# Test module for ACME validation functions
#
# Copyright © 2026 Ernst Strüngmann Institute (ESI) of the Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import pytest
import numpy as np
import os
import tempfile
import types

# Local imports
from acme.validators import (
    validate_pmap,
    validate_output_flags,
    validate_result_shape,
    validate_logfile,
    validate_stop_client,
    validate_n_workers,
)


class TestValidateParallelMapInstance:
    """Tests for validate_parallelmap_instance"""

    def test_valid_parallelmap_instance(self):
        """Test that valid ParallelMap instance passes"""

        class ParallelMap:
            def __init__(self):
                pass

        pmap = ParallelMap()
        # Should not raise
        validate_pmap(pmap)

    def test_invalid_parallelmap_instance(self):
        """Test that non-ParallelMap instance raises TypeError"""
        fake_pmap = "not a ParallelMap"
        with pytest.raises(TypeError, match="has to be a `ParallelMap` instance"):
            validate_pmap(fake_pmap)

    def test_object_without_class_name(self):
        """Test object that doesn't have __class__.__name__"""

        # Create a mock object that doesn't have the expected class name
        class FakeParallelMap:
            pass

        fake_pmap = FakeParallelMap()
        with pytest.raises(TypeError, match="has to be a `ParallelMap` instance"):
            validate_pmap(fake_pmap)


class TestValidateBooleanFlags:
    """Tests for validate_boolean_flags"""

    def test_valid_boolean_flags(self):
        """Test that valid boolean flags pass"""
        config = {
            "write_worker_results": True,
            "write_pickle": False,
            "single_file": False,
            "output_dir": None,
            "result_shape": None,
        }
        # Should not raise
        validate_output_flags(config)

    def test_invalid_write_worker_results_type(self):
        """Test that non-boolean write_worker_results raises TypeError"""
        config = {
            "write_worker_results": "true",
            "write_pickle": False,
            "single_file": False,
        }
        with pytest.raises(
            TypeError, match="`write_worker_results` has to be `True` or `False`"
        ):
            validate_output_flags(config)

    def test_invalid_single_file_type(self):
        """Test that non-boolean single_file raises TypeError"""
        config = {
            "write_worker_results": True,
            "single_file": "yes",
            "write_pickle": False,
        }
        with pytest.raises(
            TypeError, match="`single_file` has to be `True` or `False`"
        ):
            validate_output_flags(config)

    def test_invalid_write_pickle_type(self):
        """Test that non-boolean write_pickle raises TypeError"""
        config = {
            "write_worker_results": True,
            "single_file": False,
            "write_pickle": "no",
        }
        with pytest.raises(
            TypeError, match="`write_pickle` has to be `True` or `False`"
        ):
            validate_output_flags(config)

    def test_pickle_with_single_file_raises(self):
        """Test that pickle with single_file raises ValueError"""
        config = {
            "write_worker_results": True,
            "write_pickle": True,
            "single_file": True,
        }
        with pytest.raises(ValueError, match="does not support single output file"):
            validate_output_flags(config)


class TestValidateResultShape:
    """Tests for validate_result_shape"""

    def test_none_result_shape(self):
        """Test that None result shape returns None"""
        shape, stacking_dim, dtype = validate_result_shape(None, "float", 10, True)
        assert shape is None
        assert stacking_dim is None
        assert dtype is None

    def test_valid_result_shape(self):
        """Test that valid result shape passes"""
        test_shape = (10, 5, None)
        shape, stacking_dim, dtype = validate_result_shape(
            test_shape, "float", 10, True
        )

        assert shape == (10, 5, 10)
        assert stacking_dim == 2
        assert dtype == np.dtype("float")

    def test_invalid_result_shape_type(self):
        """Test that non-list/tuple result shape raises TypeError"""
        with pytest.raises(TypeError, match="has to be either `None` or tuple"):
            validate_result_shape("invalid", "float", 10, True)

    def test_invalid_result_dtype_type(self):
        """Test that non-string result dtype raises TypeError"""
        with pytest.raises(TypeError, match="`result_dtype` has to be a string"):
            validate_result_shape((10, None), 123, 10, True)

    def test_result_shape_missing_none(self):
        """Test that result_shape without exactly one None raises ValueError"""
        with pytest.raises(ValueError, match="must contain exactly one `None` entry"):
            validate_result_shape((10, 5, 3), "float", 10, True)

    def test_result_shape_multiple_nones(self):
        """Test that result_shape with multiple Nones raises ValueError"""
        with pytest.raises(ValueError, match="must contain exactly one `None` entry"):
            validate_result_shape((None, 5, None), "float", 10, True)

    def test_result_shape_with_inf_without_write_worker_results(self):
        """Test that np.inf without write_worker_results raises ValueError"""
        test_shape = (10, np.inf, None)
        with pytest.raises(
            ValueError, match="only valid if `write_worker_results` is `True`"
        ):
            validate_result_shape(test_shape, "float", 10, False)

    def test_result_shape_multiple_infs_raises(self):
        """Test that multiple np.inf values raises ValueError - must have one None"""
        test_shape = (np.inf, np.inf, None)  # Must have exactly one None
        with pytest.raises(ValueError, match="cannot use more than one `np.inf"):
            validate_result_shape(test_shape, "float", 10, True)

    def test_result_shape_non_numerical_values(self):
        """Test that non-numerical values in result_shape raises ValueError"""
        test_shape = (10, "invalid", None)
        with pytest.raises(ValueError, match="must only contain numerical values"):
            validate_result_shape(test_shape, "float", 10, True)

    def test_result_shape_negative_values(self):
        """Test that negative values in result_shape raises ValueError"""
        test_shape = (10, -5, None)
        with pytest.raises(ValueError, match="must only contain non-negative integers"):
            validate_result_shape(test_shape, "float", 10, True)

    def test_result_shape_float_values(self):
        """Test that non-integer float values in result_shape raises ValueError"""
        test_shape = (10, 5.5, None)
        with pytest.raises(ValueError, match="must only contain non-negative integers"):
            validate_result_shape(test_shape, "float", 10, True)

    def test_invalid_result_dtype(self):
        """Test that invalid result dtype raises TypeError"""
        with pytest.raises(TypeError, match="has to be a valid NumPy datatype"):
            validate_result_shape((10, None), "invalid_dtype", 10, True)


class TestValidateLogfile:
    """Tests for validate_logfile"""

    def test_none_logfile_with_write_worker_results(self):
        """Test that None logfile with write_worker_results=True creates auto log"""

        # We need to mock the inspect.getfile to avoid trying to get function location
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple function for testing
            def test_func():
                pass

            # Temporarily store the function in a real file location
            func_file = os.path.join(tmpdir, "test_module.py")
            with open(func_file, "w") as f:
                f.write("def test_func(): pass\n")

            # Mock the inspect.getfile to return our test file
            module = types.ModuleType("test_module")
            with open(func_file) as f:
                exec(f.read(), module.__dict__)
            test_func = module.test_func

            logfile = validate_logfile(None, True, None, out_dir=tmpdir)
            assert logfile is not None
            assert "ACME_<lambda>" in logfile
            assert logfile.endswith(".log")

    def test_none_logfile_without_write_worker_results(self):
        """Test that None logfile with write_worker_results=False returns None"""
        logfile = validate_logfile(None, False, "test_func")
        assert logfile is None

    def test_true_logfile_with_out_dir(self):
        """Test that True logfile with out_dir uses output directory"""
        out_dir = "/tmp/test_output"
        logfile = validate_logfile(True, True, None, out_dir=out_dir)
        assert logfile.startswith(out_dir)
        assert "ACME_<lambda>" in logfile

    def test_false_logfile(self):
        """Test that False logfile returns None"""
        logfile = validate_logfile(False, True, "test_func")
        assert logfile is None

    def test_string_logfile_path(self):
        """Test that string logfile path is normalized"""

        with tempfile.TemporaryDirectory() as tmpdir:
            logfile_path = os.path.join(tmpdir, "test.log")
            logfile = validate_logfile(logfile_path, True, "test_func")
            assert os.path.isabs(logfile)
            assert logfile.endswith("test.log")

    def test_directory_logfile_raises(self):
        """Test that directory logfile raises IOError"""

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(IOError, match="a directory"):
                validate_logfile(tmpdir, True, "test_func")

    def test_invalid_logfile_type(self):
        """Test that invalid logfile type raises TypeError"""
        with pytest.raises(
            TypeError, match="has to be `None`, `True`, `False` or a valid file-name"
        ):
            validate_logfile(123, True, "test_func")


class TestValidateStopClient:
    """Tests for validate_stop_client"""

    def test_invalid_string_stop_client(self):
        """Test that invalid string stop_client raises ValueError"""
        with pytest.raises(ValueError, match="has to be 'auto' or Boolean"):
            validate_stop_client("invalid")

    def test_invalid_type_stop_client(self):
        """Test that invalid type stop_client raises TypeError"""
        with pytest.raises(
            TypeError,
            match="`stop_client` has to be `True` or `False`, not <class 'int'>",
        ):
            validate_stop_client(123)


class TestValidateNWorkers:
    """Tests for validate_n_workers"""

    def test_auto_with_slurm(self):
        """Test that 'auto' with SLURM returns n_calls"""
        result = validate_n_workers("auto", 10, True)
        assert result == 10

    def test_auto_without_slurm(self):
        """Test that 'auto' without SLURM returns None"""
        result = validate_n_workers("auto", 10, False)
        assert result is None

    def test_invalid_string(self):
        """Test that invalid string raises ValueError"""
        with pytest.raises(ValueError, match="has to be 'auto' or an integer >= 1"):
            validate_n_workers("invalid", 10, True)

    def test_valid_integer(self):
        """Test that valid integer returns the integer"""
        result = validate_n_workers(5, 10, True)
        assert result == 5

    def test_zero_workers(self):
        """Test that zero workers raises exception"""
        with pytest.raises((ValueError, TypeError)):
            validate_n_workers(0, 10, True)

    def test_negative_workers(self):
        """Test that negative workers raises exception"""
        with pytest.raises((ValueError, TypeError)):
            validate_n_workers(-5, 10, True)

    def test_float_workers(self):
        """Test that float workers raises ValueError"""
        with pytest.raises(ValueError):
            validate_n_workers(5.5, 10, True)

    def test_list_workers(self):
        """Test that list workers raises TypeError"""
        with pytest.raises(TypeError):
            validate_n_workers([1, 2, 3], 10, True)
