#
# Test module for ACME configuration classes
#
# Copyright © 2025 Ernst Strüngmann Institute (ESI) for Neuroscience
# in Cooperation with Max Planck Society
#
# SPDX-License-Identifier: BSD-3-Clause
#

# Builtin/3rd party package imports
import pytest
from unittest.mock import Mock, MagicMock

# Local imports
from acme.config import ACMEConfig
from acme.shared import is_slurm_node


class TestACMEConfig:
    """Tests for ACMEConfig dataclass"""

    def test_default_values(self):
        """Test that ACMEConfig has correct default values"""
        config = ACMEConfig()

        assert config.n_workers == "auto"
        assert config.dryrun is False
        assert config.setup_timeout == 60
        assert config.setup_interactive is True
        assert config.stop_client == "auto"
        assert config.write_worker_results is True
        assert config.write_pickle is False
        assert config.single_file is False
        assert config.output_dir is None
        assert config.result_shape is None
        assert config.result_dtype == "float"
        assert config.partition == "auto"
        assert config.mem_per_worker == "auto"
        assert config.verbose is None
        assert config.logfile is None

    def test_custom_values(self):
        """Test creating ACMEConfig with custom values"""
        config = ACMEConfig(
            n_workers=10,
            write_worker_results=False,
            partition="8GBXS",
            mem_per_worker="2GB",
            setup_timeout=120,
        )

        assert config.n_workers == 10
        assert config.write_worker_results is False
        assert config.partition == "8GBXS"
        assert config.mem_per_worker == "2GB"
        assert config.setup_timeout == 120

    def test_validate_valid_config(self):
        """Test that valid configuration passes validation"""
        config = ACMEConfig(
            n_workers=5,
            write_worker_results=True,
            single_file=False,
            write_pickle=False,
        )

        # Should not raise
        config.validate()

    def test_validate_invalid_boolean_combination(self):
        """Test that invalid boolean combination raises ValueError"""
        config = ACMEConfig(write_pickle=True, single_file=True)

        with pytest.raises(ValueError, match="does not support single output file"):
            config.validate()

    def test_validate_invalid_stop_client(self):
        """Test that invalid stop_client raises ValueError"""
        config = ACMEConfig(stop_client="invalid")

        with pytest.raises(ValueError, match="has to be 'auto' or Boolean"):
            config.validate()

    def test_validate_invalid_n_workers(self):
        """Test that invalid n_workers raises ValueError"""
        config = ACMEConfig(n_workers=0)

        with pytest.raises(
            ValueError,
            match="<_scalar_parser> `n_workers` has to be an integer between 1 and inf, not 0",
        ):
            config.validate()

    def test_validate_invalid_result_shape_type(self):
        """Test that invalid result_shape type raises TypeError"""
        config = ACMEConfig(result_shape="invalid")

        with pytest.raises(TypeError, match="has to be either `None` or tuple"):
            config.validate()

    def test_validate_invalid_partition_type(self):
        """Test that invalid partition type raises TypeError"""
        if not is_slurm_node():
            return
        config = ACMEConfig(partition=123)

        with pytest.raises(
            TypeError, match="has to be 'auto' or a valid partition name"
        ):
            config.validate()

    def test_validate_invalid_setup_timeout_type(self):
        """Test that invalid setup_timeout type raises TypeError"""
        config = ACMEConfig(setup_timeout="60")

        with pytest.raises(
            TypeError,
            match="<_scalar_parser> `setup_timeout` has to be a scalar, not <class 'str'>",
        ):
            config.validate()

    def test_validate_negative_setup_timeout(self):
        """Test that negative setup_timeout raises ValueError"""
        config = ACMEConfig(setup_timeout=-10)

        with pytest.raises(
            ValueError,
            match="<_scalar_parser> `setup_timeout` has to be an integer between 0 and inf, not -10",
        ):
            config.validate()

    def test_config_with_result_shape(self):
        """Test config with result_shape configured"""
        config = ACMEConfig(result_shape=(10, 5, None), result_dtype="float32")

        assert config.result_shape == (10, 5, None)
        assert config.result_dtype == "float32"

        # Should not raise validation error
        config.validate()

    def test_config_immutability_concept(self):
        """Test that config objects maintain their state"""
        config1 = ACMEConfig(n_workers=5)
        config2 = ACMEConfig(n_workers=10)

        assert config1.n_workers == 5
        assert config2.n_workers == 10

        # Modifying one shouldn't affect the other
        config1.n_workers = 15
        assert config1.n_workers == 15
        assert config2.n_workers == 10


class TestACMEConfigEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_large_n_workers(self):
        """Test config with very large n_workers"""
        config = ACMEConfig(n_workers=10000)
        config.validate()
        assert config.n_workers == 10000

    def test_very_long_setup_timeout(self):
        """Test config with very long setup_timeout"""
        config = ACMEConfig(setup_timeout=3600)  # 1 hour
        config.validate()
        assert config.setup_timeout == 3600

    def test_complex_result_shape(self):
        """Test config with complex result_shape"""
        config = ACMEConfig(
            result_shape=(100, 50, 25, None, 10), result_dtype="complex128"
        )
        config.validate()
        assert len(config.result_shape) == 5

    def test_config_with_all_false(self):
        """Test config with most boolean flags set to False"""
        config = ACMEConfig(
            dryrun=False,
            write_worker_results=False,
            write_pickle=False,
            single_file=False,
            setup_interactive=False,
        )
        config.validate()

    def test_config_with_all_true(self):
        """Test config with most boolean flags set to True"""
        config = ACMEConfig(
            dryrun=True,
            write_worker_results=True,
            write_pickle=True,  # Add this to trigger the error
            single_file=True,
            setup_interactive=True,
        )

        # Should fail due to incompatible combination
        with pytest.raises(ValueError, match="does not support single output file"):
            config.validate()

    def test_empty_partition_string(self):
        """Test config with empty partition string"""
        config = ACMEConfig(partition="")
        config.validate()
        assert config.partition == ""

    def test_none_mem_per_worker(self):
        """Test that mem_per_worker can handle None-like string"""
        config = ACMEConfig(mem_per_worker="auto")
        config.validate()
        assert config.mem_per_worker == "auto"
