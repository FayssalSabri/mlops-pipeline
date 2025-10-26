"""
Unit tests for the configuration management.
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from src.config import Config, get_config, get_setting


class TestConfig:
    """Test Config class functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        # Test app settings
        assert config.get("app.name") == "MLOps Pipeline"
        assert config.get("app.version") == "1.0.0"
        assert config.get("app.debug") is False
        assert config.get("app.host") == "0.0.0.0"
        assert config.get("app.port") == 8000

        # Test model settings
        assert config.get("model.type") == "logistic"
        assert config.get("model.path") == "models/model.pkl"

        # Test data settings
        assert config.get("data.test_size") == 0.2
        assert config.get("data.random_state") == 42

        # Test logging settings
        assert config.get("logging.level") == "INFO"

    def test_config_from_file(self):
        """Test loading configuration from file."""
        config_data = {
            "app": {"debug": True, "port": 9000},
            "model": {"type": "random_forest"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            config = Config(config_file=config_file)

            # Test overridden values
            assert config.get("app.debug") is True
            assert config.get("app.port") == 9000
            assert config.get("model.type") == "random_forest"

            # Test default values that weren't overridden
            assert config.get("app.name") == "MLOps Pipeline"
            assert config.get("data.test_size") == 0.2
        finally:
            os.unlink(config_file)

    def test_config_from_env(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "APP_DEBUG": "true",
            "APP_PORT": "9000",
            "MODEL_TYPE": "random_forest",
            "DATA_TEST_SIZE": "0.3",
            "LOG_LEVEL": "DEBUG",
        }

        # Set environment variables
        original_values = {}
        for key, value in env_vars.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            config = Config()

            # Test environment variable overrides
            assert config.get("app.debug") is True
            assert config.get("app.port") == 9000
            assert config.get("model.type") == "random_forest"
            assert config.get("data.test_size") == 0.3
            assert config.get("logging.level") == "DEBUG"
        finally:
            # Restore original environment variables
            for key, original_value in original_values.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value

    def test_config_file_and_env_priority(self):
        """Test that environment variables override file configuration."""
        config_data = {"app": {"debug": False, "port": 8000}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            # Set environment variable
            os.environ["APP_DEBUG"] = "true"

            config = Config(config_file=config_file)

            # Environment variable should override file
            assert config.get("app.debug") is True
            # File value should be used for port (no env var)
            assert config.get("app.port") == 8000
        finally:
            os.unlink(config_file)
            os.environ.pop("APP_DEBUG", None)

    def test_get_nested_value(self):
        """Test getting nested configuration values."""
        config = Config()

        # Test existing nested values
        assert config.get("app.name") == "MLOps Pipeline"
        assert config.get("model.path") == "models/model.pkl"

        # Test non-existent values
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", "default") == "default"

    def test_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config = Config()
        config.validate()  # Should not raise

        # Test invalid port
        config.config["app"]["port"] = -1
        with pytest.raises(ValueError, match="app.port must be a positive integer"):
            config.validate()

        # Test invalid model type
        config.config["model"]["type"] = "invalid_model"
        with pytest.raises(ValueError, match="model.type must be"):
            config.validate()

        # Test invalid test size
        config.config["data"]["test_size"] = 1.5
        with pytest.raises(ValueError, match="data.test_size must be between 0 and 1"):
            config.validate()

        # Test invalid log level
        config.config["logging"]["level"] = "INVALID"
        with pytest.raises(ValueError, match="logging.level must be one of"):
            config.validate()

    def test_str_to_bool(self):
        """Test string to boolean conversion."""
        config = Config()

        # Test various true values
        assert config._str_to_bool("true") is True
        assert config._str_to_bool("TRUE") is True
        assert config._str_to_bool("1") is True
        assert config._str_to_bool("yes") is True
        assert config._str_to_bool("on") is True

        # Test various false values
        assert config._str_to_bool("false") is False
        assert config._str_to_bool("FALSE") is False
        assert config._str_to_bool("0") is False
        assert config._str_to_bool("no") is False
        assert config._str_to_bool("off") is False
        assert config._str_to_bool("") is False

    def test_parse_list(self):
        """Test comma-separated string parsing."""
        config = Config()

        # Test single item
        assert config._parse_list("item1") == ["item1"]

        # Test multiple items
        assert config._parse_list("item1,item2,item3") == ["item1", "item2", "item3"]

        # Test with spaces
        assert config._parse_list("item1, item2 , item3") == ["item1", "item2", "item3"]

        # Test empty string
        assert config._parse_list("") == [""]

    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["app"]["name"] == "MLOps Pipeline"
        assert config_dict["model"]["type"] == "logistic"

    def test_save_to_file(self):
        """Test saving configuration to file."""
        config = Config()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_file = f.name

        try:
            config.save_to_file(config_file)

            # Verify file was created and contains valid JSON
            with open(config_file, "r") as f:
                saved_config = json.load(f)

            assert saved_config["app"]["name"] == "MLOps Pipeline"
            assert saved_config["model"]["type"] == "logistic"
        finally:
            os.unlink(config_file)

    def test_deep_merge(self):
        """Test deep merging of dictionaries."""
        config = Config()

        base = {
            "level1": {
                "level2": {"value1": "original", "value2": "original"},
                "value3": "original",
            }
        }

        update = {"level1": {"level2": {"value1": "updated"}, "value4": "new"}}

        config._deep_merge(base, update)

        # Check updated values
        assert base["level1"]["level2"]["value1"] == "updated"
        assert base["level1"]["value4"] == "new"

        # Check preserved values
        assert base["level1"]["level2"]["value2"] == "original"
        assert base["level1"]["value3"] == "original"

    def test_set_nested_value(self):
        """Test setting nested values."""
        config = Config()

        # Test setting new nested value
        config._set_nested_value("new.nested.value", "test")
        assert config.get("new.nested.value") == "test"

        # Test updating existing value
        config._set_nested_value("app.port", 9000)
        assert config.get("app.port") == 9000


class TestGlobalConfig:
    """Test global configuration functions."""

    def test_get_config(self):
        """Test getting global configuration instance."""
        config = get_config()
        assert isinstance(config, Config)
        assert config.get("app.name") == "MLOps Pipeline"

    def test_get_setting(self):
        """Test getting individual settings."""
        # Test existing setting
        assert get_setting("app.name") == "MLOps Pipeline"

        # Test non-existent setting with default
        assert get_setting("nonexistent.key", "default") == "default"

        # Test non-existent setting without default
        assert get_setting("nonexistent.key") is None
