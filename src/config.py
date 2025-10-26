"""
Configuration management for the MLOps pipeline.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for managing environment variables and settings."""

    # Default configuration
    DEFAULT_CONFIG = {
        "app": {
            "name": "MLOps Pipeline",
            "version": "1.0.0",
            "debug": False,
            "host": "0.0.0.0",
            "port": 8000,
        },
        "model": {
            "type": "logistic",
            "path": "models/model.pkl",
            "scaler_path": "models/scaler.pkl",
            "metadata_path": "models/metadata.json",
            "fallback_path": "src/model.pkl",
        },
        "data": {"path": None, "test_size": 0.2, "random_state": 42, "n_samples": 1000},
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "api": {
            "cors_origins": ["*"],
            "max_request_size": 10485760,  # 10MB
            "timeout": 30,
        },
    }

    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_file: Path to configuration file (optional)
        """
        self.config = self.DEFAULT_CONFIG.copy()

        # Load from file if provided
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)

        # Override with environment variables
        self.load_from_env()

        # Validate configuration
        self.validate()

    def load_from_file(self, config_file: str):
        """Load configuration from JSON file.

        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, "r") as f:
                file_config = json.load(f)

            # Deep merge with default config
            self._deep_merge(self.config, file_config)
            logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")

    def load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # App settings
            "APP_DEBUG": ("app.debug", self._str_to_bool),
            "APP_HOST": ("app.host", str),
            "APP_PORT": ("app.port", int),
            # Model settings
            "MODEL_TYPE": ("model.type", str),
            "MODEL_PATH": ("model.path", str),
            "SCALER_PATH": ("model.scaler_path", str),
            "METADATA_PATH": ("model.metadata_path", str),
            # Data settings
            "DATA_PATH": ("data.path", str),
            "DATA_TEST_SIZE": ("data.test_size", float),
            "DATA_RANDOM_STATE": ("data.random_state", int),
            "DATA_N_SAMPLES": ("data.n_samples", int),
            # Logging settings
            "LOG_LEVEL": ("logging.level", str),
            # API settings
            "CORS_ORIGINS": ("api.cors_origins", self._parse_list),
            "MAX_REQUEST_SIZE": ("api.max_request_size", int),
            "API_TIMEOUT": ("api.timeout", int),
        }

        for env_var, (config_path, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    self._set_nested_value(config_path, converted_value)
                    logger.debug(
                        f"Set {config_path} = {converted_value} from {env_var}"
                    )
                except Exception as e:
                    logger.warning(f"Error setting {config_path} from {env_var}: {e}")

    def _deep_merge(self, base_dict: Dict, update_dict: Dict):
        """Deep merge two dictionaries."""
        for key, value in update_dict.items():
            if (
                key in base_dict
                and isinstance(base_dict[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value

    def _set_nested_value(self, path: str, value: Any):
        """Set a nested value in the config dictionary.

        Args:
            path: Dot-separated path (e.g., 'app.debug')
            value: Value to set
        """
        keys = path.split(".")
        current = self.config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def get(self, path: str, default: Any = None) -> Any:
        """Get a configuration value by path.

        Args:
            path: Dot-separated path (e.g., 'app.debug')
            default: Default value if path not found

        Returns:
            Configuration value
        """
        keys = path.split(".")
        current = self.config

        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

    def validate(self):
        """Validate configuration values."""
        # Validate app settings
        if not isinstance(self.get("app.port"), int) or self.get("app.port") < 1:
            raise ValueError("app.port must be a positive integer")

        # Validate model settings
        if self.get("model.type") not in ["logistic", "random_forest"]:
            raise ValueError("model.type must be 'logistic' or 'random_forest'")

        # Validate data settings
        test_size = self.get("data.test_size")
        if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
            raise ValueError("data.test_size must be between 0 and 1")

        # Validate logging level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.get("logging.level").upper() not in valid_levels:
            raise ValueError(f"logging.level must be one of {valid_levels}")

        logger.info("Configuration validation passed")

    def _str_to_bool(self, value: str) -> bool:
        """Convert string to boolean."""
        return value.lower() in ("true", "1", "yes", "on")

    def _parse_list(self, value: str) -> list:
        """Parse comma-separated string to list."""
        return [item.strip() for item in value.split(",")]

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()

    def save_to_file(self, file_path: str):
        """Save configuration to JSON file.

        Args:
            file_path: Path to save configuration
        """
        try:
            with open(file_path, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving config to {file_path}: {e}")


# Global configuration instance
config = Config()


# Convenience functions
def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def get_setting(path: str, default: Any = None) -> Any:
    """Get a configuration setting by path.

    Args:
        path: Dot-separated path (e.g., 'app.debug')
        default: Default value if path not found

    Returns:
        Configuration value
    """
    return config.get(path, default)
