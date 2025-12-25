"""
Configuration handling and validation for Yogimass.
"""

from __future__ import annotations
import yaml
from pathlib import Path

class ConfigError(Exception):
    """
    Exception raised for configuration errors.

    Attributes:
        path (str): The dotted path to the configuration key where the error occurred.
        message (str): The error message.
    """
    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"ConfigError at '{path}': {message}")


def load_config(path: str | Path) -> dict:
    """
    Loads a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        dict: The loaded configuration.

    Raises:
        ConfigError: If the file cannot be read or parsed.
    """
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError("root", f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigError("root", f"Error parsing YAML: {e}")
