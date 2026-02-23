# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Configuration module for the MT Evaluation Framework.

This module provides utilities for loading configuration from YAML files
and accessing metric configurations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Get the directory containing this module
CONFIG_DIR = Path(__file__).parent


def load_yaml_config(config_name: str = "metrics.yaml") -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_name: Name of the configuration file (default: metrics.yaml).

    Returns:
        Dict[str, Any]: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
    """
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_metrics_config() -> Dict[str, Any]:
    """
    Load the metrics configuration.

    Returns:
        Dict[str, Any]: The metrics configuration dictionary.
    """
    return load_yaml_config("metrics.yaml")


def get_autoevals_with_no_extended_spans() -> List[str]:
    """
    Get the list of autoevals that do not produce extended spans.

    Returns:
        List[str]: List of autoeval names without extended spans.
    """
    config = get_metrics_config()
    return config.get("autoevals_with_no_extended_spans", [])


def get_metrics_to_evaluate(test_set: str) -> List[Dict[str, Any]]:
    """
    Get the metrics to evaluate for a specific test set.

    Args:
        test_set: The test set name (e.g., "wmt22", "wmt23", "wmt24", "wmt25").

    Returns:
        List[Dict[str, Any]]: List of metric configurations.
    """
    config = get_metrics_config()
    metrics = config.get(test_set, [])
    
    # Convert outputs_path to Path objects for compatibility
    for metric in metrics:
        if "outputs_path" in metric:
            metric["outputs_path"] = Path(metric["outputs_path"])
    
    return metrics


# =============================================================================
# Model Aliases Configuration
# =============================================================================

# Cache for model aliases to avoid repeated file reads
_model_aliases_cache: Optional[Dict[str, str]] = None


def get_model_aliases() -> Dict[str, str]:
    """
    Load the model aliases configuration.

    Returns a mapping from full model IDs to short display names.

    Returns:
        Dict[str, str]: Mapping of model_id -> short_name.
    """
    global _model_aliases_cache
    
    if _model_aliases_cache is None:
        config = load_yaml_config("model_aliases.yaml")
        _model_aliases_cache = config.get("model_aliases", {})
    
    return _model_aliases_cache


def get_model_short_name(model_id: str) -> str:
    """
    Get the short display name for a model.

    Args:
        model_id: The full model identifier.

    Returns:
        str: The short display name, or the original model_id if not found.
    """
    aliases = get_model_aliases()
    return aliases.get(model_id, model_id)


__all__ = [
    "load_yaml_config",
    "get_metrics_config",
    "get_autoevals_with_no_extended_spans",
    "get_metrics_to_evaluate",
    "get_model_aliases",
    "get_model_short_name",
]
