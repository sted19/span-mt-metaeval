# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Utility functions for the MT Evaluation Framework.

This module provides common utility functions used across the framework,
including PyTorch configuration, logging setup, and naming utilities.
"""

import logging
import os
import sys
from collections import defaultdict

import torch

# Re-export naming utilities
from mt_evaluation.utils.naming import (
    standardize_name,
    get_metric_display_name,
)

# Re-export string utilities
from mt_evaluation.utils.string import find_all_literal


def convert_defaultdict_to_dict(d):
    """Recursively convert defaultdicts to regular dicts for pickling"""
    if isinstance(d, defaultdict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        d = [convert_defaultdict_to_dict(item) for item in d]
    return d


def configure_torch() -> None:
    """
    Configure PyTorch settings for optimal performance.

    Sets up PyTorch with high precision matrix multiplication,
    disables tokenizer parallelism to avoid conflicts, and
    configures environment variables for better debugging.
    """
    torch.set_float32_matmul_precision("high")

    # Disable tokenizers parallelism to avoid deadlocks in multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Enable torch compilation logs for debugging
    os.environ["TORCH_LOGS"] = "recompiles"

    # Set transformers verbosity level
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"


def setup_logging(logging_level: str) -> None:
    """
    Configure logging for the entire application.

    Args:
        logging_level: The logging level to use (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Raises:
        ValueError: If an invalid logging level is provided.
    """
    # Validate logging level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if logging_level.upper() not in valid_levels:
        raise ValueError(
            f"Invalid logging level: {logging_level}. Must be one of {valid_levels}"
        )

    # Convert string to logging level
    numeric_level = getattr(logging, logging_level.upper())

    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set specific loggers to appropriate levels to reduce noise
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def is_all_nones(values):
    """Check if all values in a list are None.
    
    Args:
        values: A list of values to check.
        
    Returns:
        bool: True if all values are None, False otherwise.
    """
    if values is None:
        return True
    return all(val is None for val in values)


__all__ = [
    # Naming utilities
    "standardize_name",
    "get_metric_display_name",
    # String utilities
    "find_all_literal",
    # General utilities
    "convert_defaultdict_to_dict",
    "configure_torch",
    "setup_logging",
    "is_all_nones",
]
