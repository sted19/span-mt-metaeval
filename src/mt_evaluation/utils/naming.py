# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Naming utilities for the MT Evaluation Framework.

This module provides functions for standardizing and formatting names
for models, autoevals, and metrics throughout the framework.
"""

from mt_evaluation.config import get_model_short_name


def standardize_name(name: str) -> str:
    """
    Standardize a name by replacing problematic characters.

    Replaces forward slashes with underscores to avoid path issues
    when using names in file paths or identifiers.

    Args:
        name: The name to standardize (e.g., autoeval name or model name).

    Returns:
        str: The standardized name with slashes replaced by underscores.

    Example:
        >>> standardize_name("google/gemma-3-12b-it")
        "google_gemma-3-12b-it"
    """
    return name.replace("/", "_")


def get_metric_display_name(
    autoeval_name: str,
    model_name: str,
    run_specific_info: str,
) -> str:
    """
    Generate a human-readable display name for a metric.

    Combines the autoeval name, short model name, and run-specific info
    into a consistent display format.

    Args:
        autoeval_name: Name of the automatic evaluator.
        model_name: Full model identifier.
        run_specific_info: Additional info distinguishing this run.

    Returns:
        str: Formatted metric name for display.

    Example:
        >>> get_metric_display_name(
        ...     "unified-mqm-boosted-v5",
        ...     "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        ...     "wmt24"
        ... )
        "unified-mqm-boosted-v5_claude-3-5-haiku_wmt24"
    """
    autoeval_name = standardize_name(autoeval_name)
    short_model_name = standardize_name(get_model_short_name(model_name))

    return f"{autoeval_name}_{short_model_name}_{run_specific_info}"


__all__ = [
    "standardize_name",
    "get_metric_display_name",
]
