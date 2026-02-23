# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Automatic evaluation module for machine translation quality assessment.

This module provides automatic evaluators that assess translation quality using
language models.

Main API:
    get_autoeval: Factory function to get evaluator class by name.
    list_available_evaluators: List all registered evaluator names.
    AutoEval: Base class for all automatic evaluators.

Note:
    Core data structures (Prompt, Error, Sample, etc.) should be imported
    from mt_evaluation.core, not from this module.

Example:
    >>> from mt_evaluation.autoevals import get_autoeval, list_available_evaluators
    >>> from mt_evaluation.core import Sample, Prompt
    >>> print(list_available_evaluators())
    >>> EvaluatorClass = get_autoeval("gemba-mqm")
"""

# Base evaluator class
from mt_evaluation.autoevals.autoeval import AutoEval

# Parsing utilities
from mt_evaluation.autoevals.utils import extract_json_response

# Factory functions
from mt_evaluation.autoevals.factory import get_autoeval, list_available_evaluators

__all__ = [
    # Factory functions
    "get_autoeval",
    "list_available_evaluators",
    # Base class
    "AutoEval",
    # Utilities
    "extract_json_response",
]
