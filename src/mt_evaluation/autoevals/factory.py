# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Factory for automatic evaluators.

This module provides a registry system for automatic evaluators and utility
functions to access and list available evaluators.
"""

import logging
from typing import Dict, Type, List

from mt_evaluation.autoevals.gemba_mqm import GembaMQM
from mt_evaluation.autoevals.autoeval import AutoEval
from mt_evaluation.autoevals.unified.simple import Simple
from mt_evaluation.autoevals.unified.simplest import Simplest
from mt_evaluation.autoevals.unified.unified_mqm_critical import UnifiedMQMCritical
from mt_evaluation.autoevals.unified.unified_mqm_boosted_v5 import UnifiedMQMBoostedV5
from mt_evaluation.autoevals.unified.unified_mqm_boosted_doc_context import (
    UnifiedMQMBoostedDocContext,
)

logger = logging.getLogger(__name__)

# Registry of available automatic evaluators
AUTOEVAL_REGISTRY: Dict[str, Type[AutoEval]] = {
    "gemba-mqm": GembaMQM,
    "unified-mqm-critical": UnifiedMQMCritical,
    "unified-mqm-boosted-v5": UnifiedMQMBoostedV5,
    "unified-mqm-boosted-doc-context": UnifiedMQMBoostedDocContext,
    "unified-simple": Simple,
    "unified-simplest": Simplest,
}


def get_autoeval(evaluation_schema: str) -> Type[AutoEval]:
    """
    Get an AutoEval class by name.

    Args:
        evaluation_schema: The name of the evaluation schema to retrieve.

    Returns:
        Type[AutoEval]: The AutoEval class corresponding to the schema.

    Raises:
        ValueError: If the evaluation schema is not found in the registry.

    Example:
        >>> evaluator_class = get_autoeval("gemba-mqm")
        >>> evaluator = evaluator_class(model, "gemba-mqm")
    """
    if evaluation_schema not in AUTOEVAL_REGISTRY:
        available = ", ".join(AUTOEVAL_REGISTRY.keys())
        raise ValueError(
            f"Unknown evaluation schema '{evaluation_schema}'. Available: {available}"
        )

    return AUTOEVAL_REGISTRY[evaluation_schema]


def list_available_evaluators() -> List[str]:
    """
    Get list of available evaluator names.

    Returns:
        List[str]: List of available evaluator schema names.

    Example:
        >>> evaluators = list_available_evaluators()
        >>> print(evaluators)
        ['gemba-mqm', 'unified-mqm', ...]
    """
    return list(AUTOEVAL_REGISTRY.keys())
