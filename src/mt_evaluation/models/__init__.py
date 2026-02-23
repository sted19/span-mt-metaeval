# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Models module for the MT Evaluation Framework.

This module provides model implementations for running language models,
including local models (HuggingFace) and API-based models (Bedrock).

Main API:
    get_model: Factory function to get model class by ID.
    list_available_models: List all registered model IDs.
    Model: Base class for all model implementations.

Submodules:
    bedrock: AWS Bedrock model implementations (Claude, Llama, Nova, etc.)
    huggingface: Local HuggingFace model implementations

Example:
    >>> from mt_evaluation.models import get_model, list_available_models
    >>> print(list_available_models())
    >>> ModelClass = get_model("us.anthropic.claude-3-5-haiku-20241022-v1:0")
    >>> model = ModelClass("us.anthropic.claude-3-5-haiku-20241022-v1:0")
"""

# Factory functions - main API
from mt_evaluation.models.factory import get_model, list_available_models

# Base model class
from mt_evaluation.models.base import Model

# Re-export from core for backward compatibility
from mt_evaluation.core import FewShots, Response

__all__ = [
    # Factory functions
    "get_model",
    "list_available_models",
    # Base class
    "Model",
    # Data structures (backward compatibility)
    "FewShots",
    "Response",
]
