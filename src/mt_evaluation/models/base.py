# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Base model interface for the MT Evaluation Framework.

This module defines the abstract base class for all model implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Union, Any, Optional, TYPE_CHECKING

from mt_evaluation.core import Response

# Type checking import to avoid runtime dependency on transformers
if TYPE_CHECKING:
    from transformers import PreTrainedModel


class Model(ABC):
    """
    Abstract base class for all model implementations.

    This class defines the interface that all model implementations must follow.
    Models can be backed by various providers (Bedrock, HuggingFace, etc.).

    Attributes:
        name: Identifier for the model.
        model: The underlying model object (optional, used by HuggingFace models).
    """

    def __init__(self, model_id: str):
        """
        Initialize the model.

        Args:
            model_id: Unique identifier for the model.
        """
        self.name = model_id
        # Optional underlying model object - used by HuggingFace implementations
        self.model: Optional[Any] = None

    @abstractmethod
    def __call__(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        few_shots: List[Dict[str, List[str]]],
        **kwargs,
    ) -> List[Response]:
        """
        Generate responses for given prompts.

        Args:
            system_prompts: List of system prompts for each request.
            user_prompts: List of user prompts for each request.
            few_shots: List of few-shot examples for each request.
                Each element is a dict with 'user_prompts' and 'assistant_responses' keys.
            **kwargs: Additional model-specific parameters.

        Returns:
            List of Response objects containing generated text and metadata.
        """
        pass
