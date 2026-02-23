# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Model input/output data structures for the MT Evaluation Framework.

This module defines data structures for communication with language models,
including few-shot examples and model responses.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class FewShots:
    """
    Container for few-shot examples used in in-context learning.

    This class stores paired user prompts and assistant responses that can be
    used as examples when prompting language models.

    Attributes:
        user_prompts: List of user prompt strings for few-shot examples.
        assistant_responses: List of corresponding assistant response strings.
    """

    user_prompts: List[str]
    assistant_responses: List[str]

    def __str__(self) -> str:
        """Return string representation of the few-shots."""
        return f"FewShots(user_prompts={self.user_prompts}, assistant_responses={self.assistant_responses})"

    def __repr__(self) -> str:
        """Return string representation of the few-shots."""
        return self.__str__()

    def to_dict(self) -> Dict[str, List[str]]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dict[str, List[str]]: Dictionary representation of few-shots.
        """
        return {
            "user_prompts": self.user_prompts,
            "assistant_responses": self.assistant_responses,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> "FewShots":
        """
        Create FewShots instance from dictionary.

        Args:
            data: Dictionary containing user_prompts and assistant_responses.

        Returns:
            FewShots: New FewShots instance.
        """
        return FewShots(**data)


@dataclass
class Response:
    """
    Container for model response.

    This class encapsulates the response from a language model, including
    the response text and optional metadata like cost.

    Attributes:
        response: The response text from the model.
        cost: Optional cost of the API call.
    """

    response: str
    cost: float = 0.0
