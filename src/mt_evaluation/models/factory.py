# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Model factory for the MT Evaluation Framework.

This module provides factory functions for instantiating model classes
and lists available models.
"""

from typing import Dict, Type

from mt_evaluation.models.base import Model
from mt_evaluation.models.bedrock.gpt_oss import GPTOss20B, GPTOss120B
from mt_evaluation.models.huggingface.gemma3 import Gemma3
from mt_evaluation.models.huggingface.llama3 import Llama3
from mt_evaluation.models.huggingface.qwen3 import Qwen3
from mt_evaluation.models.bedrock.llama import (
    Llama318B,
    LLama4Scout,
    LLama4Maverick,
    Llama3370B,
    Llama3211B,
)
from mt_evaluation.models.bedrock.nova import NovaPro
from mt_evaluation.models.bedrock.claude import (
    Claude35Haiku,
    Claude37Sonnet,
    Claude45Sonnet,
    Claude45Haiku,
)
from mt_evaluation.models.bedrock.qwen import Qwen3235B, Qwen332B

MODEL_REGISTRY: Dict[str, Type[Model]] = {
    "google/gemma-3-1b-it": Gemma3,
    "google/gemma-3-4b-it": Gemma3,
    "google/gemma-3-12b-it": Gemma3,
    "google/gemma-3-27b-it": Gemma3,
    "meta-llama/Meta-Llama-3-8B-Instruct": Llama3,
    "Qwen/Qwen3-8B": Qwen3,
    "amazon.nova-pro-v1:0": NovaPro,
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": Claude35Haiku,
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": Claude37Sonnet,
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": Claude45Haiku,
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0": Claude45Sonnet,
    "us.meta.llama3-1-8b-instruct-v1:0": Llama318B,
    "us.meta.llama3-2-11b-instruct-v1:0": Llama3211B,
    "us.meta.llama3-3-70b-instruct-v1:0": Llama3370B,
    "us.meta.llama4-scout-17b-instruct-v1:0": LLama4Scout,
    "us.meta.llama4-maverick-17b-instruct-v1:0": LLama4Maverick,
    "openai.gpt-oss-20b-1:0": GPTOss20B,
    "openai.gpt-oss-120b-1:0": GPTOss120B,
    "qwen.qwen3-235b-a22b-2507-v1:0": Qwen3235B,
    "qwen.qwen3-32b-v1:0": Qwen332B,
}


def get_model(model_id: str) -> Type[Model]:
    """Get a Model class by name."""
    if model_id not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model_id '{model_id}'. Available: {available}")

    return MODEL_REGISTRY[model_id]


def list_available_models() -> list[str]:
    """Get list of available model names."""
    return list(MODEL_REGISTRY.keys())
