# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
MT Evaluation Framework
=======================

A comprehensive framework for automatic evaluation of machine translation quality.

This package provides tools for:
- Running automatic evaluation of translations using LLM-based metrics
- Computing meta-evaluation scores comparing automatic metrics to human judgments
- Span-level analysis of translation errors
- Caching and data management utilities

Package Structure
-----------------
- **core**: Core data structures (Sample, Error, Evaluation, etc.)
- **autoevals**: Automatic evaluation implementations (GembaMQM, UnifiedMQM, etc.)
- **models**: Language model wrappers (Bedrock, HuggingFace)
- **data**: Data loading, caching, and language utilities
- **meta_evaluation**: Tools for evaluating metric quality
- **config**: YAML-based configuration management

Quick Start
-----------
>>> from mt_evaluation.core import Sample, Error
>>> from mt_evaluation.autoevals.factory import get_autoeval
>>> from mt_evaluation.models.factory import get_model

Example Usage
-------------
1. Create a sample for evaluation:
   >>> sample = Sample(
   ...     src="Hello, world!",
   ...     tgt="Hallo, Welt!",
   ...     src_lang="English",
   ...     tgt_lang="German"
   ... )

2. Load an evaluator:
   >>> model = get_model("us.anthropic.claude-3-5-haiku-20241022-v1:0")(model_id)
   >>> evaluator = get_autoeval("unified-mqm-boosted-v5")(model, "unified-mqm-boosted-v5")

3. Run evaluation:
   >>> evaluations = evaluator.evaluate([sample])

For more detailed usage, see the documentation in each submodule.
"""

__version__ = "0.1.0"

# Re-export commonly used items for convenience
from mt_evaluation.core import (
    Sample,
    Error,
    Evaluation,
    HumanEvaluation,
    AutomaticEvaluation,
    Prompt,
    FewShots,
    Response,
)

__all__ = [
    # Version info
    "__version__",
    # Core data structures
    "Sample",
    "Error",
    "Evaluation",
    "HumanEvaluation",
    "AutomaticEvaluation",
    "Prompt",
    "FewShots",
    "Response",
]
