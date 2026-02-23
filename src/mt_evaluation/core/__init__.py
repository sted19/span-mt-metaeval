# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Core data structures for the MT Evaluation Framework.

This module provides the fundamental data structures used throughout the framework,
including prompts, errors, evaluations, samples, and model communication classes.

Classes:
    Prompt: Container for system and user prompt templates.
    Error: Represents a translation error identified during evaluation.
    Evaluation: Base class for evaluation results.
    HumanEvaluation: Human evaluation result with rater information.
    AutomaticEvaluation: Automatic evaluation result with model metadata.
    Sample: Translation sample for evaluation.
    FewShots: Container for few-shot examples in in-context learning.
    Response: Container for model responses.

Constants:
    Error type constants (non_translation, omission, etc.)
    Severity constants
    WMT language pair constants
"""

from mt_evaluation.core.datastructures import (
    Prompt,
    Error,
    Evaluation,
    HumanEvaluation,
    AutomaticEvaluation,
    Sample,
)
from mt_evaluation.core.model_io import FewShots, Response
from mt_evaluation.core.scoring import (
    severity_to_score,
    assign_score_based_on_severity,
    SEVERITY_SCORES,
    SPECIAL_CATEGORY_SCORES,
)
from mt_evaluation.core.types import (
    StandardizationResult,
    PreprocessingStats,
)
from mt_evaluation.core.constants import (
    # Error type constants
    non_translation,
    unintelligible,
    omission,
    source_issue,
    source_issue2,
    source_error,
    source_error2,
    creative_reinterpretation,
    no_error,
    no_error2,
    # Severity constants
    all_severities,
    UNKNOWN_SEVERITY,
    # WMT language pair constants
    wmt22_lps,
    wmt23_lps,
    wmt24_lps,
    wmt25_lps_mqm,
    wmt25_lps_esa,
    wmt25_lps,
)

__all__ = [
    # Data structures
    "Prompt",
    "Error",
    "Evaluation",
    "HumanEvaluation",
    "AutomaticEvaluation",
    "Sample",
    "FewShots",
    "Response",
    # Error type constants
    "non_translation",
    "unintelligible",
    "omission",
    "source_issue",
    "source_issue2",
    "source_error",
    "source_error2",
    "creative_reinterpretation",
    "no_error",
    "no_error2",
    # Severity constants
    "all_severities",
    "UNKNOWN_SEVERITY",
    # WMT language pair constants
    "wmt22_lps",
    "wmt23_lps",
    "wmt24_lps",
    "wmt25_lps_mqm",
    "wmt25_lps_esa",
    "wmt25_lps",
    # Scoring functions
    "severity_to_score",
    "assign_score_based_on_severity",
    "SEVERITY_SCORES",
    "SPECIAL_CATEGORY_SCORES",
    # Dataclasses for structured return values
    "StandardizationResult",
    "PreprocessingStats",
]
