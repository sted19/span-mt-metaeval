# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Data handling module for the MT Evaluation Framework.

This module provides utilities for data handling, caching, and language code mappings
used throughout the evaluation framework.

Submodules:
    cache: Caching system for evaluation results (MTEvaluationCache).
    language_codes: Language code to language name mappings.
    utils: General data utilities (document context, flattening, cache management).
    wmt_loaders: WMT-specific data loaders and parsers.

Note:
    For backward compatibility, commonly used classes and functions are re-exported
    from this module.
"""

# Re-export commonly used classes and functions
from mt_evaluation.data.cache import MTEvaluationCache
from mt_evaluation.data.language_codes import (
    LANG_CODE_TO_NAME,
    lang_code2lang,
    get_language_name,
    get_language_name_safe,
)
from mt_evaluation.data.utils import (
    get_cache_dir,
    flatten_samples_for_evaluation,
    parse_lps,
    get_autoeval2lp2sys2samples_with_automatic_evaluations,
)
from mt_evaluation.data.wmt_loaders import (
    get_raters_evaluations,
    get_super_raters_from_raters,
    parse_tsv_wmt25_submission,
    enes_subcategory_to_category_mapping,
)
# Import from canonical location
from mt_evaluation.core.scoring import assign_score_based_on_severity

__all__ = [
    # Cache
    "MTEvaluationCache",
    # Language codes
    "LANG_CODE_TO_NAME",
    "lang_code2lang",
    "get_language_name",
    "get_language_name_safe",
    # General utilities
    "get_cache_dir",
    "flatten_samples_for_evaluation",
    "parse_lps",
    "get_autoeval2lp2sys2samples_with_automatic_evaluations",
    # WMT loaders
    "get_raters_evaluations",
    "get_super_raters_from_raters",
    "parse_tsv_wmt25_submission",
    "enes_subcategory_to_category_mapping",
    # Scoring
    "assign_score_based_on_severity",
]
