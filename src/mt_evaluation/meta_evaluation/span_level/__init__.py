# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Span-level meta-evaluation module.

This module provides utilities for comparing automatic and human error annotations
at the span level, including bipartite matching, standardization, and metrics.

Submodules:
    matching: Bipartite matching algorithms for error span comparison
    standardization: Text and error standardization utilities
    preprocessing: Sample preprocessing for evaluation comparison
    metrics: Span-level evaluation metrics (precision, recall, F1)
    utils: Legacy utilities (re-exports from submodules for backward compatibility)
"""

# Matching functions
from mt_evaluation.meta_evaluation.span_level.matching import (
    compute_overlap_length,
    find_greedy_bipartite_matching,
    find_optimal_bipartite_matching,
    find_greedy_bipartite_matching_simple,
    compute_simple_tp_fp_fn,
    MatchInfo,
)

# Standardization functions
from mt_evaluation.meta_evaluation.span_level.standardization import (
    standardize_text,
    standardize_severity,
    standardize_human_error,
    standardize_automatic_error,
    standardize_human_evaluation,
    standardize_automatic_evaluation,
    find_span_in_paragraph,
)

# Preprocessing functions
from mt_evaluation.meta_evaluation.span_level.preprocessing import (
    remove_overlapping_errors_func,
    preprocess_samples_with_human_evaluations,
    preprocess_samples_with_automatic_evaluations,
    preprocess_single_autoeval,
    preprocess_single_autoeval_wrapper,
    errors_overlap,
    select_more_severe_error,
    count_errors_by_severity,
    compute_evaluations_stats,
)

# Metrics (if available)
try:
    from mt_evaluation.meta_evaluation.span_level.metrics import (
        compute_precision_recall_f1,
    )
except ImportError:
    pass

__all__ = [
    # Matching
    "compute_overlap_length",
    "find_greedy_bipartite_matching",
    "find_greedy_bipartite_matching_simple",
    "compute_simple_tp_fp_fn",
    "MatchInfo",
    # Standardization
    "standardize_text",
    "standardize_severity",
    "standardize_human_error",
    "standardize_automatic_error",
    "standardize_human_evaluation",
    "standardize_automatic_evaluation",
    "find_span_in_paragraph",
    # Preprocessing
    "remove_overlapping_errors_func",
    "preprocess_samples_with_human_evaluations",
    "preprocess_samples_with_automatic_evaluations",
    "preprocess_single_autoeval",
    "preprocess_single_autoeval_wrapper",
    "errors_overlap",
    "select_more_severe_error",
    "count_errors_by_severity",
    "compute_evaluations_stats",
]
