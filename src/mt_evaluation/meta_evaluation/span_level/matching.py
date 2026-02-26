# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Bipartite matching utilities for span-level error comparison.

This module provides functions for matching automatic errors to human errors
using greedy bipartite matching based on overlap.
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

from mt_evaluation.core import Error
from mt_evaluation.meta_evaluation.span_level.metrics import (
    compute_overlap_length,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchInfo:
    """Information about a matched error pair."""

    avg_overlap: float
    overlap_length: int
    matched_index: int


def find_greedy_bipartite_matching(
    auto_errors: List[Error],
    human_errors: List[Error],
    src: str = "",
    tgt: str = "",
    severity_penalty: float = 0.0,
) -> Tuple[Dict[int, MatchInfo], Dict[int, MatchInfo]]:
    """
    Find a greedy bipartite matching between automatic and human errors.

    Uses greedy matching based on average overlap ratio, with optional severity penalty
    for mismatched severities. Returns detailed match information for computing metrics.

    Args:
        auto_errors: List of automatic evaluation errors
        human_errors: List of human evaluation errors
        src: Source text (optional, for validation)
        tgt: Target text (optional, for validation)
        severity_penalty: Penalty factor (0-1) for severity mismatch.
                         0 = no penalty, 1 = treat mismatch as no overlap

    Returns:
        Tuple of (auto_matches, human_matches) where:
        - auto_matches: Dict mapping auto error index -> MatchInfo
        - human_matches: Dict mapping human error index -> MatchInfo
    """
    if not auto_errors or not human_errors:
        return {}, {}

    p = float(severity_penalty)

    # Compute all pairwise scores
    candidates = []
    for i, auto_error in enumerate(auto_errors):
        for j, human_error in enumerate(human_errors):
            # Skip if errors are on different sides (source vs target)
            if auto_error.is_source_error != human_error.is_source_error:
                continue

            overlap_len = compute_overlap_length(
                (auto_error.start, auto_error.end),
                (human_error.start, human_error.end),
            )

            if overlap_len == 0:
                continue

            # Compute average overlap ratio
            avg_overlap = (
                2 * overlap_len / (len(auto_error.span) + len(human_error.span))
            )

            # Apply severity penalty
            severity_matches = auto_error.severity == human_error.severity
            severity_reward = 1.0 if severity_matches else (1.0 - p)
            effective_score = avg_overlap * severity_reward

            candidates.append((effective_score, avg_overlap, overlap_len, i, j))

    # Sort by effective score (descending)
    candidates.sort(reverse=True, key=lambda x: x[0])

    # Greedy matching
    used_auto = set()
    used_human = set()
    auto_matches: Dict[int, MatchInfo] = {}
    human_matches: Dict[int, MatchInfo] = {}

    for effective_score, avg_overlap, overlap_len, i, j in candidates:
        if i not in used_auto and j not in used_human:
            auto_matches[i] = MatchInfo(avg_overlap, overlap_len, j)
            human_matches[j] = MatchInfo(avg_overlap, overlap_len, i)
            used_auto.add(i)
            used_human.add(j)

    return auto_matches, human_matches


def find_optimal_bipartite_matching(
    auto_errors: List[Error],
    human_errors: List[Error],
    src: str,
    tgt: str,
    objective: Callable[[Tuple[int, int], Tuple[int, int], str, str], float],
    severity_penalty: float = 0.0,
) -> Tuple[Dict[int, MatchInfo], Dict[int, MatchInfo]]:
    """
    Find the optimal one-to-one matching between automatic and human errors.

    Uses the Hungarian algorithm (via scipy.optimize.linear_sum_assignment) to
    find the assignment that maximizes the sum of the user-supplied objective
    across all matched pairs.

    Args:
        auto_errors: List of automatic evaluation errors.
        human_errors: List of human evaluation errors.
        src: Source text.
        tgt: Target text.
        objective: A function (auto_error, human_error, src, tgt) -> float that
                   scores how well a pair of errors matches. Higher is better.
                   This is evaluated for every (auto, human) pair; the algorithm
                   finds the assignment that maximizes the total score.
        severity_penalty: Penalty factor (0-1) for severity mismatch.
                         0 = no penalty, 1 = treat mismatch as no match.
                         Applied multiplicatively: score *= (1 - penalty) when
                         severities differ.

    Returns:
        Tuple of (auto_matches, human_matches) where:
        - auto_matches: Dict mapping auto error index -> MatchInfo
        - human_matches: Dict mapping human error index -> MatchInfo
        Same format as find_greedy_bipartite_matching for interchangeability.
    """
    if not auto_errors or not human_errors:
        return {}, {}

    p = float(severity_penalty)
    n_auto = len(auto_errors)
    n_human = len(human_errors)

    # Build the score matrix: score_matrix[i, j] is the objective value for
    # matching auto_errors[i] with human_errors[j], after severity penalty.
    score_matrix = np.zeros((n_auto, n_human), dtype=np.float64)

    for i, auto_error in enumerate(auto_errors):
        for j, human_error in enumerate(human_errors):
            # Skip if errors are on different sides (source vs target)
            if auto_error.is_source_error != human_error.is_source_error:
                continue

            raw_score = objective(
                (auto_error.start, auto_error.end),
                (human_error.start, human_error.end),
                src,
                tgt,
            )

            if raw_score <= 0.0:
                continue

            # Apply severity penalty
            severity_matches = auto_error.severity == human_error.severity
            severity_reward = 1.0 if severity_matches else (1.0 - p)
            score_matrix[i, j] = raw_score * severity_reward

    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    # Build result dicts, filtering out pairs with non-positive objective
    auto_matches: Dict[int, MatchInfo] = {}
    human_matches: Dict[int, MatchInfo] = {}

    for i, j in zip(row_ind, col_ind):
        if score_matrix[i, j] <= 0.0:
            continue

        overlap_len = compute_overlap_length(
            (auto_errors[i].start, auto_errors[i].end),
            (human_errors[j].start, human_errors[j].end),
        )

        auto_span_len = len(auto_errors[i].span)
        human_span_len = len(human_errors[j].span)
        total_span_len = auto_span_len + human_span_len
        avg_overlap = (2 * overlap_len / total_span_len) if total_span_len > 0 else 0.0

        auto_matches[i] = MatchInfo(avg_overlap, overlap_len, j)
        human_matches[j] = MatchInfo(avg_overlap, overlap_len, i)

    return auto_matches, human_matches


def find_greedy_bipartite_matching_simple(
    auto_errors: List[Error],
    human_errors: List[Error],
    severity_penalty: float = 0.0,
) -> Tuple[
    List[Tuple[Error, Error, float]],  # matches: (auto, human, adjusted_overlap)
    List[Error],  # unmatched auto errors
    List[Error],  # unmatched human errors
]:
    """
    Simplified greedy bipartite matching returning error tuples.

    This is a convenience wrapper around find_greedy_bipartite_matching()
    for cases where you just need the matched/unmatched errors.

    Args:
        auto_errors: List of automatic evaluation errors
        human_errors: List of human evaluation errors
        severity_penalty: Penalty factor (0-1) for severity mismatch.

    Returns:
        Tuple of (matches, unmatched_auto, unmatched_human)
    """
    auto_matches, _ = find_greedy_bipartite_matching(
        auto_errors, human_errors, severity_penalty=severity_penalty
    )

    # Build result lists
    matches = []
    matched_auto_indices = set()
    matched_human_indices = set()

    for auto_idx, match_info in auto_matches.items():
        matches.append(
            (
                auto_errors[auto_idx],
                human_errors[match_info.matched_index],
                match_info.avg_overlap,
            )
        )
        matched_auto_indices.add(auto_idx)
        matched_human_indices.add(match_info.matched_index)

    unmatched_auto = [
        e for i, e in enumerate(auto_errors) if i not in matched_auto_indices
    ]
    unmatched_human = [
        e for i, e in enumerate(human_errors) if i not in matched_human_indices
    ]

    return matches, unmatched_auto, unmatched_human


def compute_simple_tp_fp_fn(
    auto_errors: List[Error],
    human_errors: List[Error],
    is_source_side: Optional[bool] = None,
    severity_penalty: float = 0.0,
) -> Tuple[int, int, int]:
    """
    Compute simple TP, FP, FN counts using greedy bipartite matching.

    Args:
        auto_errors: Automatic evaluation errors
        human_errors: Human evaluation errors
        is_source_side: If True, only consider source-side errors.
                        If False, only target-side. If None, consider all.
        severity_penalty: Penalty for severity mismatch

    Returns:
        Tuple of (tp, fp, fn)
    """
    # Filter by source/target side if specified
    if is_source_side is True:
        filtered_auto = [e for e in auto_errors if e.is_source_error]
        filtered_human = [e for e in human_errors if e.is_source_error]
    elif is_source_side is False:
        filtered_auto = [e for e in auto_errors if not e.is_source_error]
        filtered_human = [e for e in human_errors if not e.is_source_error]
    else:
        filtered_auto = auto_errors
        filtered_human = human_errors

    matches, unmatched_auto, unmatched_human = find_greedy_bipartite_matching_simple(
        filtered_auto, filtered_human, severity_penalty
    )

    tp = len(matches)
    fp = len(unmatched_auto)
    fn = len(unmatched_human)

    return tp, fp, fn
