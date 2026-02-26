# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Metrics classes for span-level meta-evaluation.

This module contains classes for computing and aggregating precision, recall, and F1
metrics at both micro and macro levels.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union


def span_tp_char_counts(
    span1: Tuple[int, int],
    span2: Tuple[int, int],
    src: str,
    tgt: str,
) -> float:
    """
    Compute the true positives for two spans based on character counts. For the other measures, I can compute directly the span f-score. However, character counts (wmt25) cannot be computed via averaging span-level scores, and instead requires computing tp, fp, and fn across spans and then computing final f-score. Therefore, the f-score is maximized where tp are maximized (given that both p and r are monotonically increasing in tp), so I can use tp as the objective for optimal bipartite matching.

    This can be used as an objective function for optimal bipartite matching.

    Args:
        span1: Tuple of (start, end) for first span
        span2: Tuple of (start, end) for second span
    Returns:
        A score in [0, min(len(span1), len(span2))] where:
        - min(len(span1), len(span2)) means that the spans overlap by the full length of the shorter span (perfect match for character-level metric)
        - 0.0 means the spans do not overlap
        - Values in between reflect the number of characters in the overlap
    """
    start1, end1 = span1
    start2, end2 = span2

    assert start1 < end1, f"Invalid span1: {span1}"
    assert start2 < end2, f"Invalid span2: {span2}"

    overlap_len = compute_overlap_length(span1, span2)
    return overlap_len


def span_f_score_exact_match(
    span1: Tuple[int, int],
    span2: Tuple[int, int],
    src: str,
    tgt: str,
) -> float:
    """
    Compute an F-score-like metric for two spans based on exact match.

    This can be used as an objective function for optimal bipartite matching.

    Args:
        span1: Tuple of (start, end) for first span
        span2: Tuple of (start, end) for second span
    Returns:
        A score in {0,1} where:
        - 1.0 means that the spans are identical (exact match)
        - 0.0 means the spans are different in any way (no exact match)
    """
    start1, end1 = span1
    start2, end2 = span2

    assert start1 < end1, f"Invalid span1: {span1}"
    assert start2 < end2, f"Invalid span2: {span2}"

    if start1 == start2 and end1 == end2:
        return 1.0
    else:
        return 0.0


def span_f_score_partial(
    span1: Tuple[int, int],
    span2: Tuple[int, int],
    src: str,
    tgt: str,
) -> float:
    """
    Compute an F-score-like metric for two spans that accounts for partial overlap, but gives either full credit or no credit for each span.

    This can be used as an objective function for optimal bipartite matching.

    Args:
        span1: Tuple of (start, end) for first span
        span2: Tuple of (start, end) for second span
    Returns:
        A score in {0,1} where:
        - 1.0 means that the spans overlap
        - 0.0 means no overlap
    """
    start1, end1 = span1
    start2, end2 = span2

    assert start1 < end1, f"Invalid span1: {span1}"
    assert start2 < end2, f"Invalid span2: {span2}"

    overlap_len = compute_overlap_length(span1, span2)
    if overlap_len > 0:
        return 1.0
    else:
        return 0.0


def span_f_score_proportion_chars(
    span1: Tuple[int, int],
    span2: Tuple[int, int],
    src: str,
    tgt: str,
) -> float:
    """
    Compute an F-score-like metric for two spans that accounts for partial overlap.

    This can be used as an objective function for optimal bipartite matching.

    Args:
        span1: Tuple of (start, end) for first span
        span2: Tuple of (start, end) for second span
    Returns:
        A score between 0 and 1 representing the degree of match, where:
        - 1.0 means perfect match (identical spans)
        - 0.0 means no overlap
        - Values in between reflect partial overlap.
    """
    start1, end1 = span1
    start2, end2 = span2

    assert start1 < end1, f"Invalid span1: {span1}"
    assert start2 < end2, f"Invalid span2: {span2}"

    overlap_len = compute_overlap_length(span1, span2)
    if overlap_len == 0:
        return 0.0

    precision = overlap_len / (end1 - start1)
    recall = overlap_len / (end2 - start2)

    if precision + recall == 0:
        return 0.0

    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score


def compute_overlap_length(span1: Tuple[int, int], span2: Tuple[int, int]) -> int:
    """
    Compute the length of overlap between two spans.

    Args:
        span1: Tuple of (start, end) for first span
        span2: Tuple of (start, end) for second span

    Returns:
        Length of overlap (0 if no overlap)
    """
    start1, end1 = span1
    start2, end2 = span2

    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    return max(0, overlap_end - overlap_start)


def compute_p_r_f1_from_tp_fp_fn(
    tp_for_p: int | float,
    tp_for_r: int | float,
    fp: int | float,
    fn: int | float,
    fix_edge_cases: bool = False,
) -> Tuple[Union[int, float], Union[int, float], Union[int, float]]:
    """
    Compute precision, recall, and F1 from true/false positive/negative counts.

    Args:
        tp_for_p: True positives for precision calculation
        tp_for_r: True positives for recall calculation
        fp: False positives
        fn: False negatives
        fix_edge_cases: If True, return 0.0 precision when no predictions but
                       gold labels exist

    Returns:
        Tuple of (precision, recall, f1)
    """
    p_denominator = tp_for_p + fp
    r_denominator = tp_for_r + fn

    if fix_edge_cases:
        p = (
            tp_for_p / p_denominator
            if p_denominator > 0
            else (1.0 if r_denominator == 0 else 0.0)
        )
    else:
        p = tp_for_p / p_denominator if p_denominator > 0 else 1.0

    r = tp_for_r / r_denominator if r_denominator > 0 else 1.0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    return p, r, f1


class Metrics(ABC):
    """Abstract base class for metrics aggregation."""

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update metrics with new values."""
        raise NotImplementedError()

    @abstractmethod
    def get_precision_recall_f1(self, *args, **kwargs) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 from accumulated values."""
        raise NotImplementedError()

    @abstractmethod
    def get_values(self) -> List:
        """Get raw accumulated values."""
        raise NotImplementedError()


class MicroMetrics(Metrics):
    """
    Micro-averaged metrics accumulator.

    Accumulates raw TP/FP/FN counts across all samples,
    then computes precision/recall/F1 from the totals.
    """

    def __init__(self):
        self.tp_for_p: float = 0.0
        self.tp_for_r: float = 0.0
        self.fp: float = 0.0
        self.fn: float = 0.0

    def update(
        self,
        tp_for_p: float,
        tp_for_r: float,
        fp: float,
        fn: float,
    ):
        """Accumulate TP/FP/FN counts."""
        self.tp_for_p += tp_for_p
        self.tp_for_r += tp_for_r
        self.fp += fp
        self.fn += fn

    def get_precision_recall_f1(self) -> Tuple[float, float, float]:
        """Compute P/R/F1 from accumulated counts."""
        return compute_p_r_f1_from_tp_fp_fn(
            self.tp_for_p, self.tp_for_r, self.fp, self.fn
        )

    def get_values(self) -> List[float]:
        """Get raw accumulated values."""
        return [self.tp_for_p, self.tp_for_r, self.fp, self.fn]


class MacroMetrics(Metrics):
    """
    Macro-averaged metrics accumulator.

    Accumulates per-sample precision/recall values,
    then computes the average across all samples.
    """

    def __init__(self):
        self.precision: float = 0.0
        self.recall: float = 0.0
        self.f_score: float = 0.0
        self.num_samples: int = 0

    def update(
        self,
        precision: float,
        recall: float,
        f_score: float = 0.0,
        num_samples: int = 1,
    ):
        """Accumulate precision/recall from one or more samples."""
        self.precision += precision
        self.recall += recall
        self.f_score += f_score
        self.num_samples += num_samples

    def get_precision_recall_f1(
        self, use_average_f_score: bool = True
    ) -> Tuple[float, float, float]:
        """Compute average P/R/F1 across all samples.

        args:
            use_average_f_score: If True, macro f-score is the average of f-scores. Else, macro f-score is computed from macro P and R. It is recommended to compute macro f-score as the average of sample-level f-scores. Check this paper from Juri Optiz https://arxiv.org/abs/1911.03347
        """
        precision = self.precision / self.num_samples
        recall = self.recall / self.num_samples
        if use_average_f_score:
            f1 = self.f_score / self.num_samples
        else:
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if precision + recall > 0
                else 0
            )

        return precision, recall, f1

    def get_values(self) -> List[float]:
        """Get raw accumulated values."""
        return [self.precision, self.recall, self.f_score, self.num_samples]
