# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Metrics classes for span-level meta-evaluation.

This module contains classes for computing and aggregating precision, recall, and F1
metrics at both micro and macro levels.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Union


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
        self.num_samples: int = 0

    def update(
        self,
        precision: float,
        recall: float,
        num_samples: int = 1,
    ):
        """Accumulate precision/recall from one or more samples."""
        self.precision += precision
        self.recall += recall
        self.num_samples += num_samples

    def get_precision_recall_f1(self) -> Tuple[float, float, float]:
        """Compute average P/R/F1 across all samples."""
        precision = self.precision / self.num_samples
        recall = self.recall / self.num_samples
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )

        return precision, recall, f1

    def get_values(self) -> List[float]:
        """Get raw accumulated values."""
        return [self.precision, self.recall, self.num_samples]
