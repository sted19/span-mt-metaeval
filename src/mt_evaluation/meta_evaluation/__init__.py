# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Meta-evaluation module for assessing the quality of automatic MT evaluation metrics.

This module provides tools for evaluating how well automatic metrics correlate
with human judgments, including span-level and score-level analysis.

Classes:
    SeverityCounts: Container for error counts by severity level.
    MetricStats: Statistics about metric performance.
    MetricResults: Results container for precision/recall/F1.
    TypeMetricResults: Results for a specific metric type.
    TypeSentinelCounts: Sentinel counts for a specific metric type.
    SentinelCounts: Container for sentinel experiment counts.

Note:
    WMT language pair constants are re-exported from mt_evaluation.core for
    backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Dict

# Re-export constants from core for backward compatibility
from mt_evaluation.core import (
    all_severities,
    UNKNOWN_SEVERITY,
    wmt22_lps,
    wmt23_lps,
    wmt24_lps,
    wmt25_lps_mqm,
    wmt25_lps_esa,
    wmt25_lps,
)

# Metric types for span-level evaluation
METRIC_TYPES = [
    "Exact\nMatch",
    "Partial\nOverlap",
    "Character\nCounts",
    "Character\nProportion",
]


@dataclass
class SeverityCounts:
    """
    Container for counting errors by severity level.

    Attributes:
        neutral: Count of neutral severity errors.
        minor: Count of minor severity errors.
        major: Count of major severity errors.
        critical: Count of critical severity errors.
    """
    neutral: int = 0
    minor: int = 0
    major: int = 0
    critical: int = 0

    def add_counts(self, counts: Dict[str, int]) -> None:
        """Add counts from a dictionary."""
        self.neutral += counts.get("neutral", 0)
        self.minor += counts.get("minor", 0)
        self.major += counts.get("major", 0)
        self.critical += counts.get("critical", 0)

    def total(self) -> int:
        """Return total count across all severities."""
        return self.neutral + self.minor + self.major + self.critical

    def __getitem__(self, key: str) -> int:
        """Allow dict-like read access."""
        if not hasattr(self, key):
            raise KeyError(f"Invalid severity: {key}")
        return getattr(self, key)

    def __setitem__(self, key: str, value: int) -> None:
        """Allow dict-like write access."""
        if not hasattr(self, key):
            raise KeyError(f"Invalid severity: {key}")
        setattr(self, key, value)


@dataclass
class MetricStats:
    """
    Statistics about metric evaluation performance.

    Tracks various counts and statistics during the evaluation of metrics,
    including error counts, sample counts, and severity breakdowns.
    """
    num_samples: int = 0
    num_errors: int = 0
    num_samples_with_no_errors: int = 0
    num_ill_formed_errors: int = 0
    num_errors_with_ambiguous_match: int = 0
    num_errors_with_ambiguous_match_with_extended_span: int = 0
    num_ill_formed_extended_span: int = 0
    num_severity_filtered_errors: int = 0
    num_category_filtered_errors: int = 0
    original_severity_counts: SeverityCounts = field(default_factory=SeverityCounts)
    final_severity_counts: SeverityCounts = field(default_factory=SeverityCounts)
    total_span_length: int = 0
    num_score_0_errors: int = 0
    num_overlapping_errors: int = 0

    # Properties computed from other fields
    avg_errors_per_sample: float = 0
    avg_span_length: float = 0
    num_removed_errors: int = 0
    severity_breakdown: Dict[str, str] = field(default_factory=dict)

    def update(
        self,
        num_samples: int,
        num_errors: int,
        num_samples_with_no_errors: int,
        num_ill_formed_errors: int,
        num_errors_with_ambiguous_match: int,
        num_errors_with_ambiguous_match_with_extended_span: int,
        num_ill_formed_extended_span: int,
        num_severity_filtered_errors: int,
        num_category_filtered_errors: int,
        original_severity_counts: Dict[str, int],
        final_severity_counts: Dict[str, int],
        total_span_length: int,
        num_score_0_errors: int,
        num_overlapping_errors: int,
        **kwargs,  # Allow updating via unpacking another MetricStats object
    ) -> None:
        """Update stats from evaluation results."""
        self.num_samples += num_samples
        self.num_errors += num_errors
        self.num_samples_with_no_errors += num_samples_with_no_errors
        self.num_ill_formed_errors += num_ill_formed_errors
        self.num_errors_with_ambiguous_match += num_errors_with_ambiguous_match
        self.num_errors_with_ambiguous_match_with_extended_span += (
            num_errors_with_ambiguous_match_with_extended_span
        )
        self.num_ill_formed_extended_span += num_ill_formed_extended_span
        self.num_severity_filtered_errors += num_severity_filtered_errors
        self.num_category_filtered_errors += num_category_filtered_errors
        self.original_severity_counts.add_counts(original_severity_counts)
        self.final_severity_counts.add_counts(final_severity_counts)
        self.total_span_length += total_span_length
        self.num_score_0_errors += num_score_0_errors
        self.num_overlapping_errors += num_overlapping_errors

        self.avg_errors_per_sample = self.num_errors / self.num_samples
        self.avg_span_length = self.total_span_length / self.num_errors
        self.num_removed_errors = (
            self.original_severity_counts.total() - self.final_severity_counts.total()
        )
        self.severity_breakdown = {}
        for severity in all_severities:
            original = self.original_severity_counts[severity]
            final = self.final_severity_counts[severity]
            self.severity_breakdown[severity] = f"{original} / {final}"


@dataclass
class TypeMetricResults:
    """Results for a specific metric type (precision, recall, F1)."""
    precision: float = 0
    recall: float = 0
    f1: float = 0

    def update(self, precision: float, recall: float, f1: float) -> None:
        """Update results by adding values."""
        self.precision += precision
        self.recall += recall
        self.f1 += f1


@dataclass
class MetricResults:
    """
    Container for metric results across different metric types.

    Stores precision/recall/F1 results for each metric type (Exact Match,
    Partial Overlap, Character Counts, Character Proportion).
    """
    results: Dict[str, TypeMetricResults] = field(
        default_factory=lambda: {
            metric_type: TypeMetricResults() for metric_type in METRIC_TYPES
        }
    )

    def update(self, metric_type: str, precision: float, recall: float, f1: float) -> None:
        """Update results for a specific metric type."""
        self.results[metric_type].update(precision, recall, f1)

    def get(self, metric_type: str) -> TypeMetricResults:
        """Get results for a specific metric type."""
        return self.results[metric_type]


@dataclass
class TypeSentinelCounts:
    """Sentinel experiment counts for a specific metric type."""
    num_ext_only_greater_than_normal: int = 0
    num_ext_only_smaller_or_equal_than_normal: int = 0
    num_no_ext_greater_than_normal: int = 0
    num_no_ext_smaller_or_equal_than_normal: int = 0
    num_remove_all_1_greater_than_normal: int = 0
    num_remove_all_1_smaller_or_equal_than_normal: int = 0

    def update(
        self,
        num_ext_only_greater_than_normal: int = 0,
        num_ext_only_smaller_or_equal_than_normal: int = 0,
        num_no_ext_greater_than_normal: int = 0,
        num_no_ext_smaller_or_equal_than_normal: int = 0,
        num_remove_all_1_greater_than_normal: int = 0,
        num_remove_all_1_smaller_or_equal_than_normal: int = 0,
    ) -> None:
        """Update sentinel counts."""
        self.num_ext_only_greater_than_normal += num_ext_only_greater_than_normal
        self.num_ext_only_smaller_or_equal_than_normal += (
            num_ext_only_smaller_or_equal_than_normal
        )
        self.num_no_ext_greater_than_normal += num_no_ext_greater_than_normal
        self.num_no_ext_smaller_or_equal_than_normal += (
            num_no_ext_smaller_or_equal_than_normal
        )
        self.num_remove_all_1_greater_than_normal += (
            num_remove_all_1_greater_than_normal
        )
        self.num_remove_all_1_smaller_or_equal_than_normal += (
            num_remove_all_1_smaller_or_equal_than_normal
        )


@dataclass
class SentinelCounts:
    """
    Container for sentinel experiment counts across metric types.

    Used to track results of sentinel experiments that help evaluate
    the reliability of span-level metrics.
    """
    counts: Dict[str, TypeSentinelCounts] = field(
        default_factory=lambda: {
            metric_type: TypeSentinelCounts() for metric_type in METRIC_TYPES
        }
    )

    def update(self, metric_type: str, *args) -> None:
        """Update counts for a specific metric type."""
        self.counts[metric_type].update(*args)

    def get(self, metric_type: str) -> TypeSentinelCounts:
        """Get counts for a specific metric type."""
        return self.counts[metric_type]


__all__ = [
    # Re-exported from core
    "all_severities",
    "UNKNOWN_SEVERITY",
    "wmt22_lps",
    "wmt23_lps",
    "wmt24_lps",
    "wmt25_lps_mqm",
    "wmt25_lps_esa",
    "wmt25_lps",
    # Defined in this module
    "METRIC_TYPES",
    "SeverityCounts",
    "MetricStats",
    "TypeMetricResults",
    "MetricResults",
    "TypeSentinelCounts",
    "SentinelCounts",
]
