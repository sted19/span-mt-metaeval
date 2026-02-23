# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Type aliases and dataclasses for the MT Evaluation Framework.

This module defines:
- Reusable type aliases for complex nested dictionary types
- Dataclasses for structured return values from functions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mt_evaluation.core.datastructures import Sample, Error
    from mt_evaluation.meta_evaluation.span_level.metrics import Metrics, MicroMetrics, MacroMetrics


# =============================================================================
# Dataclasses for structured return values
# =============================================================================

@dataclass
class StandardizationResult:
    """
    Result from standardizing a single error (human or automatic).
    
    Attributes:
        error: The standardized error, or None if filtered out
        severity_filtered: Whether the error was filtered due to severity
        category_filtered: Whether the error was filtered due to category
        ill_formed: Whether the error was ill-formed (invalid span)
        score_0: Whether the error has a score of 0
        num_occurrences: Number of times span was found (automatic only)
        num_extended_occurrences: Number of times extended span was found (automatic only)
        ill_formed_extended_span: Whether extended span was ill-formed (automatic only)
    """
    error: Optional["Error"] = None
    severity_filtered: bool = False
    category_filtered: bool = False
    ill_formed: bool = False
    score_0: bool = False
    # Automatic error specific fields
    num_occurrences: int = 0
    num_extended_occurrences: int = 0
    ill_formed_extended_span: bool = False
    
    @property
    def was_filtered(self) -> bool:
        """Return True if the error was filtered out for any reason."""
        return self.error is None
    
    @property
    def filter_reason(self) -> Optional[str]:
        """Return the reason the error was filtered, or None if not filtered."""
        if self.severity_filtered:
            return "severity"
        if self.category_filtered:
            return "category"
        if self.ill_formed:
            return "ill_formed"
        if self.score_0:
            return "score_0"
        return None


@dataclass
class PreprocessingStats:
    """
    Statistics from preprocessing a set of samples.
    
    Attributes:
        severity_filtered: Number of errors filtered due to severity
        category_filtered: Number of errors filtered due to category
        ill_formed: Number of ill-formed errors
        score_0: Number of errors with score 0
        overlapping: Number of overlapping errors removed
        ambiguous_match: Number of errors with ambiguous span match (automatic only)
        ambiguous_match_extended: Number of extended spans with ambiguous match (automatic only)
        ill_formed_extended_span: Number of ill-formed extended spans (automatic only)
    """
    severity_filtered: int = 0
    category_filtered: int = 0
    ill_formed: int = 0
    score_0: int = 0
    overlapping: int = 0
    # Automatic evaluation specific fields
    ambiguous_match: int = 0
    ambiguous_match_extended: int = 0
    ill_formed_extended_span: int = 0
    
    def __add__(self, other: "PreprocessingStats") -> "PreprocessingStats":
        """Add two PreprocessingStats together."""
        return PreprocessingStats(
            severity_filtered=self.severity_filtered + other.severity_filtered,
            category_filtered=self.category_filtered + other.category_filtered,
            ill_formed=self.ill_formed + other.ill_formed,
            score_0=self.score_0 + other.score_0,
            overlapping=self.overlapping + other.overlapping,
            ambiguous_match=self.ambiguous_match + other.ambiguous_match,
            ambiguous_match_extended=self.ambiguous_match_extended + other.ambiguous_match_extended,
            ill_formed_extended_span=self.ill_formed_extended_span + other.ill_formed_extended_span,
        )
    
    @property
    def total_filtered(self) -> int:
        """Total number of errors filtered for any reason."""
        return (
            self.severity_filtered + 
            self.category_filtered + 
            self.ill_formed + 
            self.score_0 +
            self.overlapping
        )

# =============================================================================
# Sample-based nested dictionary types
# =============================================================================

# system_name -> List[Sample]
Sys2Samples = Dict[str, List["Sample"]]

# language_pair -> system_name -> List[Sample]
Lp2Sys2Samples = Dict[str, Sys2Samples]

# autoeval_name -> language_pair -> system_name -> List[Sample]
Autoeval2Lp2Sys2Samples = Dict[str, Lp2Sys2Samples]

# rater_name -> language_pair -> system_name -> List[Sample]
Rater2Lp2Sys2Samples = Dict[str, Lp2Sys2Samples]

# test_set -> language_pair -> system_name -> List[Sample]
TestSet2Lp2Sys2Samples = Dict[str, Lp2Sys2Samples]

# language_pair -> List[Sample] (flattened samples)
Lp2Samples = Dict[str, List["Sample"]]

# autoeval_name -> language_pair -> List[Sample]
Autoeval2Lp2Samples = Dict[str, Lp2Samples]


# =============================================================================
# Metrics-related nested dictionary types
# =============================================================================

# metric_type -> Metrics (e.g., "Exact
Match" -> MicroMetrics)
MetricType2Metrics = Dict[str, "Metrics"]

# matching_type -> metric_type -> Metrics (e.g., "matching" -> "Exact
Match" -> Metrics)
MatchType2MetricType2Metrics = Dict[str, MetricType2Metrics]

# aggregation_type -> matching_type -> metric_type -> Metrics
AggrType2MatchType2MetricType2Metrics = Dict[str, MatchType2MetricType2Metrics]

# language_pair -> aggregation_type -> matching_type -> metric_type -> Metrics
Lp2AggrType2MatchType2MetricType2Metrics = Dict[str, AggrType2MatchType2MetricType2Metrics]

# autoeval_name -> language_pair -> aggregation_type -> matching_type -> metric_type -> Metrics
Autoeval2Lp2Metrics = Dict[str, Lp2AggrType2MatchType2MetricType2Metrics]
