# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Sample preprocessing utilities for span-level meta-evaluation.

This module provides functions for preprocessing samples with human and
automatic evaluations before comparison, including error counting and
statistics computation.
"""

import logging
from typing import List, Tuple, Dict
from collections import defaultdict
import random

from mt_evaluation.utils import setup_logging
from mt_evaluation.core import (
    AutomaticEvaluation,
    HumanEvaluation,
    Evaluation,
    Error,
    Sample,
)
from mt_evaluation.meta_evaluation import all_severities, UNKNOWN_SEVERITY, MetricStats

logger = logging.getLogger(__name__)


# =============================================================================
# Error Statistics Functions
# =============================================================================

def count_errors_by_severity(errors: List[Error]) -> Dict[str, int]:
    """
    Count errors by severity level.
    
    Args:
        errors: List of Error objects to count
        
    Returns:
        Dictionary mapping severity names to counts
        
    Raises:
        ValueError: If any error in the list is None
    """
    severity_counts = {
        "neutral": 0,
        "minor": 0,
        "major": 0,
        "critical": 0,
    }

    for error in errors:
        if error is None:
            raise ValueError("Error is None")

        severity = error.severity.lower() if error.severity else UNKNOWN_SEVERITY
        if severity in severity_counts:
            severity_counts[severity] += 1
        elif any(severity_name in severity for severity_name in severity_counts):
            for severity_name in severity_counts:
                if severity_name in severity:
                    severity_counts[severity_name] += 1
        else:
            if UNKNOWN_SEVERITY not in severity_counts:
                severity_counts[UNKNOWN_SEVERITY] = 0
            severity_counts[UNKNOWN_SEVERITY] += 1

    return severity_counts


def compute_evaluations_stats(
    evaluations: List[Evaluation],
) -> Tuple[int, int, int]:
    """
    Compute statistics about a list of evaluations.
    
    Args:
        evaluations: List of Evaluation objects
        
    Returns:
        Tuple of (num_samples_with_no_errors, num_total_error_spans, total_span_length)
    """
    if not evaluations:
        return 0, 0, 0

    num_total_error_spans = 0
    num_samples_with_no_errors = 0
    total_span_length = 0

    for evaluation in evaluations:
        num_error_spans = 0
        for error in evaluation.errors:
            if error.span:
                total_span_length += len(error.span)
                num_error_spans += 1

        num_total_error_spans += num_error_spans

        if num_error_spans == 0:
            num_samples_with_no_errors += 1

    return (
        num_samples_with_no_errors,
        num_total_error_spans,
        total_span_length,
    )


# =============================================================================
# Overlap Handling Functions
# =============================================================================

# Severity constants for ranking
severity2rank = {
    "critical": 3,
    "major": 2,
    "minor": 1,
}


def remove_overlapping_errors_func(errors: List[Error]) -> List[Error]:
    """
    Remove overlapping errors from a list, keeping the first in each overlap.

    Args:
        errors: List of errors to process

    Returns:
        List of non-overlapping errors
    """

    def remove_overlapping_side_errors(sorted_errors: List[Error]) -> List[Error]:
        if len(sorted_errors) <= 1:
            return sorted_errors

        new_errors = []
        for i in range(len(sorted_errors) - 1):
            error = sorted_errors[i]
            next_error = sorted_errors[i + 1]

            if error.end > next_error.start:
                pass
            else:
                new_errors.append(error)
        new_errors.append(sorted_errors[-1])
        return new_errors

    filtered_src_errors = remove_overlapping_side_errors(
        sorted(
            [error for error in errors if error.is_source_error], key=lambda e: e.start
        )
    )
    filtered_tgt_errors = remove_overlapping_side_errors(
        sorted(
            [error for error in errors if not error.is_source_error],
            key=lambda e: e.start,
        )
    )

    return filtered_src_errors + filtered_tgt_errors


def preprocess_samples_with_human_evaluations(
    samples: List[Sample],
    included_severities: List[str],
    included_categories: List[str],
    remove_overlapping_errors: bool = False,
    transform_critical_into_major: bool = True,
) -> Tuple[List[Sample], int, int, int, int, int]:
    """
    Preprocess samples with human evaluations by standardizing errors.

    Returns:
        Tuple of (samples, severity_filtered, category_filtered, ill_formed, score_0, overlapping)
    """
    from mt_evaluation.meta_evaluation.span_level.standardization import (
        standardize_text,
        standardize_human_evaluation,
    )

    (
        total_human_severity_filtered,
        total_human_category_filtered,
        total_human_ill_formed,
        total_human_score_0,
        total_overlapping_errors,
    ) = (0, 0, 0, 0, 0)
    new_samples = []
    for sample in samples:
        src = standardize_text(sample.src)
        tgt = standardize_text(sample.tgt)
        (
            human_evaluation,
            human_severity_filtered,
            human_category_filtered,
            human_ill_formed,
            human_score_0,
            num_overlapping_errors,
        ) = standardize_human_evaluation(
            sample.human_evaluation,
            sample,
            included_severities,
            included_categories,
            remove_overlapping_errors,
            transform_critical_into_major,
        )
        new_sample = Sample.from_dict(sample.to_dict())
        new_sample.human_evaluation = human_evaluation
        new_sample.src = src
        new_sample.tgt = tgt
        new_samples.append(new_sample)

        total_human_severity_filtered += human_severity_filtered
        total_human_category_filtered += human_category_filtered
        total_human_ill_formed += human_ill_formed
        total_human_score_0 += human_score_0
        total_overlapping_errors += num_overlapping_errors

    return (
        new_samples,
        total_human_severity_filtered,
        total_human_category_filtered,
        total_human_ill_formed,
        total_human_score_0,
        total_overlapping_errors,
    )


def preprocess_samples_with_automatic_evaluations(
    samples: List[Sample],
    included_severities: List[str],
    included_categories: List[str] | str,
    remove_overlapping_errors: bool = False,
    do_not_assume_no_none_evaluations: bool = False,
    transform_critical_into_major: bool = True,
) -> Tuple[List[Sample], int, int, int, int, int, int, int, int]:
    """
    Preprocess samples with automatic evaluations by standardizing errors.

    Returns:
        Tuple of (samples, severity_filtered, category_filtered, ill_formed,
                  ambiguous, ambiguous_extended, ill_formed_extended, score_0, overlapping)
    """
    from mt_evaluation.meta_evaluation.span_level.standardization import (
        standardize_text,
        standardize_automatic_evaluation,
    )

    (
        total_auto_severity_filtered,
        total_auto_category_filtered,
        total_auto_ill_formed,
        total_auto_num_errors_with_ambiguous_match,
        total_auto_num_errors_with_ambiguous_match_with_extended_span,
        total_auto_ill_formed_extended_span,
        total_auto_num_score_0,
        total_overlapping_errors,
    ) = (0, 0, 0, 0, 0, 0, 0, 0)
    new_samples = []
    for sample in samples:

        src = standardize_text(sample.src)
        tgt = standardize_text(sample.tgt)
        (
            automatic_evaluation,
            auto_severity_filtered,
            auto_category_filtered,
            auto_ill_formed,
            auto_num_errors_with_ambiguous_match,
            auto_num_errors_with_ambiguous_match_with_extended_span,
            auto_num_ill_formed_extended_span,
            auto_num_score_0,
            auto_num_overlapping_errors,
        ) = standardize_automatic_evaluation(
            sample.evaluation,
            sample,
            included_severities,
            included_categories,
            remove_overlapping_errors,
            do_not_assume_no_none_evaluations,
            transform_critical_into_major,
        )
        new_sample = Sample.from_dict(sample.to_dict())
        new_sample.evaluation = automatic_evaluation
        new_sample.src = src
        new_sample.tgt = tgt
        new_samples.append(new_sample)

        total_auto_severity_filtered += auto_severity_filtered
        total_auto_category_filtered += auto_category_filtered
        total_auto_ill_formed += auto_ill_formed
        total_auto_num_errors_with_ambiguous_match += (
            auto_num_errors_with_ambiguous_match
        )
        total_auto_num_errors_with_ambiguous_match_with_extended_span += (
            auto_num_errors_with_ambiguous_match_with_extended_span
        )
        total_auto_ill_formed_extended_span += auto_num_ill_formed_extended_span
        total_auto_num_score_0 += auto_num_score_0
        total_overlapping_errors += auto_num_overlapping_errors

    return (
        new_samples,
        total_auto_severity_filtered,
        total_auto_category_filtered,
        total_auto_ill_formed,
        total_auto_num_errors_with_ambiguous_match,
        total_auto_num_errors_with_ambiguous_match_with_extended_span,
        total_auto_ill_formed_extended_span,
        total_auto_num_score_0,
        total_overlapping_errors,
    )


def errors_overlap(error1: Error, error2: Error) -> bool:
    """Check if two errors overlap in their spans."""
    return not (error1.end <= error2.start or error2.end <= error1.start)


def select_more_severe_error(error1: Error, error2: Error) -> Error:
    """
    Select the more severe error between two overlapping errors.
    If equal severity, select randomly.
    """
    rank1 = severity2rank[error1.severity]
    rank2 = severity2rank[error2.severity]

    if rank1 > rank2:
        return error1
    elif rank2 > rank1:
        return error2
    else:
        return random.choice([error1, error2])


def preprocess_single_autoeval_wrapper(args_tuple):
    """Wrapper for multiprocessing - unpacks arguments and calls preprocess_single_autoeval"""
    (
        autoeval,
        lp2sys2samples_with_automatic_evaluations,
        lp2preprocessed_samples_with_human_evaluations,
        human_as_a_metric_rating_keys,
        do_not_verify_completeness,
        remove_overlapping_errors,
        included_auto_severities,
        included_auto_categories,
        transform_critical_into_major,
        logging_level,
    ) = args_tuple

    setup_logging(logging_level)

    metric_stats, lp2preprocessed_samples_with_automatic_evaluations = (
        preprocess_single_autoeval(
            autoeval=autoeval,
            lp2sys2samples_with_automatic_evaluations=lp2sys2samples_with_automatic_evaluations,
            lp2preprocessed_samples_with_human_evaluations=lp2preprocessed_samples_with_human_evaluations,
            human_as_a_metric_rating_keys=human_as_a_metric_rating_keys,
            do_not_verify_completeness=do_not_verify_completeness,
            remove_overlapping_errors=remove_overlapping_errors,
            included_auto_severities=included_auto_severities,
            included_auto_categories=included_auto_categories,
            transform_critical_into_major=transform_critical_into_major,
        )
    )

    return autoeval, metric_stats, lp2preprocessed_samples_with_automatic_evaluations


def preprocess_single_autoeval(
    autoeval: str,
    lp2sys2samples_with_automatic_evaluations: Dict[str, Dict[str, List[Sample]]],
    lp2preprocessed_samples_with_human_evaluations: Dict[str, List[Sample]],
    human_as_a_metric_rating_keys: List[str],
    do_not_verify_completeness: bool = False,
    remove_overlapping_errors: bool = False,
    included_auto_severities: List[str] = None,
    included_auto_categories: List[str] | str = "All",
    transform_critical_into_major: bool = True,
) -> Tuple[Dict[str, MetricStats], Dict[str, List[Sample]]]:
    """
    Preprocess a single automatic evaluator's results.
    """
    from mt_evaluation.meta_evaluation import UNKNOWN_SEVERITY

    if included_auto_severities is None:
        included_auto_severities = all_severities

    lps = list(lp2sys2samples_with_automatic_evaluations.keys())

    metric_stats = {lp: MetricStats() for lp in lps}

    lp2preprocessed_samples_with_automatic_evaluations = dict()

    for lp in lps:

        samples_with_automatic_evaluations_flattened = [
            sample
            for sys, sys_samples in lp2sys2samples_with_automatic_evaluations[
                lp
            ].items()
            for sample in sys_samples
        ]

        # Count original errors by severity before preprocessing
        original_auto_errors = []
        for sample in samples_with_automatic_evaluations_flattened:
            if any(
                human_as_a_metric_rating_key in autoeval
                for human_as_a_metric_rating_key in human_as_a_metric_rating_keys
            ):
                original_auto_errors.extend(sample.human_evaluation.errors)
            else:
                if do_not_verify_completeness:
                    errors = (
                        sample.evaluation.errors
                        if sample.evaluation is not None
                        else []
                    )
                else:
                    errors = sample.evaluation.errors
                original_auto_errors.extend(errors)

        original_auto_counts = count_errors_by_severity(original_auto_errors)
        if original_auto_counts.get(UNKNOWN_SEVERITY, 0) > 0:
            logger.error(
                f"{original_auto_counts.get(UNKNOWN_SEVERITY, 0)} unknown severities found in {lp} for {autoeval}."
            )

        # Preprocessing samples with automatic evaluations
        num_errors_with_ambiguous_match = 0
        num_errors_with_ambiguous_match_with_extended_span = 0
        num_ill_formed_extended_span = 0
        if any(
            human_as_a_metric_rating_key in autoeval
            for human_as_a_metric_rating_key in human_as_a_metric_rating_keys
        ):
            (
                preprocessed_samples_with_automatic_evaluations_flattened,
                num_auto_severity_filtered_errors,
                num_auto_category_filtered_errors,
                num_auto_ill_formed_errors,
                num_auto_score_0_errors,
                num_auto_overlapping_errors,
            ) = preprocess_samples_with_human_evaluations(
                samples_with_automatic_evaluations_flattened,
                included_severities=included_auto_severities,
                included_categories=included_auto_categories,
                remove_overlapping_errors=remove_overlapping_errors,
            )
        else:
            (
                preprocessed_samples_with_automatic_evaluations_flattened,
                num_auto_severity_filtered_errors,
                num_auto_category_filtered_errors,
                num_auto_ill_formed_errors,
                num_errors_with_ambiguous_match,
                num_errors_with_ambiguous_match_with_extended_span,
                num_ill_formed_extended_span,
                num_auto_score_0_errors,
                num_auto_overlapping_errors,
            ) = preprocess_samples_with_automatic_evaluations(
                samples_with_automatic_evaluations_flattened,
                included_severities=included_auto_severities,
                included_categories=included_auto_categories,
                remove_overlapping_errors=remove_overlapping_errors,
                do_not_assume_no_none_evaluations=do_not_verify_completeness,
                transform_critical_into_major=transform_critical_into_major,
            )

        # Convert human evaluations to automatic evaluation format
        if any(
            human_as_a_metric_rating_key in autoeval
            for human_as_a_metric_rating_key in human_as_a_metric_rating_keys
        ):
            for sample in preprocessed_samples_with_automatic_evaluations_flattened:
                sample.evaluation = AutomaticEvaluation(
                    score=sample.human_evaluation.score,
                    errors=sample.human_evaluation.errors,
                    annotation="",
                    parsing_error=False,
                )
                sample.human_evaluation = None

        lp2preprocessed_samples_with_automatic_evaluations[lp] = (
            preprocessed_samples_with_automatic_evaluations_flattened
        )

        # Count final errors by severity after preprocessing
        final_auto_errors = [
            error
            for sample in preprocessed_samples_with_automatic_evaluations_flattened
            for error in sample.evaluation.errors
        ]
        final_auto_counts = count_errors_by_severity(final_auto_errors)

        (
            samples_with_automatic_evaluations,
            samples_with_human_evaluations,
        ) = ([], [])
        for sample_with_auto, sample_with_human in zip(
            preprocessed_samples_with_automatic_evaluations_flattened,
            lp2preprocessed_samples_with_human_evaluations[lp],
        ):

            samples_with_automatic_evaluations.append(sample_with_auto)
            samples_with_human_evaluations.append(sample_with_human)

        (
            num_auto_no_errors,
            num_auto_total_errors,
            auto_total_span_length,
        ) = compute_evaluations_stats(
            [sample.evaluation for sample in samples_with_automatic_evaluations]
        )

        metric_stats[lp].update(
            len(samples_with_automatic_evaluations),
            num_auto_total_errors,
            num_auto_no_errors,
            num_auto_ill_formed_errors,
            num_errors_with_ambiguous_match,
            num_errors_with_ambiguous_match_with_extended_span,
            num_ill_formed_extended_span,
            num_auto_severity_filtered_errors,
            num_auto_category_filtered_errors,
            original_auto_counts,
            final_auto_counts,
            auto_total_span_length,
            num_auto_score_0_errors,
            num_auto_overlapping_errors,
        )

        if not num_auto_total_errors > 0:
            raise RuntimeError(
                f"Metric name: {autoeval} has no errors with your filtering selection!"
            )

    return (
        metric_stats,
        lp2preprocessed_samples_with_automatic_evaluations,
    )
