# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Text and error standardization utilities for span-level meta-evaluation.

This module provides functions for standardizing text and error spans
for comparison between automatic and human evaluations.
"""

import logging
from typing import List, Tuple, Union, Optional

from mt_evaluation.utils import find_all_literal
from mt_evaluation.core import (
    AutomaticEvaluation,
    HumanEvaluation,
    Error,
    Sample,
    source_error,
    source_error2,
    non_translation,
    unintelligible,
    omission,
    source_issue,
    source_issue2,
    creative_reinterpretation,
)
from mt_evaluation.meta_evaluation import all_severities

logger = logging.getLogger(__name__)


def standardize_text(text: str) -> str:
    """
    Standardize text for comparison by normalizing case and quotation marks.

    NOTE: Important: Do not modify the spacing or remove/add characters!
    Position of characters must be preserved!

    Args:
        text: Text to standardize (string or list of strings)

    Returns:
        Standardized text string

    Raises:
        ValueError: If span is None
    """
    if text is None:
        raise ValueError("You passed a None text")

    # Normalize quotation marks - replace various quote types with standard ASCII quotes
    symbol_mapping = str.maketrans(
        {
            "“": '"',  # Left double quotation mark
            "”": '"',  # right double quotation mark
            "’": "'",  # Left single quotation mark
            "‘": "'",  # Right single quotation mark
            "«": '"',  # Left-pointing double angle quotation mark
            "»": '"',  # Right-pointing double angle quotation mark
            "‹": "'",  # Single left-pointing angle quotation mark
            "›": "'",  # Single right-pointing angle quotation mark"
        }
    )

    std_text = []
    for ch in text.translate(symbol_mapping):
        lower_ch = ch.lower()
        if len(lower_ch) == 1:
            std_text.append(lower_ch)
        else:
            std_text.append(ch)

    std_text = "".join(std_text)

    assert len(std_text) == len(
        text
    ), f"Standardization should not change text length! Original: {text}, Standardized: {std_text}"

    return std_text


def standardize_severity(severity: str, transform_critical_into_major: bool) -> str:
    """
    Standardize severity string to match standard severity names.

    Args:
        severity: Raw severity string
        transform_critical_into_major: If True, map critical to major

    Returns:
        Standardized severity string

    Raises:
        ValueError: If severity is unknown
    """
    severity_mapping = {severity: severity for severity in all_severities}
    if transform_critical_into_major:
        severity_mapping["critical"] = "major"

    for severity_name in all_severities:
        if severity_name in severity:
            return severity_mapping[severity_name]

    raise ValueError(f"Unknown severity: {severity}")


def find_span_in_paragraph(
    paragraph: str, span: str, extended_span: Optional[str] = None
) -> Tuple[int, int, int, int]:
    """
    Find a span within a paragraph and return position and match counts.

    Args:
        paragraph: Text to search in
        span: Span to find
        extended_span: Optional extended span for disambiguation

    Returns:
        Tuple of (start, end, num_occurrences, num_extended_occurrences)

    Raises:
        ValueError: If span cannot be found
    """
    matches = find_all_literal(paragraph, span)
    if len(matches) == 1:
        return matches[0][0], matches[0][1], 1, 1

    if len(matches) > 1:
        if extended_span is None or extended_span.find(span) == -1:
            return matches[0][0], matches[0][1], len(matches), len(matches)

        extended_matches = find_all_literal(paragraph, extended_span)
        if len(extended_matches) >= 1:
            matches_in_extended_span = find_all_literal(extended_span, span)
            start = extended_matches[0][0] + matches_in_extended_span[0][0]
            end = start + len(span)

            return (
                start,
                end,
                len(matches),
                len(extended_matches),
            )
        else:
            raise ValueError(
                f"Error finding the extended span!\n"
                f"Standardized paragraph: {paragraph}"
                f"Standardized span: {extended_span}"
            )

    raise ValueError(
        f"Error finding standardized span!\n"
        f"Standardized paragraph: {paragraph}"
        f"Standardized span: {span}"
    )


def standardize_human_error(
    error: Error,
    sample: Sample,
    included_severities: List[str],
    included_categories: List[str] | str,
    transform_critical_into_major: bool = True,
) -> Tuple[Error | None, bool, bool, bool, bool]:
    """
    Standardize a human error annotation.

    Args:
        error: The error to standardize
        sample: The sample containing the error
        included_severities: List of severities to include
        included_categories: Categories to include (list or "All")
        transform_critical_into_major: Whether to map critical to major

    Returns:
        Tuple of (processed_error, severity_filtered, category_filtered, ill_formed, score_0)
    """
    severity_filtered, category_filtered, score_0 = False, False, False

    if error is None:
        raise ValueError(f"Error is None: {error}")

    severity = error.severity.lower()
    if severity not in included_severities:
        severity_filtered = True
    else:
        severity = standardize_severity(severity, transform_critical_into_major)

    category = error.category.lower()
    if isinstance(included_categories, list) and not any(
        cat in category for cat in included_categories
    ):
        category_filtered = True

    if error.score is None:
        raise ValueError(f"Human error with score=None: {error}")
    if error.score == 0:
        assert (
            source_error in error.category
            or source_error2 in error.category
            or source_issue in error.category
            or creative_reinterpretation in error.category
            or source_issue2 in error.category
        )
        score_0 = True

    if error.span is None or error.span == "":
        logger.debug(f"\nEmpty or None error span for Human error {error}\n")
        return None, severity_filtered, category_filtered, True, score_0

    standard_span = standardize_text(error.span)
    standard_src = standardize_text(sample.src)
    standard_tgt = standardize_text(sample.tgt)

    if standard_span == "":
        logger.debug(f"\nEmpty error span for human error {error}\n")
        return None, severity_filtered, category_filtered, True, score_0

    assert (
        error.is_source_error is not None
    ), f"error.is_source_error is None for error {error}"

    if error.is_source_error and standard_span not in standard_src:
        logger.debug(
            f"\nHuman error span '{standard_span}' not contained in '{standard_src}'"
            f"Error: {error}"
        )
        return None, severity_filtered, category_filtered, True, score_0

    if not error.is_source_error and standard_span not in standard_tgt:
        logger.debug(
            f"\nHuman error span '{standard_span}' not contained in '{standard_tgt}'"
            f"Error: {error}"
        )
        return None, severity_filtered, category_filtered, True, score_0

    if severity_filtered or category_filtered or score_0:
        return None, severity_filtered, category_filtered, False, score_0

    if error.is_source_error:
        assert standard_span == standard_src[error.start : error.end]
        if error.end > len(standard_src):
            error.end = len(standard_src)
    else:
        assert standard_span == standard_tgt[error.start : error.end]
        if error.end > len(standard_tgt):
            error.end = len(standard_tgt)

    new_error = Error(
        span=standard_span,
        category=error.category,
        severity=severity,
        start=error.start,
        end=error.end,
        is_source_error=error.is_source_error,
        score=error.score,
        explanation=error.explanation,
    )

    return new_error, False, False, False, False


def standardize_automatic_error(
    error: Error,
    sample: Sample,
    included_severities: List[str],
    included_categories: Union[str, List[str]],
    transform_critical_into_major: bool = True,
) -> Tuple[Error | None, bool, bool, bool, int, int, bool, bool]:
    """
    Standardize an automatic error annotation.

    Returns:
        Tuple of (processed_error, severity_filtered, category_filtered, ill_formed,
                  num_potential_matches, num_extended_occurrences, ill_formed_extended_span, score_0)
    """
    if error is None:
        raise ValueError(f"Error is None: {error}")

    severity_filtered, category_filtered, score_0, none_or_ill_formed_extended_span = (
        False,
        False,
        False,
        False,
    )

    severity = error.severity.lower()
    if not any(severity_name in severity for severity_name in included_severities):
        severity_filtered = True
    else:
        severity = standardize_severity(severity, transform_critical_into_major)

    category = error.category.lower()
    if isinstance(included_categories, list) and not any(
        cat in category for cat in included_categories
    ):
        category_filtered = True

    extended_span = error.extended_span
    if extended_span is None or extended_span == "":
        none_or_ill_formed_extended_span = True

    # Assign is_source_error based on category
    if (
        omission in category
        or source_issue in category
        or source_issue2 in category
        or source_error in category
        or source_error2 in category
    ):
        error.is_source_error = True
    else:
        error.is_source_error = False

    # Remove source errors as they don't contribute to the final score
    if (
        source_issue in category
        or source_issue2 in category
        or source_error in category
        or source_error2 in category
    ):
        score_0 = True

    error_span = error.span
    if unintelligible in category:
        error_span = sample.tgt
        extended_span = sample.tgt
        category = unintelligible

    if non_translation in category:
        error_span = sample.tgt
        extended_span = sample.tgt
        category = non_translation

    if error_span is None or error_span == "":
        logger.debug(f"\nError span is None or empty for error {error}\n")
        return (
            None,
            severity_filtered,
            category_filtered,
            True,
            0,
            0,
            True,
            score_0,
        )

    # Handle list spans (backward compatibility)
    if type(error_span) == list:
        if len(error_span) > 0:
            error_span = error_span[0]
            if error_span is None or error_span == "":
                logger.debug(
                    f"\nError span is None or empty. This error will be removed from sample: {sample}\n"
                )
                return (
                    None,
                    severity_filtered,
                    category_filtered,
                    True,
                    0,
                    0,
                    True,
                    score_0,
                )
        else:
            return (
                None,
                severity_filtered,
                category_filtered,
                True,
                0,
                0,
                True,
                score_0,
            )

    # Standardize and validate spans
    standard_span = standardize_text(error_span)
    standard_extended_span = standardize_text(extended_span) if extended_span else None
    standard_src = standardize_text(sample.src)
    standard_tgt = standardize_text(sample.tgt)

    if (
        standard_extended_span is not None
        and standard_extended_span.find(standard_span) == -1
    ):
        none_or_ill_formed_extended_span = True

    if standard_span == "":
        logger.debug(f"\nStandard error span is empty for error {error}\n")
        return (
            None,
            severity_filtered,
            category_filtered,
            True,
            0,
            0,
            True,
            score_0,
        )

    if standard_extended_span is not None and standard_extended_span == "":
        none_or_ill_formed_extended_span = True

    if standard_span not in standard_src and standard_span not in standard_tgt:
        logger.debug(
            f"\nError span '{standard_span}' not contained in source nor target\n"
            f"Source: {standard_src}\n"
            f"Target: {standard_tgt}\n"
        )
        return (
            None,
            severity_filtered,
            category_filtered,
            True,
            0,
            0,
            True,
            score_0,
        )

    if (
        standard_extended_span is not None
        and standard_extended_span not in standard_src
        and standard_extended_span not in standard_tgt
    ):
        none_or_ill_formed_extended_span = True

    assert (
        error.is_source_error
        if omission in category
        or source_issue in category
        or source_error in category
        or source_error2 in category
        else not error.is_source_error
    ), f"Error category is {category} but error.is_source_error={error.is_source_error}"

    if error.is_source_error and standard_span not in standard_src:
        logger.debug(f"\nSpan contained in target and not source for error: {error}")
        return (
            None,
            severity_filtered,
            category_filtered,
            True,
            0,
            0,
            True,
            score_0,
        )
    if (
        error.is_source_error
        and standard_extended_span is not None
        and standard_extended_span not in standard_src
    ):
        none_or_ill_formed_extended_span = True

    if not error.is_source_error and standard_span not in standard_tgt:
        logger.debug(f"\nSpan contained in source and not target for error: {error}")
        return (
            None,
            severity_filtered,
            category_filtered,
            True,
            0,
            0,
            True,
            score_0,
        )

    if (
        not error.is_source_error
        and standard_extended_span is not None
        and standard_extended_span not in standard_tgt
    ):
        none_or_ill_formed_extended_span = True

    if severity_filtered or category_filtered or score_0:
        return (
            None,
            severity_filtered,
            category_filtered,
            False,
            0,
            0,
            none_or_ill_formed_extended_span,
            score_0,
        )

    score = error.score
    if score is None:
        logger.debug(f"Score is None for error {error}. Setting it to 0.0")
        score = 0.0

    standard_extended_span = (
        None if none_or_ill_formed_extended_span else standard_extended_span
    )

    start, end, num_occurrences, num_extended_occurrences = error.start, error.end, 0, 0
    if start is None or end is None:
        assert start is None and end is None

        if error.is_source_error:
            start, end, num_occurrences, num_extended_occurrences = (
                find_span_in_paragraph(
                    standard_src, standard_span, standard_extended_span
                )
            )
        elif not error.is_source_error:
            start, end, num_occurrences, num_extended_occurrences = (
                find_span_in_paragraph(
                    standard_tgt, standard_span, standard_extended_span
                )
            )
        else:
            raise RuntimeError(
                f"Error is either a source error or not. Instead, error.is_source_error = {error.is_source_error}"
            )

    if error.is_source_error:
        assert standard_span == standard_src[start:end]
        if end > len(standard_src):
            end = len(standard_src)
    else:
        assert standard_span == standard_tgt[start:end]
        if end > len(standard_tgt):
            end = len(standard_tgt)

    new_error = Error(
        span=standard_span,
        extended_span=standard_extended_span,
        category=category,
        severity=severity,
        start=start,
        end=end,
        is_source_error=error.is_source_error,
        score=score,
        explanation=error.explanation,
    )

    return (
        new_error,
        False,
        False,
        False,
        num_occurrences,
        num_extended_occurrences,
        none_or_ill_formed_extended_span,
        score_0,
    )


def standardize_human_evaluation(
    human_evaluation: HumanEvaluation | None,
    sample: Sample,
    included_severities: List[str],
    included_categories: List[str],
    remove_overlapping_errors: bool = False,
    transform_critical_into_major: bool = True,
) -> Tuple[HumanEvaluation | None, bool, bool, bool, int, int]:
    """
    Standardize a human evaluation by processing all its errors.

    Returns:
        Tuple of (evaluation, severity_filtered, category_filtered, ill_formed, score_0, num_overlapping)
    """
    from mt_evaluation.meta_evaluation.span_level.preprocessing import (
        remove_overlapping_errors_func,
    )

    if human_evaluation is None:
        return None, False, False, False, 0, 0

    (
        local_severity_filtered,
        local_category_filtered,
        local_ill_formed,
        local_score_0,
    ) = (
        False,
        False,
        False,
        0,
    )

    new_errors = []
    for error in human_evaluation.errors:
        processed_error, severity_filtered, category_filtered, ill_formed, score_0 = (
            standardize_human_error(
                error,
                sample,
                included_severities,
                included_categories,
                transform_critical_into_major,
            )
        )

        if severity_filtered or category_filtered or ill_formed or score_0:
            assert not processed_error

        if not processed_error:
            assert severity_filtered or category_filtered or ill_formed or score_0

        local_severity_filtered += int(severity_filtered)
        local_category_filtered += int(category_filtered)
        local_ill_formed += int(ill_formed)
        local_score_0 += int(score_0)

        if (
            not severity_filtered
            and not category_filtered
            and not ill_formed
            and not score_0
        ):
            new_errors.append(processed_error)

    non_overlapping_errors = remove_overlapping_errors_func(new_errors)
    num_overlapping_errors = len(new_errors) - len(non_overlapping_errors)
    if remove_overlapping_errors:
        new_errors = non_overlapping_errors

    new_human_evaluation = HumanEvaluation(
        score=sum(error.score for error in new_errors),
        errors=new_errors,
    )
    return (
        new_human_evaluation,
        local_severity_filtered,
        local_category_filtered,
        local_ill_formed,
        local_score_0,
        num_overlapping_errors,
    )


def standardize_automatic_evaluation(
    automatic_evaluation: AutomaticEvaluation,
    sample: Sample,
    included_severities: List[str],
    included_categories: List[str] | str,
    remove_overlapping_errors: bool = False,
    do_not_assume_no_none_evaluations: bool = False,
    transform_critical_into_major: bool = True,
) -> Tuple[AutomaticEvaluation | None, int, int, int, int, int, int, int, int]:
    """
    Standardize an automatic evaluation by processing all its errors.

    Returns:
        Tuple of (evaluation, severity_filtered, category_filtered, ill_formed,
                  num_ambiguous, num_ambiguous_extended, ill_formed_extended, score_0, num_overlapping)
    """
    from mt_evaluation.meta_evaluation.span_level.preprocessing import (
        remove_overlapping_errors_func,
    )

    if do_not_assume_no_none_evaluations and automatic_evaluation is None:
        new_automatic_evaluation = AutomaticEvaluation(
            score=0.0,
            errors=[],
            annotation="",
            user_prompt="",
            system_prompt="",
            few_shots=None,
            cost=0.0,
            parsing_error=False,
        )
        return new_automatic_evaluation, 0, 0, 0, 0, 0, 0, 0, 0

    if automatic_evaluation.parsing_error:
        logger.debug("Automatic evaluation has parsing errors")
        assert automatic_evaluation.errors == []
        assert automatic_evaluation.score == 0

    new_errors = []
    (
        local_severity_filtered,
        local_category_filtered,
        local_ill_formed,
        num_errors_with_ambiguous_match,
        num_errors_with_ambiguous_match_with_extended_span,
        local_num_potential_span_matches,
        local_num_potential_span_matches_with_extended_span,
        local_ill_formed_extended_span,
        local_score_0,
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0)

    for error in automatic_evaluation.errors:
        (
            processed_error,
            severity_filtered,
            category_filtered,
            ill_formed,
            num_potential_span_matches,
            num_potential_span_matches_with_extended_span,
            ill_formed_extended_span,
            score_0,
        ) = standardize_automatic_error(
            error,
            sample,
            included_severities,
            included_categories,
            transform_critical_into_major,
        )

        if severity_filtered or category_filtered or ill_formed or score_0:
            assert not processed_error

        if not processed_error:
            assert severity_filtered or category_filtered or ill_formed or score_0

        local_severity_filtered += int(severity_filtered)
        local_category_filtered += int(category_filtered)
        local_ill_formed += int(ill_formed)
        local_ill_formed_extended_span += int(ill_formed_extended_span)
        local_num_potential_span_matches += num_potential_span_matches
        local_num_potential_span_matches_with_extended_span += (
            num_potential_span_matches_with_extended_span
        )
        local_score_0 += int(score_0)

        if num_potential_span_matches > 1:
            num_errors_with_ambiguous_match += 1

        if num_potential_span_matches_with_extended_span > 1:
            num_errors_with_ambiguous_match_with_extended_span += 1

        if (
            not severity_filtered
            and not category_filtered
            and not ill_formed
            and not score_0
        ):
            new_errors.append(processed_error)

    non_overlapping_errors = remove_overlapping_errors_func(new_errors)
    num_overlapping_errors = len(new_errors) - len(non_overlapping_errors)
    if remove_overlapping_errors:
        new_errors = non_overlapping_errors

    new_automatic_evaluation = AutomaticEvaluation(
        score=sum(error.score for error in new_errors),
        errors=new_errors,
        annotation=automatic_evaluation.annotation,
        user_prompt=automatic_evaluation.user_prompt,
        system_prompt=automatic_evaluation.system_prompt,
        few_shots=automatic_evaluation.few_shots,
        cost=automatic_evaluation.cost,
        parsing_error=automatic_evaluation.parsing_error,
    )
    return (
        new_automatic_evaluation,
        local_severity_filtered,
        local_category_filtered,
        local_ill_formed,
        num_errors_with_ambiguous_match,
        num_errors_with_ambiguous_match_with_extended_span,
        local_ill_formed_extended_span,
        local_score_0,
        num_overlapping_errors,
    )
