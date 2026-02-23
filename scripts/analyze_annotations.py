# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Analyze annotations from LLM-based machine translation evaluation.

This script analyzes cached evaluation results to compute statistics about:
1. Evaluations impossible to parse (parsing_error = True)
2. Malformed errors (None values, empty fields)

Statistics are computed both overall and per-language pair.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter

from mt_evaluation.core import all_severities as all_severities

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mt_evaluation.core import AutomaticEvaluation, Sample, Error

logger = logging.getLogger(__name__)


def load_samples_from_cache(cache_file: Path) -> List[Sample]:
    """
    Load samples from a cache file.

    Args:
        cache_file: Path to the cache.jsonl file

    Returns:
        List of Sample objects with evaluations
    """
    samples = []

    if not cache_file.exists():
        logger.warning(f"Cache file does not exist: {cache_file}")
        return samples

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if data.get("evaluation"):
                        sample = Sample.from_dict(data)
                        samples.append(sample)
                    else:
                        logger.warning(
                            f"Skipping line {line_num}: missing 'evaluation' key"
                        )
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

    except Exception as e:
        logger.error(f"Error reading cache file {cache_file}: {e}")

    return samples


def is_malformed_error(error: Error) -> Tuple[bool, List[str]]:
    """
    Check if an error is malformed and return the reasons.

    Args:
        error: Error object to check

    Returns:
        Tuple of (is_malformed, list_of_reasons)
    """
    reasons = []

    # Check for None values
    if error.span is None:
        reasons.append("span is None")

    if error.category is None:
        reasons.append("category is None")

    if error.severity is None:
        reasons.append("severity is None")

    if error.score is None:
        reasons.append("score is None")

    # Check for empty fields
    if (
        error.span is not None
        and isinstance(error.span, str)
        and error.span.strip() == ""
    ):
        reasons.append("span is empty")

    if (
        error.category is not None
        and isinstance(error.category, str)
        and error.category.strip() == ""
    ):
        reasons.append("category is empty")

    if (
        error.severity is not None
        and isinstance(error.severity, str)
        and error.severity.strip() == ""
    ):
        reasons.append("severity is empty")

    if (
        error.severity is not None
        and isinstance(error.severity, str)
        and not any(severity in error.severity.lower() for severity in all_severities)
    ):
        reasons.append(f"severity is not allowed")

    return len(reasons) > 0, reasons


def analyze_evaluation(evaluation: AutomaticEvaluation) -> Dict[str, Any]:
    """
    Analyze a single evaluation for parsing errors and malformed errors.

    Args:
        evaluation: AutomaticEvaluation object

    Returns:
        Dictionary with analysis results
    """
    result = {
        "parsing_error": evaluation.parsing_error,
        "total_errors": len(evaluation.errors),
        "malformed_errors": 0,
        "malformed_error_reasons": Counter(),
        "malformed_error_details": [],
    }

    # Check each error for malformation
    for i, error in enumerate(evaluation.errors):
        is_malformed, reasons = is_malformed_error(error)
        if is_malformed:
            result["malformed_errors"] += 1
            result["malformed_error_reasons"].update(reasons)
            result["malformed_error_details"].append(
                {
                    "error_index": i,
                    "reasons": reasons,
                    "error_data": {
                        "span": error.span,
                        "category": error.category,
                        "severity": error.severity,
                        "score": error.score,
                    },
                }
            )

    return result


def escape_markdown_content(content: str) -> str:
    """
    Escape markdown content to prevent formatting conflicts.

    Args:
        content: Raw content that may contain markdown

    Returns:
        Escaped content safe for markdown code blocks
    """
    if not content:
        return content

    # Replace triple backticks with escaped version to prevent code block conflicts
    content = content.replace("```", "\\`\\`\\`")

    return content


def compute_statistics(
    samples: List[Sample],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute comprehensive statistics for all samples.

    Args:
        samples: List of Sample objects with evaluations

    Returns:
        Tuple of (statistics_dictionary, list_of_parsing_error_samples, list_of_disallowed_severity_samples)
    """
    stats = {
        "total_samples": len(samples),
        "samples_with_parsing_errors": 0,
        "samples_with_malformed_errors": 0,
        "total_errors": 0,
        "total_malformed_errors": 0,
        "malformed_error_reasons": Counter(),
        "per_language_pair": defaultdict(
            lambda: {
                "total_samples": 0,
                "samples_with_parsing_errors": 0,
                "samples_with_malformed_errors": 0,
                "total_errors": 0,
                "total_malformed_errors": 0,
                "malformed_error_reasons": Counter(),
            }
        ),
    }

    parsing_error_samples = []
    disallowed_severity_samples = []

    for sample_idx, sample in enumerate(samples):
        if sample.evaluation is None:
            logger.warning("Sample has no evaluation, skipping")
            continue

        # Language pair key
        lp = f"{sample.src_lang}-{sample.tgt_lang}"

        # Analyze this evaluation
        eval_analysis = analyze_evaluation(sample.evaluation)

        if eval_analysis["parsing_error"] is None:
            logger.warning(
                "Sample has no parsing error field. Will assume no parsing errors."
            )
            eval_analysis["parsing_error"] = False

        # Update overall statistics
        if eval_analysis["parsing_error"]:
            stats["samples_with_parsing_errors"] += 1
            stats["per_language_pair"][lp]["samples_with_parsing_errors"] += 1

            # Collect parsing error sample for logging
            parsing_error_samples.append(
                {
                    "sample_index": sample_idx,
                    "language_pair": lp,
                    "src": sample.src,
                    "tgt": sample.tgt,
                    "annotation": sample.evaluation.annotation,
                    "errors": [
                        {
                            "span": error.span,
                            "category": error.category,
                            "severity": error.severity,
                            "score": error.score,
                            "explanation": error.explanation,
                        }
                        for error in sample.evaluation.errors
                    ],
                    "score": sample.evaluation.score,
                    "user_prompt": (
                        sample.evaluation.user_prompt[:500] + "..."
                        if sample.evaluation.user_prompt
                        and len(sample.evaluation.user_prompt) > 500
                        else sample.evaluation.user_prompt
                    ),
                }
            )

        if eval_analysis["malformed_errors"] > 0:
            stats["samples_with_malformed_errors"] += 1
            stats["per_language_pair"][lp]["samples_with_malformed_errors"] += 1

        stats["total_errors"] += eval_analysis["total_errors"]
        stats["total_malformed_errors"] += eval_analysis["malformed_errors"]
        stats["malformed_error_reasons"].update(
            eval_analysis["malformed_error_reasons"]
        )

        # Update per-language-pair statistics
        stats["per_language_pair"][lp]["total_samples"] += 1
        stats["per_language_pair"][lp]["total_errors"] += eval_analysis["total_errors"]
        stats["per_language_pair"][lp]["total_malformed_errors"] += eval_analysis[
            "malformed_errors"
        ]
        stats["per_language_pair"][lp]["malformed_error_reasons"].update(
            eval_analysis["malformed_error_reasons"]
        )

        # Collect samples with disallowed severity errors
        for error_idx, error in enumerate(sample.evaluation.errors):
            if (
                error.severity is not None
                and isinstance(error.severity, str)
                and not any(
                    severity in error.severity.lower() for severity in all_severities
                )
            ):
                disallowed_severity_samples.append(
                    {
                        "sample_index": sample_idx,
                        "language_pair": lp,
                        "src": sample.src,
                        "tgt": sample.tgt,
                        "annotation": sample.evaluation.annotation,
                        "error_index": error_idx,
                        "disallowed_severity": error.severity,
                        "error_data": {
                            "span": error.span,
                            "category": error.category,
                            "severity": error.severity,
                            "score": error.score,
                            "explanation": error.explanation,
                        },
                        "all_errors": [
                            {
                                "span": err.span,
                                "category": err.category,
                                "severity": err.severity,
                                "score": err.score,
                                "explanation": err.explanation,
                            }
                            for err in sample.evaluation.errors
                        ],
                        "score": sample.evaluation.score,
                        "user_prompt": (
                            sample.evaluation.user_prompt[:500] + "..."
                            if sample.evaluation.user_prompt
                            and len(sample.evaluation.user_prompt) > 500
                            else sample.evaluation.user_prompt
                        ),
                    }
                )

    return stats, parsing_error_samples, disallowed_severity_samples


def write_well_formed_evaluations_log(
    samples: List[Sample],
    autoeval_name: str,
    model_name: str,
    run_info: str,
    min_samples_per_dimension: int = 3,
) -> Path:
    """
    Write a log of well-formed evaluations showing representative samples across different dimensions.

    Args:
        samples: All samples with evaluations
        autoeval_name: Name of the automatic evaluator
        model_name: Name of the model
        run_info: Run information
        min_samples_per_dimension: Minimum samples to log per dimension value

    Returns:
        Path to the created log file
    """
    # Filter to well-formed evaluations (no parsing errors, no malformed errors)
    well_formed_samples = []
    for sample in samples:
        if (
            sample.evaluation
            and not sample.evaluation.parsing_error
            and all(
                not is_malformed_error(error)[0] for error in sample.evaluation.errors
            )
        ):
            well_formed_samples.append(sample)

    if not well_formed_samples:
        return None

    # Collect samples by different dimensions
    samples_by_dimension = {
        "language_pair": defaultdict(list),
        "severity": defaultdict(list),
        "category": defaultdict(list),
        "score_range": defaultdict(list),
        "error_count": defaultdict(list),
    }

    for sample in well_formed_samples:
        lp = f"{sample.src_lang}-{sample.tgt_lang}"
        samples_by_dimension["language_pair"][lp].append(sample)

        # Score ranges
        score = sample.evaluation.score
        if score == 0:
            score_range = "0 (no errors)"
        elif score > -5:
            score_range = "-1 to -4 (minor issues)"
        elif score > -15:
            score_range = "-5 to -14 (moderate issues)"
        else:
            score_range = "-15 or lower (major issues)"
        samples_by_dimension["score_range"][score_range].append(sample)

        # Error count
        error_count = len(sample.evaluation.errors)
        if error_count == 0:
            error_count_range = "0 errors"
        elif error_count <= 2:
            error_count_range = "1-2 errors"
        elif error_count <= 5:
            error_count_range = "3-5 errors"
        else:
            error_count_range = "6+ errors"
        samples_by_dimension["error_count"][error_count_range].append(sample)

        # Collect by error severity and category
        for error in sample.evaluation.errors:
            if error.severity:
                samples_by_dimension["severity"][error.severity].append(sample)
            if error.category:
                samples_by_dimension["category"][error.category].append(sample)

    # Select representative samples
    selected_samples = []
    sample_sources = []  # Track where each sample came from

    for dimension, dimension_samples in samples_by_dimension.items():
        for value, value_samples in dimension_samples.items():
            # Take up to min_samples_per_dimension samples for each dimension value
            selected = value_samples[:min_samples_per_dimension]
            for sample in selected:
                if sample not in selected_samples:
                    selected_samples.append(sample)
                    sample_sources.append(f"{dimension}={value}")

    # Create log path
    log_filename = "well_formed_evaluations_log.md"
    log_path = Path("analysis") / autoeval_name / model_name / run_info / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Well-Formed Evaluations Log\n\n")
        f.write(f"**Evaluator:** {autoeval_name}\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Run Info:** {run_info}\n")
        f.write(f"**Total Well-Formed Evaluations:** {len(well_formed_samples)}\n")
        f.write(f"**Representative Samples Logged:** {len(selected_samples)}\n\n")

        # Show breakdown by dimensions
        f.write("**Sample Distribution by Dimensions:**\n\n")

        for dimension, dimension_samples in samples_by_dimension.items():
            f.write(f"**{dimension.replace('_', ' ').title()}:**\n")
            for value, value_samples in sorted(dimension_samples.items()):
                logged_count = min(len(value_samples), min_samples_per_dimension)
                f.write(
                    f"- {value}: {len(value_samples)} total, {logged_count} logged\n"
                )
            f.write("\n")

        f.write("---\n\n")

        for i, sample in enumerate(selected_samples, 1):
            f.write(f"## Sample {i}\n\n")
            f.write(f"**Language Pair:** {sample.src_lang}-{sample.tgt_lang}\n")
            f.write(f"**Selected for:** {sample_sources[i-1]}\n\n")

            f.write(f"**Source Text:**\n")
            f.write(f"```\n{escape_markdown_content(sample.src)}\n```\n\n")

            f.write(f"**Target Text:**\n")
            f.write(f"```\n{escape_markdown_content(sample.tgt)}\n```\n\n")

            f.write(f"**Raw Annotation:**\n")
            f.write(
                f"```\n{escape_markdown_content(sample.evaluation.annotation)}\n```\n\n"
            )

            f.write(f"**Parsed Score:** {sample.evaluation.score}\n\n")

            if sample.evaluation.errors:
                f.write(f"**Parsed Errors ({len(sample.evaluation.errors)}):**\n")
                for j, error in enumerate(sample.evaluation.errors, 1):
                    escaped_span = (
                        escape_markdown_content(str(error.span))
                        if error.span
                        else error.span
                    )
                    escaped_explanation = (
                        escape_markdown_content(str(error.explanation))
                        if error.explanation
                        else error.explanation
                    )
                    f.write(
                        f"{j}. **Span:** `{escaped_span}` | **Category:** `{error.category}` | **Severity:** `{error.severity}` | **Score:** `{error.score}`\n"
                    )
                    if escaped_explanation:
                        f.write(f"   - **Explanation:** {escaped_explanation}\n")
                f.write("\n")
            else:
                f.write("**Parsed Errors:** None\n\n")

            f.write("---\n\n")

    return log_path


def write_disallowed_severity_errors_log(
    disallowed_severity_samples: List[Dict[str, Any]],
    autoeval_name: str,
    model_name: str,
    run_info: str,
) -> Path:
    """
    Write detailed disallowed severity errors to a log file for visual inspection.

    Args:
        disallowed_severity_samples: List of samples with disallowed severity errors
        autoeval_name: Name of the automatic evaluator
        model_name: Name of the model
        run_info: Run information

    Returns:
        Path to the created log file
    """
    if not disallowed_severity_samples:
        return None

    # Group samples by language pair and limit to 50 per language pair
    samples_by_lp = defaultdict(list)
    for sample in disallowed_severity_samples:
        samples_by_lp[sample["language_pair"]].append(sample)

    # Limit to 10 samples per language pair
    limited_samples = []
    for lp, samples in samples_by_lp.items():
        limited_samples.extend(samples[:10])

    # Create log path with same structure as outputs: analysis/<autoeval>/<model>/<run-info>/
    log_filename = "disallowed_severity_errors_log.md"
    log_path = Path("analysis") / autoeval_name / model_name / run_info / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get unique disallowed severities for summary
    unique_severities = set(
        sample["disallowed_severity"] for sample in disallowed_severity_samples
    )
    allowed_severities_str = ", ".join(all_severities)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Disallowed Severity Errors Log\n\n")
        f.write(f"**Evaluator:** {autoeval_name}\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Run Info:** {run_info}\n")
        f.write(f"**Allowed Severities:** {allowed_severities_str}\n")
        f.write(
            f"**Found Disallowed Severities:** {', '.join(sorted(unique_severities))}\n"
        )
        f.write(
            f"**Total Disallowed Severity Errors:** {len(disallowed_severity_samples)}\n"
        )
        f.write(
            f"**Logged Samples:** {len(limited_samples)} (max 10 per language pair)\n\n"
        )

        # Show breakdown by language pair
        f.write("**Breakdown by Language Pair:**\n")
        for lp, samples in samples_by_lp.items():
            logged_count = min(len(samples), 10)
            f.write(f"- {lp}: {len(samples)} total, {logged_count} logged\n")
        f.write("\n")

        # Show breakdown by disallowed severity
        severity_counts = Counter(
            sample["disallowed_severity"] for sample in disallowed_severity_samples
        )
        f.write("**Breakdown by Disallowed Severity:**\n")
        for severity, count in severity_counts.most_common():
            f.write(f"- `{severity}`: {count} errors\n")
        f.write("\n---\n\n")

        for i, sample in enumerate(limited_samples, 1):
            f.write(f"## Sample {i} (Index: {sample['sample_index']})\n\n")
            f.write(f"**Language Pair:** {sample['language_pair']}\n")
            f.write(f"**Disallowed Severity:** `{sample['disallowed_severity']}`\n")
            f.write(f"**Error Index:** {sample['error_index']}\n\n")

            f.write(f"**Source Text:**\n")
            f.write(f"```\n{escape_markdown_content(sample['src'])}\n```\n\n")

            f.write(f"**Target Text:**\n")
            f.write(f"```\n{escape_markdown_content(sample['tgt'])}\n```\n\n")

            f.write(f"**Raw Annotation:**\n")
            f.write(f"```\n{escape_markdown_content(sample['annotation'])}\n```\n\n")

            f.write(f"**Parsed Score:** {sample['score']}\n\n")

            # Highlight the specific error with disallowed severity
            error_data = sample["error_data"]
            f.write(f"**Error with Disallowed Severity:**\n")
            escaped_span = (
                escape_markdown_content(str(error_data["span"]))
                if error_data["span"]
                else error_data["span"]
            )
            escaped_explanation = (
                escape_markdown_content(str(error_data["explanation"]))
                if error_data["explanation"]
                else error_data["explanation"]
            )
            f.write(
                f"- **Span:** `{escaped_span}` | **Category:** `{error_data['category']}` | **Severity:** `{error_data['severity']}` | **Score:** `{error_data['score']}`\n"
            )
            if escaped_explanation:
                f.write(f"  - **Explanation:** {escaped_explanation}\n")
            f.write("\n")

            # Show all errors for context
            if sample["all_errors"]:
                f.write(
                    f"**All Parsed Errors ({len(sample['all_errors'])}) for Context:**\n"
                )
                for j, error in enumerate(sample["all_errors"], 1):
                    escaped_span = (
                        escape_markdown_content(str(error["span"]))
                        if error["span"]
                        else error["span"]
                    )
                    escaped_explanation = (
                        escape_markdown_content(str(error["explanation"]))
                        if error["explanation"]
                        else error["explanation"]
                    )
                    severity_marker = " ⚠️" if j - 1 == sample["error_index"] else ""
                    f.write(
                        f"{j}. **Span:** `{escaped_span}` | **Category:** `{error['category']}` | **Severity:** `{error['severity']}` | **Score:** `{error['score']}`{severity_marker}\n"
                    )
                    if escaped_explanation:
                        f.write(f"   - **Explanation:** {escaped_explanation}\n")
                f.write("\n")

            f.write("---\n\n")

    return log_path


def write_parsing_errors_log(
    parsing_error_samples: List[Dict[str, Any]],
    autoeval_name: str,
    model_name: str,
    run_info: str,
) -> Path:
    """
    Write detailed parsing errors to a log file for visual inspection.

    Args:
        parsing_error_samples: List of samples with parsing errors
        autoeval_name: Name of the automatic evaluator
        model_name: Name of the model
        run_info: Run information

    Returns:
        Path to the created log file
    """
    if not parsing_error_samples:
        return None

    # Group samples by language pair and limit to 10 per language pair
    samples_by_lp = defaultdict(list)
    for sample in parsing_error_samples:
        samples_by_lp[sample["language_pair"]].append(sample)

    # Limit to 10 samples per language pair
    limited_samples = []
    for lp, samples in samples_by_lp.items():
        limited_samples.extend(samples[:10])

    # Create log path with same structure as outputs: analysis/<autoeval>/<model>/<run-info>/
    log_filename = "parsing_errors_log.md"
    log_path = Path("analysis") / autoeval_name / model_name / run_info / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Parsing Errors Log\n\n")
        f.write(f"**Evaluator:** {autoeval_name}\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Run Info:** {run_info}\n")
        f.write(f"**Total Parsing Errors:** {len(parsing_error_samples)}\n")
        f.write(
            f"**Logged Samples:** {len(limited_samples)} (max 10 per language pair)\n\n"
        )

        # Show breakdown by language pair
        f.write("**Breakdown by Language Pair:**\n")
        for lp, samples in samples_by_lp.items():
            logged_count = min(len(samples), 10)
            f.write(f"- {lp}: {len(samples)} total, {logged_count} logged\n")
        f.write("\n---\n\n")

        for i, sample in enumerate(limited_samples, 1):
            f.write(f"## Sample {i} (Index: {sample['sample_index']})\n\n")
            f.write(f"**Language Pair:** {sample['language_pair']}\n\n")

            f.write(f"**Source Text:**\n")
            f.write(f"```\n{escape_markdown_content(sample['src'])}\n```\n\n")

            f.write(f"**Target Text:**\n")
            f.write(f"```\n{escape_markdown_content(sample['tgt'])}\n```\n\n")

            f.write(f"**Raw Annotation:**\n")
            f.write(f"```\n{escape_markdown_content(sample['annotation'])}\n```\n\n")

            f.write(f"**Parsed Score:** {sample['score']}\n\n")

            if sample["errors"]:
                f.write(f"**Parsed Errors ({len(sample['errors'])}):**\n")
                for j, error in enumerate(sample["errors"], 1):
                    escaped_span = (
                        escape_markdown_content(str(error["span"]))
                        if error["span"]
                        else error["span"]
                    )
                    escaped_explanation = (
                        escape_markdown_content(str(error["explanation"]))
                        if error["explanation"]
                        else error["explanation"]
                    )
                    f.write(
                        f"{j}. **Span:** `{escaped_span}` | **Category:** `{error['category']}` | **Severity:** `{error['severity']}` | **Score:** `{error['score']}`\n"
                    )
                    if escaped_explanation:
                        f.write(f"   - **Explanation:** {escaped_explanation}\n")
                f.write("\n")
            else:
                f.write("**Parsed Errors:** None\n\n")

            f.write("---\n\n")

    return log_path


def print_statistics(stats: Dict[str, Any], autoeval_name: str, model_name: str):
    """
    Print formatted statistics.

    Args:
        stats: Statistics dictionary
        autoeval_name: Name of the automatic evaluator
        model_name: Name of the model
    """
    print(f"\n{'='*80}")
    print(f"ANNOTATION ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"Evaluator: {autoeval_name}")
    print(f"Model: {model_name}")
    print(f"{'='*80}")

    # Overall statistics
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total samples: {stats['total_samples']}")
    print(
        f"  Samples with parsing errors: {stats['samples_with_parsing_errors']} ({stats['samples_with_parsing_errors']/stats['total_samples']*100:.2f}%)"
    )
    print(
        f"  Samples with malformed errors: {stats['samples_with_malformed_errors']} ({stats['samples_with_malformed_errors']/stats['total_samples']*100:.2f}%)"
    )
    print(f"  Total errors: {stats['total_errors']}")
    if stats["total_errors"] > 0:
        print(
            f"  Total malformed errors: {stats['total_malformed_errors']} ({stats['total_malformed_errors']/stats['total_errors']*100:.2f}% of all errors)"
        )
    else:
        print(f"  Total malformed errors: {stats['total_malformed_errors']}")

    # Malformed error reasons
    if stats["malformed_error_reasons"]:
        print(f"\n  Malformed error breakdown:")
        for reason, count in stats["malformed_error_reasons"].most_common():
            if reason == "severity is not allowed":
                print(
                    f"    {reason}: {count} (see disallowed_severity_errors_log.md for examples)"
                )
            else:
                print(f"    {reason}: {count}")

    # Per-language-pair statistics
    print(f"\nPER-LANGUAGE-PAIR STATISTICS:")
    for lp, lp_stats in sorted(stats["per_language_pair"].items()):
        print(f"\n  {lp}:")
        print(f"    Total samples: {lp_stats['total_samples']}")
        print(
            f"    Samples with parsing errors: {lp_stats['samples_with_parsing_errors']} ({lp_stats['samples_with_parsing_errors']/lp_stats['total_samples']*100:.2f}%)"
        )
        print(
            f"    Samples with malformed errors: {lp_stats['samples_with_malformed_errors']} ({lp_stats['samples_with_malformed_errors']/lp_stats['total_samples']*100:.2f}%)"
        )
        print(f"    Total errors: {lp_stats['total_errors']}")
        if lp_stats["total_errors"] > 0:
            print(
                f"    Total malformed errors: {lp_stats['total_malformed_errors']} ({lp_stats['total_malformed_errors']/lp_stats['total_errors']*100:.2f}% of errors)"
            )
        else:
            print(f"    Total malformed errors: {lp_stats['total_malformed_errors']}")

        if lp_stats["malformed_error_reasons"]:
            print(f"    Malformed error breakdown:")
            for reason, count in lp_stats["malformed_error_reasons"].most_common():
                print(f"      {reason}: {count}")


def find_cache_files(outputs_dir: Path) -> List[Tuple[str, str, str, Path]]:
    """
    Find all cache files in the outputs directory.

    Args:
        outputs_dir: Path to outputs directory

    Returns:
        List of tuples (autoeval_name, model_name, run_info, cache_file_path)
    """
    cache_files = []

    if not outputs_dir.exists():
        logger.error(f"Outputs directory does not exist: {outputs_dir}")
        return cache_files

    # Structure: outputs/autoeval_name/model_name/run_info/cache.jsonl
    for autoeval_dir in outputs_dir.iterdir():
        if not autoeval_dir.is_dir():
            continue

        autoeval_name = autoeval_dir.name

        for model_dir in autoeval_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name

            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                run_info = run_dir.name
                cache_file = run_dir / "cache.jsonl"

                if cache_file.exists():
                    cache_files.append(
                        (autoeval_name, model_name, run_info, cache_file)
                    )

    return cache_files


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze LLM-based MT evaluation annotations"
    )

    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory containing cached evaluation results (default: outputs)",
    )

    parser.add_argument(
        "--autoeval-name",
        type=str,
        help="Specific autoeval to analyze (if not provided, analyzes all)",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        help="Specific model to analyze (if not provided, analyzes all)",
    )

    parser.add_argument(
        "--run-info",
        type=str,
        help="Specific run info to analyze (if not provided, analyzes all)",
    )

    parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.logging_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    outputs_dir = Path(args.outputs_dir)

    # Find cache files
    cache_files = find_cache_files(outputs_dir)

    if not cache_files:
        logger.error("No cache files found")
        return

    # Filter cache files based on arguments
    if args.autoeval_name:
        cache_files = [
            (ae, m, r, cf) for ae, m, r, cf in cache_files if ae == args.autoeval_name
        ]

    if args.model_name:
        cache_files = [
            (ae, m, r, cf) for ae, m, r, cf in cache_files if m == args.model_name
        ]

    if args.run_info:
        cache_files = [
            (ae, m, r, cf) for ae, m, r, cf in cache_files if r == args.run_info
        ]

    if not cache_files:
        logger.error("No cache files match the specified criteria")
        return

    logger.info(f"Found {len(cache_files)} cache files to analyze")

    # Analyze each cache file
    for autoeval_name, model_name, run_info, cache_file in cache_files:
        logger.info(f"Analyzing: {autoeval_name}/{model_name}/{run_info}")

        # Load samples
        samples = load_samples_from_cache(cache_file)

        if not samples:
            logger.warning(f"No samples found in {cache_file}")
            continue

        logger.info(f"Loaded {len(samples)} samples")

        # Compute statistics
        stats, parsing_error_samples, disallowed_severity_samples = compute_statistics(
            samples
        )

        # Print results
        print_statistics(stats, autoeval_name, f"{model_name}/{run_info}")

        # Write parsing errors log if there are any
        if parsing_error_samples:
            log_path = write_parsing_errors_log(
                parsing_error_samples, autoeval_name, model_name, run_info
            )
            if log_path:
                logger.info(f"Parsing errors logged to: {log_path}")
                print(f"\n📝 Parsing errors logged to: {log_path}")

        # Write disallowed severity errors log if there are any
        if disallowed_severity_samples:
            log_path = write_disallowed_severity_errors_log(
                disallowed_severity_samples, autoeval_name, model_name, run_info
            )
            if log_path:
                logger.info(f"Disallowed severity errors logged to: {log_path}")
                print(f"📝 Disallowed severity errors logged to: {log_path}")

        # Write well-formed evaluations log
        well_formed_log_path = write_well_formed_evaluations_log(
            samples, autoeval_name, model_name, run_info
        )
        if well_formed_log_path:
            logger.info(f"Well-formed evaluations logged to: {well_formed_log_path}")
            print(f"📝 Well-formed evaluations logged to: {well_formed_log_path}")


if __name__ == "__main__":
    main()
