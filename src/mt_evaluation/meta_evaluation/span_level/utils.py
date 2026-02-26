# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Span-level meta-evaluation utilities.

This module provides functions for computing span-level precision, recall, and F1
between automatic evaluations and human (gold) annotations.

The main high-level functions compute aggregated metrics across all samples.
Lower-level functions are in:
- standardization: Text and error standardization
- preprocessing: Sample preprocessing
- matching: Error bipartite matching algorithms
"""

from typing import List, Tuple, Union, Dict
import logging
import numpy as np
from collections import defaultdict

from mt_evaluation.utils import (
    convert_defaultdict_to_dict,
    setup_logging,
    get_metric_display_name,
)
from mt_evaluation.core import AutomaticEvaluation, Error, HumanEvaluation, Sample
from mt_evaluation.meta_evaluation import all_severities, METRIC_TYPES, MetricResults

from mt_evaluation.meta_evaluation.span_level.metrics import (
    Metrics,
    MicroMetrics,
    MacroMetrics,
    compute_p_r_f1_from_tp_fp_fn,
    span_f_score_partial,
    span_f_score_exact_match,
    span_f_score_proportion_chars,
    span_tp_char_counts,
)
from mt_evaluation.meta_evaluation.span_level.standardization import (
    standardize_automatic_evaluation,
)
from mt_evaluation.meta_evaluation.span_level.preprocessing import (
    errors_overlap,
    select_more_severe_error,
)
from mt_evaluation.meta_evaluation.span_level.matching import (
    compute_overlap_length,
    find_greedy_bipartite_matching,
    find_optimal_bipartite_matching,
    MatchInfo,
)

logger = logging.getLogger(__name__)

# Severity to index mapping for array operations
severity_to_idx = {sev: i for i, sev in enumerate(all_severities)}


def _get_match_info_as_tuple(match_info: MatchInfo) -> Tuple[float, int, int]:
    """Convert MatchInfo to tuple format for backward compatibility."""
    return (match_info.avg_overlap, match_info.overlap_length, match_info.matched_index)


def _get_match_reward(
    match_dict: Dict[int, MatchInfo],
    error_idx: int,
    error: Error,
    other_errors: list,
    p: float,
) -> Union[Tuple[float, int, Error], None]:
    """Look up a match and compute the severity-adjusted binary reward.

    Args:
        match_dict: Mapping from error index to MatchInfo.
        error_idx: Index of the current error in its list.
        error: The current error object.
        other_errors: The list of errors on the other side (human or auto).
        p: Severity penalty factor.

    Returns:
        (binary_reward, overlap_length, matched_error) if matched with
        positive overlap, or None if unmatched.
    """
    match_info = match_dict.get(error_idx)
    if not match_info or match_info.avg_overlap <= 0:
        return None
    matched_error = other_errors[match_info.matched_index]
    severity_matches = error.severity == matched_error.severity
    binary_reward = 1.0 if severity_matches else (1.0 - p)
    return binary_reward, match_info.overlap_length, matched_error


def process_single_autoeval_wrapper(args_tuple):
    """Wrapper for multiprocessing - unpacks arguments and calls process_single_autoeval"""
    (
        autoeval,
        lps,
        lp2preprocessed_samples_with_automatic_evaluations,
        lp2preprocessed_samples_with_human_evaluations,
        severity_penalty,
        remove_overlapping_errors,
        fix_edge_cases_in_precision,
        use_greedy_matching,
        logging_level,
    ) = args_tuple

    setup_logging(logging_level)

    autoeval_metrics = process_single_autoeval(
        lps=lps,
        lp2preprocessed_samples_with_automatic_evaluations=lp2preprocessed_samples_with_automatic_evaluations,
        lp2preprocessed_samples_with_human_evaluations=lp2preprocessed_samples_with_human_evaluations,
        severity_penalty=severity_penalty,
        remove_overlapping_errors=remove_overlapping_errors,
        fix_edge_cases_in_precision=fix_edge_cases_in_precision,
        use_greedy_matching=use_greedy_matching,
    )

    return autoeval, autoeval_metrics


def process_single_autoeval(
    lps: List[str],
    lp2preprocessed_samples_with_automatic_evaluations: Dict[str, List[Sample]],
    lp2preprocessed_samples_with_human_evaluations: Dict[str, List[Sample]],
    severity_penalty: float,
    remove_overlapping_errors: bool = False,
    fix_edge_cases_in_precision: bool = False,
    use_greedy_matching: bool = False,
) -> Dict[str, Dict[str, Dict[str, Dict[str, MicroMetrics | MacroMetrics]]]]:
    autoeval_metrics = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(None)))
    )

    for lp in lps:
        results = dict()
        results["matching"] = compute_all_tp_fp_fn(
            lp2preprocessed_samples_with_automatic_evaluations[lp],
            lp2preprocessed_samples_with_human_evaluations[lp],
            do_error_matching=True,
            severity_penalty=severity_penalty,
            has_overlapping_errors=not remove_overlapping_errors,
            fix_edge_cases_in_precision=fix_edge_cases_in_precision,
            use_greedy_matching=use_greedy_matching,
        )

        results["not_matching"] = compute_all_tp_fp_fn(
            lp2preprocessed_samples_with_automatic_evaluations[lp],
            lp2preprocessed_samples_with_human_evaluations[lp],
            do_error_matching=False,
            severity_penalty=severity_penalty,
            has_overlapping_errors=not remove_overlapping_errors,
            fix_edge_cases_in_precision=fix_edge_cases_in_precision,
            use_greedy_matching=False,
        )

        for aggr_type in ["micro", "macro"]:
            for matching_type in ["matching", "not_matching"]:
                for tp_type in METRIC_TYPES:
                    autoeval_metrics[lp][aggr_type][matching_type][tp_type] = results[
                        matching_type
                    ][aggr_type][tp_type]

    autoeval_metrics = convert_defaultdict_to_dict(autoeval_metrics)
    return autoeval_metrics


def merge_autoevaluators(
    lp2sys2samples_with_automatic_evaluations_list: List[
        Dict[str, Dict[str, List[Sample]]]
    ],
    strategy: str,
) -> Dict[str, Dict[str, List[Sample]]]:
    """Merge multiple autoevaluators using the specified strategy."""
    merged_lp2sys2samples = {}

    num_autoevaluators = len(lp2sys2samples_with_automatic_evaluations_list)
    lps = list(lp2sys2samples_with_automatic_evaluations_list[0].keys())

    for lp in lps:
        merged_lp2sys2samples[lp] = {}
        mt_systems = list(lp2sys2samples_with_automatic_evaluations_list[0][lp].keys())
        for sys in mt_systems:
            merged_lp2sys2samples[lp][sys] = []
            num_samples = len(
                lp2sys2samples_with_automatic_evaluations_list[0][lp][sys]
            )
            for sample_idx in range(num_samples):
                autoevaluator_samples = [
                    lp2sys2samples_with_automatic_evaluations_list[idx][lp][sys][
                        sample_idx
                    ]
                    for idx in range(num_autoevaluators)
                ]

                evaluations = [sample.evaluation for sample in autoevaluator_samples]
                merged_evaluation = merge_automatic_evaluations(
                    evaluations, autoevaluator_samples[0], strategy=strategy
                )

                merged_sample = Sample(
                    src=autoevaluator_samples[0].src,
                    tgt=autoevaluator_samples[0].tgt,
                    src_lang=autoevaluator_samples[0].src_lang,
                    tgt_lang=autoevaluator_samples[0].tgt_lang,
                    evaluation=merged_evaluation,
                )

                merged_lp2sys2samples[lp][sys].append(merged_sample)

    return merged_lp2sys2samples


def merge_automatic_evaluations(
    evaluations: List[AutomaticEvaluation],
    sample: Sample,
    strategy: str,
) -> AutomaticEvaluation:
    """Merge automatic evaluations using the specified strategy."""
    if strategy == "union":
        return merge_automatic_evaluations_by_union(evaluations, sample)
    elif strategy == "union-with-overlap":
        return merge_automatic_evaluations_by_union_with_overlap(evaluations, sample)
    else:
        raise NotImplementedError(
            f"strategy={strategy}, but must be in {{'union', 'union-with-overlap'}}"
        )


def merge_automatic_evaluations_by_union_with_overlap(
    evaluations: List[AutomaticEvaluation],
    sample: Sample,
) -> AutomaticEvaluation:
    """Merge automatic evaluations by union, allowing overlapping errors."""
    evaluations = [
        standardize_automatic_evaluation(evaluation, sample, all_severities, "All")[0]
        for evaluation in evaluations
    ]

    merged_annotations, merged_user_prompts, merged_system_prompts, total_cost = (
        "",
        "",
        "",
        0,
    )
    for idx, evaluation in enumerate(evaluations):
        merged_annotations += f"###Annotation-{idx}### - " + evaluation.annotation
        merged_user_prompts += f"###User Prompt-{idx}### - " + evaluation.user_prompt
        merged_system_prompts += (
            f"###System Prompt-{idx}### - " + evaluation.system_prompt
        )
        total_cost += evaluation.cost

    parsing_error = all(evaluation.parsing_error for evaluation in evaluations)
    if parsing_error:
        return AutomaticEvaluation(
            score=0.0,
            errors=[],
            annotation=merged_annotations,
            parsing_error=True,
            user_prompt=merged_user_prompts,
            system_prompt=merged_system_prompts,
            cost=total_cost,
        )

    all_errors = []
    for evaluation in evaluations:
        all_errors.extend(evaluation.errors)

    return AutomaticEvaluation(
        score=sum(error.score for error in all_errors),
        errors=all_errors,
        annotation=merged_annotations,
        parsing_error=False,
        user_prompt=merged_user_prompts,
        system_prompt=merged_system_prompts,
        cost=total_cost,
    )


def merge_automatic_evaluations_by_union(
    evaluations: List[AutomaticEvaluation],
    sample: Sample,
) -> AutomaticEvaluation:
    """Merge automatic evaluations by union, resolving overlaps by severity."""
    evaluations = [
        standardize_automatic_evaluation(evaluation, sample, all_severities, "All")[0]
        for evaluation in evaluations
    ]

    merged_annotations, merged_user_prompts, merged_system_prompts, total_cost = (
        "",
        "",
        "",
        0,
    )
    for idx, evaluation in enumerate(evaluations):
        merged_annotations += f"###Annotation-{idx}### - " + evaluation.annotation
        merged_user_prompts += f"###User Prompt-{idx}### - " + evaluation.user_prompt
        merged_system_prompts += (
            f"###System Prompt-{idx}### - " + evaluation.system_prompt
        )
        total_cost += evaluation.cost

    parsing_error = all(evaluation.parsing_error for evaluation in evaluations)
    if parsing_error:
        return AutomaticEvaluation(
            score=0.0,
            errors=[],
            annotation=merged_annotations,
            parsing_error=True,
            user_prompt=merged_user_prompts,
            system_prompt=merged_system_prompts,
            cost=total_cost,
        )

    all_errors = []
    for evaluation in evaluations:
        all_errors.extend(evaluation.errors)

    all_errors.sort(key=lambda e: (e.start, e.end))

    merged = []
    for current_error in all_errors:
        last_merged = merged[-1] if merged else None
        if not last_merged:
            merged.append(current_error)
            continue

        if not errors_overlap(current_error, last_merged):
            merged.append(current_error)
            continue

        winner = select_more_severe_error(current_error, last_merged)
        if winner != last_merged:
            merged[-1] = current_error

    return AutomaticEvaluation(
        score=sum(error.score for error in merged),
        errors=merged,
        annotation=merged_annotations,
        parsing_error=False,
        user_prompt=merged_user_prompts,
        system_prompt=merged_system_prompts,
        cost=total_cost,
    )


def decompose_counts(auto_counts: np.ndarray, human_counts: np.ndarray):
    """Decompose character counts into exact matches, mismatches, and unmatched."""
    num_sev, length = auto_counts.shape

    auto_exact = np.zeros((num_sev, length), dtype=float)
    auto_mismatch = np.zeros((num_sev, length), dtype=float)
    auto_unmatched = np.zeros((num_sev, length), dtype=float)
    human_exact = np.zeros((num_sev, length), dtype=float)
    human_mismatch = np.zeros((num_sev, length), dtype=float)
    human_unmatched = np.zeros((num_sev, length), dtype=float)

    for i in range(length):
        a = auto_counts[:, i].astype(float)
        h = human_counts[:, i].astype(float)

        if not (a.any() or h.any()):
            continue

        exact = np.minimum(a, h)
        auto_exact[:, i] = exact
        human_exact[:, i] = exact

        auto_rem = a - exact
        human_rem = h - exact

        a_rem_total = float(auto_rem.sum())
        h_rem_total = float(human_rem.sum())
        mismatch_pairs = min(a_rem_total, h_rem_total)

        if mismatch_pairs > 0.0:
            auto_mismatch[:, i] = auto_rem * (mismatch_pairs / a_rem_total)
            human_mismatch[:, i] = human_rem * (mismatch_pairs / h_rem_total)

        auto_unmatched[:, i] = np.maximum(0.0, auto_rem - auto_mismatch[:, i])
        human_unmatched[:, i] = np.maximum(0.0, human_rem - human_mismatch[:, i])

    return (
        auto_exact,
        auto_mismatch,
        auto_unmatched,
        human_exact,
        human_mismatch,
        human_unmatched,
    )


def compute_tp_fp_fn_without_error_matching(
    automatic_eval: AutomaticEvaluation,
    human_eval: HumanEvaluation,
    src: str,
    tgt: str,
    severity_penalty: float,
) -> Tuple[float, ...]:
    """Compute TP, FP, FN metrics without 1-to-1 error matching (handles overlaps)."""
    p = float(severity_penalty)
    num_sev = len(all_severities)

    auto_src_counts = np.zeros((num_sev, len(src)), dtype=np.int32)
    auto_tgt_counts = np.zeros((num_sev, len(tgt)), dtype=np.int32)
    human_src_counts = np.zeros((num_sev, len(src)), dtype=np.int32)
    human_tgt_counts = np.zeros((num_sev, len(tgt)), dtype=np.int32)

    for auto_error in automatic_eval.errors:
        sev_idx = severity_to_idx[auto_error.severity]
        if auto_error.is_source_error:
            auto_src_counts[sev_idx, auto_error.start : auto_error.end] += 1
        else:
            auto_tgt_counts[sev_idx, auto_error.start : auto_error.end] += 1

    for human_error in human_eval.errors:
        sev_idx = severity_to_idx[human_error.severity]
        if human_error.is_source_error:
            human_src_counts[sev_idx, human_error.start : human_error.end] += 1
        else:
            human_tgt_counts[sev_idx, human_error.start : human_error.end] += 1

    (
        auto_exact_src,
        auto_mismatch_src,
        auto_unmatched_src,
        human_exact_src,
        human_mismatch_src,
        human_unmatched_src,
    ) = decompose_counts(auto_src_counts, human_src_counts)

    (
        auto_exact_tgt,
        auto_mismatch_tgt,
        auto_unmatched_tgt,
        human_exact_tgt,
        human_mismatch_tgt,
        human_unmatched_tgt,
    ) = decompose_counts(auto_tgt_counts, human_tgt_counts)

    tpc = float(
        auto_exact_src.sum()
        + (1.0 - p) * auto_mismatch_src.sum()
        + auto_exact_tgt.sum()
        + (1.0 - p) * auto_mismatch_tgt.sum()
    )
    fpc = float(
        (p * auto_mismatch_src + auto_unmatched_src).sum()
        + (p * auto_mismatch_tgt + auto_unmatched_tgt).sum()
    )
    fnc = float(
        (p * human_mismatch_src + human_unmatched_src).sum()
        + (p * human_mismatch_tgt + human_unmatched_tgt).sum()
    )

    tp_for_precision, tp_for_recall, fp, fn = 0.0, 0.0, 0.0, 0.0
    tppc_for_precision, tppc_for_recall, fppc, fnpc = 0.0, 0.0, 0.0, 0.0

    human_src_any = human_src_counts.sum(axis=0) > 0
    human_tgt_any = human_tgt_counts.sum(axis=0) > 0
    auto_src_any = auto_src_counts.sum(axis=0) > 0
    auto_tgt_any = auto_tgt_counts.sum(axis=0) > 0

    for auto_error in automatic_eval.errors:
        sev_idx = severity_to_idx[auto_error.severity]
        start, end = auto_error.start, auto_error.end
        span_len = end - start

        if auto_error.is_source_error:
            human_any = human_src_any
            auto_counts_side = auto_src_counts
            auto_exact_side = auto_exact_src
            auto_mismatch_side = auto_mismatch_src
            auto_unmatched_side = auto_unmatched_src
            human_same_sev = human_src_counts[sev_idx] > 0
        else:
            human_any = human_tgt_any
            auto_counts_side = auto_tgt_counts
            auto_exact_side = auto_exact_tgt
            auto_mismatch_side = auto_mismatch_tgt
            auto_unmatched_side = auto_unmatched_tgt
            human_same_sev = human_tgt_counts[sev_idx] > 0

        error_mask = np.zeros(
            len(src) if auto_error.is_source_error else len(tgt), dtype=bool
        )
        error_mask[start:end] = True

        overlap_any = int(np.sum(error_mask & human_any))
        if overlap_any > 0:
            overlap_same = int(np.sum(error_mask & human_same_sev))
            binary_reward = 1.0 if overlap_same > 0 else (1.0 - p)
            tp_for_precision += binary_reward
            fp += 1.0 - binary_reward
        else:
            fp += 1.0

        span_tp_char_sum, span_fp_char_sum = 0.0, 0.0
        for i in range(start, end):
            a_s = float(auto_counts_side[sev_idx, i])
            if a_s > 0:
                span_tp_char_sum += (
                    auto_exact_side[sev_idx, i]
                    + (1.0 - p) * auto_mismatch_side[sev_idx, i]
                ) / a_s
                span_fp_char_sum += (
                    p * auto_mismatch_side[sev_idx, i] + auto_unmatched_side[sev_idx, i]
                ) / a_s

        tppc_for_precision += span_tp_char_sum / span_len
        fppc += span_fp_char_sum / span_len

    for human_error in human_eval.errors:
        sev_idx = severity_to_idx[human_error.severity]
        start, end = human_error.start, human_error.end
        span_len = end - start

        if human_error.is_source_error:
            auto_any = auto_src_any
            human_counts_side = human_src_counts
            human_exact_side = human_exact_src
            human_mismatch_side = human_mismatch_src
            human_unmatched_side = human_unmatched_src
            auto_same_sev = auto_src_counts[sev_idx] > 0
        else:
            auto_any = auto_tgt_any
            human_counts_side = human_tgt_counts
            human_exact_side = human_exact_tgt
            human_mismatch_side = human_mismatch_tgt
            human_unmatched_side = human_unmatched_tgt
            auto_same_sev = auto_tgt_counts[sev_idx] > 0

        error_mask = np.zeros(
            len(src) if human_error.is_source_error else len(tgt), dtype=bool
        )
        error_mask[start:end] = True

        overlap_any = int(np.sum(error_mask & auto_any))
        if overlap_any > 0:
            overlap_same = int(np.sum(error_mask & auto_same_sev))
            binary_reward = 1.0 if overlap_same > 0 else (1.0 - p)
            tp_for_recall += binary_reward
            fn += 1.0 - binary_reward
        else:
            fn += 1.0

        span_tp_char_sum, span_fn_char_sum = 0.0, 0.0
        for i in range(start, end):
            h_s = float(human_counts_side[sev_idx, i])
            if h_s > 0:
                span_tp_char_sum += (
                    human_exact_side[sev_idx, i]
                    + (1.0 - p) * human_mismatch_side[sev_idx, i]
                ) / h_s
                span_fn_char_sum += (
                    p * human_mismatch_side[sev_idx, i]
                    + human_unmatched_side[sev_idx, i]
                ) / h_s

        tppc_for_recall += span_tp_char_sum / span_len
        fnpc += span_fn_char_sum / span_len

    return (
        0.0,
        0.0,
        0.0,
        tp_for_precision,
        tp_for_recall,
        fp,
        fn,
        tpc,
        fpc,
        fnc,
        round(tppc_for_precision, 10),
        round(tppc_for_recall, 10),
        round(fppc, 10),
        round(fnpc, 10),
    )


def compute_tp_fp_fn_without_error_matching_without_overlaps(
    automatic_eval: AutomaticEvaluation,
    human_eval: HumanEvaluation,
    src: str,
    tgt: str,
    severity_penalty: float,
) -> Tuple[float, ...]:
    """Compute TP, FP, FN metrics without error matching (no overlapping errors)."""
    p = float(severity_penalty)

    tp_for_precision, tp_for_recall, fp, fn = 0.0, 0.0, 0.0, 0.0
    tpc, fpc, fnc = 0.0, 0.0, 0.0
    tppc_for_precision, tppc_for_recall, fppc, fnpc = 0.0, 0.0, 0.0, 0.0

    auto_src_error_chars = np.zeros((len(severity_to_idx), len(src)), dtype=bool)
    auto_tgt_error_chars = np.zeros((len(severity_to_idx), len(tgt)), dtype=bool)
    human_src_error_chars = np.zeros((len(severity_to_idx), len(src)), dtype=bool)
    human_tgt_error_chars = np.zeros((len(severity_to_idx), len(tgt)), dtype=bool)

    for auto_error in automatic_eval.errors:
        sev_idx = severity_to_idx[auto_error.severity]
        if auto_error.is_source_error:
            auto_src_error_chars[sev_idx, auto_error.start : auto_error.end] = True
        else:
            auto_tgt_error_chars[sev_idx, auto_error.start : auto_error.end] = True

    for human_error in human_eval.errors:
        sev_idx = severity_to_idx[human_error.severity]
        if human_error.is_source_error:
            human_src_error_chars[sev_idx, human_error.start : human_error.end] = True
        else:
            human_tgt_error_chars[sev_idx, human_error.start : human_error.end] = True

    for auto_error in automatic_eval.errors:
        auto_sev_idx = severity_to_idx[auto_error.severity]
        if auto_error.is_source_error:
            error_mask = np.zeros(len(src), dtype=bool)
            human_error_chars = human_src_error_chars
        else:
            error_mask = np.zeros(len(tgt), dtype=bool)
            human_error_chars = human_tgt_error_chars

        error_mask[auto_error.start : auto_error.end] = True
        overlap_chars = int(np.sum(error_mask & human_error_chars))
        overlap_chars_matching = int(
            np.sum(error_mask & human_error_chars[auto_sev_idx])
        )

        if overlap_chars > 0:
            binary_reward = 1 if overlap_chars_matching > 0 else (1 - p)
            tp_for_precision += binary_reward
            fp += 1 - binary_reward

            overlap_reward = overlap_chars_matching + (
                overlap_chars - overlap_chars_matching
            ) * (1 - p)
            tpc += overlap_reward
            fpc += len(auto_error.span) - overlap_reward

            portion_reward = overlap_reward / len(auto_error.span)
            tppc_for_precision += portion_reward
            fppc += 1 - portion_reward
        else:
            fp += 1
            fpc += len(auto_error.span)
            fppc += 1

    for human_error in human_eval.errors:
        human_sev_idx = severity_to_idx[human_error.severity]
        if human_error.is_source_error:
            error_mask = np.zeros(len(src), dtype=bool)
            auto_error_chars = auto_src_error_chars
        else:
            error_mask = np.zeros(len(tgt), dtype=bool)
            auto_error_chars = auto_tgt_error_chars

        error_mask[human_error.start : human_error.end] = True
        overlap_chars = int(np.sum(error_mask & auto_error_chars))
        overlap_chars_matching = int(
            np.sum(error_mask & auto_error_chars[human_sev_idx])
        )

        if overlap_chars > 0:
            binary_reward = 1 if overlap_chars_matching > 0 else (1 - p)
            tp_for_recall += binary_reward
            fn += 1 - binary_reward

            overlap_reward = overlap_chars_matching + (
                overlap_chars - overlap_chars_matching
            ) * (1 - p)
            fnc += len(human_error.span) - overlap_reward

            portion_reward = overlap_reward / len(human_error.span)
            tppc_for_recall += portion_reward
            fnpc += 1 - portion_reward
        else:
            fn += 1
            fnc += len(human_error.span)
            fnpc += 1

    return (
        0.0,
        0.0,
        0.0,
        tp_for_precision,
        tp_for_recall,
        fp,
        fn,
        tpc,
        fpc,
        fnc,
        round(tppc_for_precision, 10),
        round(tppc_for_recall, 10),
        round(fppc, 10),
        round(fnpc, 10),
    )


def compute_tp_fp_fn_with_error_matching(
    automatic_eval: AutomaticEvaluation,
    human_eval: HumanEvaluation,
    src: str,
    tgt: str,
    severity_penalty: float = 0.0,
    use_greedy_matching: bool = False,
) -> Tuple[float, ...]:
    """Compute TP, FP, FN metrics using greedy 1-to-1 bipartite error matching."""
    p = float(severity_penalty)

    tp_em, fp_em, fn_em = 0.0, 0.0, 0.0
    tp_for_precision, tp_for_recall, fp, fn = 0.0, 0.0, 0.0, 0.0
    tpc, fpc, fnc = 0.0, 0.0, 0.0
    tppc_for_precision, tppc_for_recall, fppc, fnpc = 0.0, 0.0, 0.0, 0.0

    if use_greedy_matching:
        auto_matches_raw, human_matches_raw = find_greedy_bipartite_matching(
            automatic_eval.errors, human_eval.errors, src, tgt, severity_penalty
        )
        # when using the greedy algorithm, a single matching is used for all measures
        auto_matches_raw_em, human_matches_raw_em = auto_matches_raw, human_matches_raw
        auto_matches_raw_p, human_matches_raw_p = auto_matches_raw, human_matches_raw
        auto_matches_raw_c, human_matches_raw_c = auto_matches_raw, human_matches_raw
        auto_matches_raw_pc, human_matches_raw_pc = auto_matches_raw, human_matches_raw
    else:
        # If not using greedy matching, compute a different optimal matching for each measure (exact match, character counts, character proportion) to give the automatic evaluation the best possible score for each measure.
        auto_matches_raw_em, human_matches_raw_em = find_optimal_bipartite_matching(
            automatic_eval.errors,
            human_eval.errors,
            src,
            tgt,
            span_f_score_exact_match,
            severity_penalty,
        )
        auto_matches_raw_p, human_matches_raw_p = find_optimal_bipartite_matching(
            automatic_eval.errors,
            human_eval.errors,
            src,
            tgt,
            span_f_score_partial,
            severity_penalty,
        )
        auto_matches_raw_c, human_matches_raw_c = find_optimal_bipartite_matching(
            automatic_eval.errors,
            human_eval.errors,
            src,
            tgt,
            span_tp_char_counts,
            severity_penalty,
        )
        auto_matches_raw_pc, human_matches_raw_pc = find_optimal_bipartite_matching(
            automatic_eval.errors,
            human_eval.errors,
            src,
            tgt,
            span_f_score_proportion_chars,
            severity_penalty,
        )

    # --- Precision side: iterate over auto errors ---
    for i, auto_error in enumerate(automatic_eval.errors):
        span_len = len(auto_error.span)

        # Exact Match (uses _em matching)
        match = _get_match_reward(
            auto_matches_raw_em, i, auto_error, human_eval.errors, p
        )
        if match:
            binary_reward, lcs_len, matched_human = match
            is_exact = (
                auto_error.start == matched_human.start
                and auto_error.end == matched_human.end
            )
            exact_match_reward = binary_reward if is_exact else 0.0
            tp_em += exact_match_reward
            fp_em += 1.0 - exact_match_reward
        else:
            fp_em += 1.0

        # Partial Overlap (uses _p matching)
        match = _get_match_reward(
            auto_matches_raw_p, i, auto_error, human_eval.errors, p
        )
        if match:
            binary_reward, lcs_len, matched_human = match
            tp_for_precision += binary_reward
            fp += 1.0 - binary_reward
        else:
            fp += 1.0

        # Character Counts (uses _c matching)
        match = _get_match_reward(
            auto_matches_raw_c, i, auto_error, human_eval.errors, p
        )
        if match:
            binary_reward, lcs_len, matched_human = match
            overlap_chars_reward = lcs_len * binary_reward
            tpc += overlap_chars_reward
            fpc += span_len - overlap_chars_reward
        else:
            fpc += span_len

        # Character Proportion (uses _pc matching)
        match = _get_match_reward(
            auto_matches_raw_pc, i, auto_error, human_eval.errors, p
        )
        if match:
            binary_reward, lcs_len, matched_human = match
            portion_overlap_reward = (lcs_len / span_len) * binary_reward
            tppc_for_precision += portion_overlap_reward
            fppc += 1.0 - portion_overlap_reward
        else:
            fppc += 1.0

    # --- Recall side: iterate over human errors ---
    for i, human_error in enumerate(human_eval.errors):
        span_len = len(human_error.span)

        # Exact Match (uses _em matching)
        match = _get_match_reward(
            human_matches_raw_em, i, human_error, automatic_eval.errors, p
        )
        if match:
            binary_reward, lcs_len, matched_auto = match
            is_exact = (
                human_error.start == matched_auto.start
                and human_error.end == matched_auto.end
            )
            exact_match_reward = binary_reward if is_exact else 0.0
            fn_em += 1.0 - exact_match_reward
        else:
            fn_em += 1.0

        # Partial Overlap (uses _p matching)
        match = _get_match_reward(
            human_matches_raw_p, i, human_error, automatic_eval.errors, p
        )
        if match:
            binary_reward, lcs_len, matched_auto = match
            tp_for_recall += binary_reward
            fn += 1.0 - binary_reward
        else:
            fn += 1.0

        # Character Counts (uses _c matching)
        match = _get_match_reward(
            human_matches_raw_c, i, human_error, automatic_eval.errors, p
        )
        if match:
            binary_reward, lcs_len, matched_auto = match
            overlap_chars_reward = lcs_len * binary_reward
            fnc += span_len - overlap_chars_reward
        else:
            fnc += span_len

        # Character Proportion (uses _pc matching)
        match = _get_match_reward(
            human_matches_raw_pc, i, human_error, automatic_eval.errors, p
        )
        if match:
            binary_reward, lcs_len, matched_auto = match
            portion_overlap_reward = (lcs_len / span_len) * binary_reward
            tppc_for_recall += portion_overlap_reward
            fnpc += 1.0 - portion_overlap_reward
        else:
            fnpc += 1.0

    return (
        tp_em,
        fp_em,
        fn_em,
        tp_for_precision,
        tp_for_recall,
        fp,
        fn,
        tpc,
        fpc,
        fnc,
        tppc_for_precision,
        tppc_for_recall,
        fppc,
        fnpc,
    )


def compute_all_tp_fp_fn(
    samples_with_automatic_evaluations: List[Sample],
    samples_with_human_evaluations: List[Sample],
    do_error_matching: bool = True,
    severity_penalty: float = 0.0,
    has_overlapping_errors: bool = False,
    fix_edge_cases_in_precision: bool = False,
    use_greedy_matching: bool = False,
) -> Dict[str, Dict[str, MicroMetrics | MacroMetrics]]:
    """Compute aggregated TP, FP, FN for all samples."""
    confusion_matrices = {metric_type: MicroMetrics() for metric_type in METRIC_TYPES}
    macro_metrics = {metric_type: MacroMetrics() for metric_type in METRIC_TYPES}

    for sample_with_auto, sample_with_human in zip(
        samples_with_automatic_evaluations, samples_with_human_evaluations
    ):
        assert sample_with_auto.get_input_hash() == sample_with_human.get_input_hash()

        src = sample_with_auto.src
        tgt = sample_with_auto.tgt
        automatic_eval = sample_with_auto.evaluation
        human_eval = sample_with_human.human_evaluation

        if do_error_matching:
            res = compute_tp_fp_fn_with_error_matching(
                automatic_eval,
                human_eval,
                src,
                tgt,
                severity_penalty,
                use_greedy_matching,
            )
        else:
            if has_overlapping_errors:
                res = compute_tp_fp_fn_without_error_matching(
                    automatic_eval, human_eval, src, tgt, severity_penalty
                )
            else:
                res = compute_tp_fp_fn_without_error_matching_without_overlaps(
                    automatic_eval, human_eval, src, tgt, severity_penalty
                )

        (
            tp_em,
            fp_em,
            fn_em,
            tp_for_precision,
            tp_for_recall,
            fp,
            fn,
            tpc,
            fpc,
            fnc,
            tppc_for_precision,
            tppc_for_recall,
            fppc,
            fnpc,
        ) = res

        confusion_matrices["Exact\nMatch"].update(tp_em, tp_em, fp_em, fn_em)
        confusion_matrices["Partial\nOverlap"].update(
            tp_for_precision, tp_for_recall, fp, fn
        )
        confusion_matrices["Character\nCounts"].update(tpc, tpc, fpc, fnc)
        confusion_matrices["Character\nProportion"].update(
            tppc_for_precision, tppc_for_recall, fppc, fnpc
        )

        p_em, r_em, f1_em = compute_p_r_f1_from_tp_fp_fn(
            tp_em, tp_em, fp_em, fn_em, fix_edge_cases_in_precision
        )
        p, r, f1 = compute_p_r_f1_from_tp_fp_fn(
            tp_for_precision, tp_for_recall, fp, fn, fix_edge_cases_in_precision
        )
        pc, rc, f1c = compute_p_r_f1_from_tp_fp_fn(
            tpc, tpc, fpc, fnc, fix_edge_cases_in_precision
        )
        ppc, rpc, f1pc = compute_p_r_f1_from_tp_fp_fn(
            tppc_for_precision, tppc_for_recall, fppc, fnpc, fix_edge_cases_in_precision
        )

        macro_metrics["Exact\nMatch"].update(p_em, r_em, f1_em)
        macro_metrics["Partial\nOverlap"].update(p, r, f1)
        macro_metrics["Character\nCounts"].update(pc, rc, f1c)
        macro_metrics["Character\nProportion"].update(ppc, rpc, f1pc)

    return {"micro": confusion_matrices, "macro": macro_metrics}


def merge_autoevaluators_based_on_info(
    merge_info: Dict[str, Union[str, List[Dict]]],
    metric_name2lp2sys2samples: Dict[str, Dict[str, Dict[str, List[Sample]]]],
):
    """Merge autoevaluators based on info dictionary."""
    merged_name = merge_info["merged_name"]
    metrics_to_merge = merge_info["metrics_to_merge"]
    merging_strategy = merge_info["merging_strategy"]

    metrics_to_merge_names = [
        get_metric_display_name(
            metric_entry["autoeval"],
            metric_entry["model"],
            metric_entry["run_specific_info"],
        )
        for metric_entry in metrics_to_merge
    ]

    assert all(
        metric_name in metric_name2lp2sys2samples
        for metric_name in metrics_to_merge_names
    )

    list_of_metrics_to_merge = [
        metric_name2lp2sys2samples[metric_name]
        for metric_name in metrics_to_merge_names
    ]

    merged_scores = merge_autoevaluators(list_of_metrics_to_merge, merging_strategy)
    metric_name2lp2sys2samples[merged_name] = merged_scores


def aggregate_metrics(
    metrics: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Metrics]]]]],
    lp_key: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Metrics]]]]]:
    """Aggregate metrics over the language pair dimension."""
    global_metrics = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(None)))
        )
    )

    for autoeval, lp2aggr2match2tp in metrics.items():
        for lp, aggr2match2tp in lp2aggr2match2tp.items():
            for aggr, match2tp in aggr2match2tp.items():
                for match, tp2metrics in match2tp.items():
                    for tp, tp_metrics in tp2metrics.items():
                        global_metrics[autoeval][lp_key][aggr][match][tp] = (
                            MicroMetrics() if aggr == "micro" else MacroMetrics()
                        )

    for autoeval, lp2aggr2match2tp in metrics.items():
        for lp, aggr2match2tp in lp2aggr2match2tp.items():
            for aggr, match2tp in aggr2match2tp.items():
                for match, tp2metrics in match2tp.items():
                    for tp, tp_metrics in tp2metrics.items():
                        global_metrics[autoeval][lp_key][aggr][match][tp].update(
                            *tp_metrics.get_values()
                        )

    return global_metrics


def compute_results_from_metrics(
    metrics: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Metrics]]]]],
) -> Dict[str, Dict[str, Dict[str, Dict[str, MetricResults]]]]:
    """Compute results from metrics dictionary."""
    results = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(MetricResults)))
    )

    for autoeval, lp2aggr2match2tp2metrics in metrics.items():
        for lp, aggr2match2tp2metrics in lp2aggr2match2tp2metrics.items():
            for aggr, match2tp2metrics in aggr2match2tp2metrics.items():
                for match, tp2metrics in match2tp2metrics.items():
                    for tp, m in tp2metrics.items():
                        results[lp][aggr][match][autoeval].update(
                            tp, *list(m.get_precision_recall_f1())
                        )

    return results


def macro_average_results_across_lps(
    results: Dict[str, Dict[str, Dict[str, Dict[str, MetricResults]]]],
    lp_key: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, MetricResults]]]]:
    global_results = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(MetricResults)))
    )
    lps = list(results.keys())
    aggregation_strategies = next(iter(results.values())).keys()
    matching_strategies = next(iter(next(iter(results.values())).values())).keys()
    autoevals = next(
        iter(next(iter(next(iter(results.values())).values())).values())
    ).keys()
    metric_types = next(
        iter(
            next(
                iter(next(iter(next(iter(results.values())).values())).values())
            ).values()
        )
    ).results.keys()

    for aggr in aggregation_strategies:
        for match in matching_strategies:
            for autoeval in autoevals:
                for metric_type in metric_types:
                    for lp in lps:
                        lp_res = results[lp][aggr][match][autoeval].get(metric_type)
                        p, r, f1 = lp_res.precision, lp_res.recall, lp_res.f1
                        global_results[lp_key][aggr][match][autoeval].update(
                            metric_type, p, r, f1
                        )
                    global_results[lp_key][aggr][match][autoeval].average(
                        metric_type, len(lps)
                    )

    return global_results
