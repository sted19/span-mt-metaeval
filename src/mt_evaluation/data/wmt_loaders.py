# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
WMT-specific data loaders for the MT Evaluation Framework.

This module provides utilities for loading WMT evaluation data from various formats
including TSV submissions, raw JSON/JSONL files, and mt-metrics-eval data.
"""

import json
import logging
from typing import Dict, List
from collections import defaultdict
from pathlib import Path

from mt_metrics_eval import data as mt_metrics_eval_data
from mt_metrics_eval import ratings as mt_metrics_eval_ratings

from mt_evaluation.core import Sample, HumanEvaluation, Error, AutomaticEvaluation
from mt_evaluation.core import no_error, no_error2, wmt25_lps_mqm
from mt_evaluation.core.scoring import assign_score_based_on_severity
from mt_evaluation.utils import find_all_literal
from mt_evaluation.data.language_codes import lang_code2lang

logger = logging.getLogger(__name__)


# Category mapping for en-es language pair (WMT25-specific)
# Maps fine-grained subcategories to MQM top-level categories
enes_subcategory_to_category_mapping = {
    "mt_hallucination": "accuracy",
    "wrong_named_entity": "accuracy",
    "untranslated": "accuracy",
    "mistranslation": "accuracy",
    "wrong_term": "accuracy",
    "omission": "accuracy",
    "addition": "accuracy",
}


def parse_tsv_wmt25_submission(
    gold_lp2sys2samples: Dict[str, Dict[str, List[Sample]]],
    autoeval_name: str,
    filepath: Path,
    lps: List[str],
    fix_indices_with_tgt_annotated: bool = False,
) -> Dict[str, Dict[str, List[Sample]]]:
    """
    Load TSV data with space-separated values in the last three columns.

    Args:
        gold_lp2sys2samples: Dictionary with keys lp, sys_name and values the list
            of samples with human annotations to use to construct the list of predictions
        autoeval_name: Name of the automatic evaluator
        filepath: Path to the .tsv file
        lps: Language pairs to process
        fix_indices_with_tgt_annotated: Whether to fix indices using annotated target

    Returns:
        Dictionary mapping lp -> system -> samples with automatic evaluations
    """
    data = []

    with open(filepath, "r", encoding="utf-8") as f:
        # Read header
        header = f.readline().strip().split("\t")

        # Read data rows
        for line in f:
            values = line.strip().split("\t")
            row = {}

            lp = values[header.index("doc_id")].split("_#_")[0]
            if lp not in lps:
                continue

            for i, col_name in enumerate(header):
                value = values[i] if i < len(values) else ""

                # Parse space-separated columns
                if col_name in ["start_indices", "end_indices"]:
                    row[col_name] = (
                        [int(x) for x in value.split() if x != "missing"]
                        if value
                        else []
                    )
                elif col_name == "error_types":
                    row[col_name] = value.split() if value else []
                elif col_name == "segment_id":
                    row[col_name] = int(value)
                else:
                    row[col_name] = value

            data.append(row)

    num_total_errors, num_errors_impossible_to_find = 0, 0
    auto_lp2sys2samples = defaultdict(lambda: defaultdict(list))
    for lp, sys2samples in gold_lp2sys2samples.items():
        for sys, samples in sys2samples.items():
            for sample in samples:
                assigned = 0
                for elem in data:
                    if (
                        elem["doc_id"] == sample.doc_id
                        and elem["segment_id"] == sample.seg_id
                        and elem["system_id"] == sys
                    ):

                        assert len(elem["start_indices"]) == len(elem["end_indices"])
                        assert len(elem["error_types"]) >= len(elem["start_indices"])

                        if elem["set_id"] != "official":
                            continue

                        errors = []
                        for start_i, end_i, severity in zip(
                            elem["start_indices"],
                            elem["end_indices"],
                            elem["error_types"],
                        ):
                            if no_error in severity or no_error2 in severity:
                                continue

                            assert start_i > -1 and end_i > -1

                            autoevals_to_fix = ["AIP.pri"]
                            if (
                                any(
                                    autoeval in autoeval_name
                                    for autoeval in autoevals_to_fix
                                )
                                and fix_indices_with_tgt_annotated
                            ):
                                if start_i == end_i or start_i >= len(
                                    sample.tgt_annotated
                                ):
                                    continue
                                error_span = sample.tgt_annotated[start_i:end_i]

                                if error_span not in sample.tgt:
                                    logger.debug("Annotated error span not in tgt!")
                                    error_span = sample.tgt[start_i:end_i]
                                    num_errors_impossible_to_find += 1
                                else:
                                    matches = find_all_literal(sample.tgt, error_span)

                                    differences = [
                                        (
                                            abs(start_i - fixed_start_i),
                                            abs(end_i - fixed_end_i),
                                            fixed_start_i,
                                            fixed_end_i,
                                        )
                                        for fixed_start_i, fixed_end_i in matches
                                    ]

                                    sorted_differences = sorted(
                                        differences, key=lambda x: (x[0], x[1])
                                    )

                                    start_i, end_i = (
                                        sorted_differences[0][2],
                                        sorted_differences[0][3],
                                    )

                            else:
                                if start_i == end_i or start_i >= len(sample.tgt):
                                    continue
                                error_span = sample.tgt[start_i:end_i]

                            errors.append(
                                Error(
                                    span=error_span,
                                    category="",
                                    severity=severity,
                                    start=start_i,
                                    end=end_i,
                                    is_source_error=False,
                                    score=assign_score_based_on_severity(severity),
                                    explanation="",
                                    extended_span=None,
                                )
                            )
                            num_total_errors += 1

                        evaluation = AutomaticEvaluation(
                            score=sum(error.score for error in errors),
                            errors=errors,
                            annotation="",
                            parsing_error=False,
                        )

                        auto_sample = Sample.from_dict(sample.to_dict())
                        auto_sample.human_evaluation = None
                        auto_sample.evaluation = evaluation

                        auto_lp2sys2samples[lp][sys].append(auto_sample)

                        assigned += 1

                assert assigned == 1

    logger.info(
        f"Num errors whose position was impossible to fix: {num_errors_impossible_to_find}/{num_total_errors}"
    )

    return auto_lp2sys2samples


def convert_raw_esa_evaluation_to_human_evaluation(
    evaluation: Dict, tgt: str
) -> HumanEvaluation:
    """Convert raw ESA evaluation format to HumanEvaluation object."""
    rater = evaluation["annotator"]
    esa_score = evaluation["score"]

    errors = []
    for error in evaluation["errors"]:
        start = error["start_i"]
        end = error["end_i"]

        if start == "missing" or end == "missing":
            continue

        severity = error["severity"]
        if severity == "undecided":
            continue

        category = ""
        is_source_error = False
        error_span = tgt[start:end]

        errors.append(
            Error(
                span=error_span,
                category=category,
                severity=severity,
                start=start,
                end=end,
                is_source_error=is_source_error,
                score=assign_score_based_on_severity(severity),
            )
        )

    return HumanEvaluation(score=esa_score, errors=errors, rater=rater)


def get_wmt25_esa_rater_data_from_raw_jsonl(
    json_path: Path, lps: List[str]
) -> Dict[str, Dict[str, Dict[str, List[Sample]]]]:
    """Load WMT25 ESA rater data from raw JSONL file."""
    data = [json.loads(line) for line in open(json_path)]

    rater2lp2sys2samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for data_sample in data:
        src = data_sample["src_text"]
        tgts = data_sample["tgt_text"]
        doc_id = data_sample["doc_id"]
        sys2scores = data_sample["scores"]

        lp = doc_id.split("_#_")[0]
        seg_id = int(doc_id.split("_#_")[-1])
        doc_id = "_#_".join(doc_id.split("_#_")[:-1])

        if lp not in lps:
            continue

        src_lang = lang_code2lang[lp.split("-")[0]]
        tgt_lang = lang_code2lang[lp.split("-")[1]]

        for sys, scores in sys2scores.items():
            tgt = tgts[sys]

            assert len(scores) == 2, f"len(scores)={len(scores)} != 2!"

            eval1 = scores[0]
            eval2 = scores[1]

            human_1_eval = convert_raw_esa_evaluation_to_human_evaluation(eval1, tgt)
            human_2_eval = convert_raw_esa_evaluation_to_human_evaluation(eval2, tgt)

            sample1 = Sample(
                src=src,
                tgt=tgt,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                evaluation=None,
                human_evaluation=human_1_eval,
                doc_id=doc_id,
                seg_id=seg_id,
            )

            sample2 = Sample.from_dict(sample1.to_dict())
            sample2.human_eval = human_2_eval

            rater2lp2sys2samples["Human-1"][lp][sys].append(sample1)
            rater2lp2sys2samples["Human-2"][lp][sys].append(sample2)

    return rater2lp2sys2samples


def get_wmt25_mqm_rater_lp_data_from_raw_json(
    json_path: Path,
) -> Dict[str, List[Sample]]:
    """Load WMT25 MQM rater data from raw JSON file."""
    data = json.loads(open(json_path).read())

    keys = list(data.keys())
    samples_keys = data[keys[0]].keys()
    samples = []
    for sample_key in samples_keys:
        sample = {}
        for field_key, values in data.items():
            sample[field_key] = values[sample_key]
        samples.append(sample)

    num_tgt_mismatches = 0
    sys2samples = defaultdict(list)
    for sample in samples:
        set_id = sample["set_id"]
        if set_id != "official":
            continue

        src_lang = lang_code2lang[sample["source_lang"]]
        tgt_lang = lang_code2lang[sample["target_lang"]]
        src = sample["source_segment"]
        tgt = sample["hypothesis_segment"]
        tgt_annotated = sample["hypothesis_annotated"]
        doc_id = sample["doc_id"]
        seg_id = int(sample["segment_id"])

        if tgt != tgt_annotated:
            logger.debug(
                "sample.tgt != sample.tgt_annotated, potential mismatches in annotations!"
            )
            num_tgt_mismatches += 1

        errors = []

        start_indices = (
            sample["start_indices_gold"].split()
            if sample["start_indices_gold"] != "-1"
            else []
        )
        end_indices = (
            sample["end_indices_gold"].split()
            if sample["end_indices_gold"] != "-1"
            else []
        )
        error_severities = (
            sample["error_types_gold"].split()
            if sample["error_types_gold"] != "no-error"
            else []
        )

        assert (
            len(error_severities)
            == len(sample["mapped_errors"])
            == len(start_indices)
            == len(end_indices)
        ), (
            f"len(error_severities)={len(error_severities)} != "
            f"len(sample['mapped_errors'])={len(sample['mapped_errors'])} != "
            f"len(start_indices)={len(start_indices)} != "
            f"len(end_indices)={len(end_indices)}"
        )

        for start, end, severity in zip(start_indices, end_indices, error_severities):
            if start is None or end is None or start == "missing" or end == "missing":
                continue

            start = int(start)
            end = int(end)

            is_source_error = False
            error_span = src[start:end] if is_source_error else tgt[start:end]

            errors.append(
                Error(
                    span=error_span,
                    category="",
                    severity=severity,
                    start=start,
                    end=end,
                    is_source_error=is_source_error,
                    score=assign_score_based_on_severity(severity),
                )
            )

        human_evaluation = HumanEvaluation(
            score=sum(error.score for error in errors),
            errors=errors,
        )
        sys2samples[sample["system_id"]].append(
            Sample(
                src=src,
                tgt=tgt,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                evaluation=None,
                human_evaluation=human_evaluation,
                doc_id=doc_id,
                seg_id=seg_id,
                tgt_annotated=tgt_annotated,
            )
        )

    logger.info(
        f"The number of target mismatches are: {num_tgt_mismatches}/{len(samples)}"
    )

    return sys2samples


def get_super_raters_from_raters(
    rater2lp2sys2samples: Dict[str, Dict[str, Dict[str, List[Sample]]]],
    super_raters: List[str],
) -> Dict[str, Dict[str, Dict[str, List[Sample]]]]:
    """
    Convert rater-based annotations to super-rater format.

    Args:
        rater2lp2sys2samples: Dictionary with keys rater, lp, system and values lists of samples
        super_raters: List of names used for the super raters

    Returns:
        Dictionary with keys super_rater, lp, system and values lists of samples
    """
    super_rater2lp2sys2samples = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    raters = sorted(rater2lp2sys2samples.keys())
    lps = set()
    for rater in raters:
        lps = lps.union(set(rater2lp2sys2samples[rater].keys()))

    for lp in sorted(lps):
        all_systems = set()
        for rater in raters:
            all_systems = all_systems.union(set(rater2lp2sys2samples[rater][lp].keys()))
        for system in sorted(all_systems):
            max_num_samples = max(
                len(rater2lp2sys2samples[rater][lp][system]) for rater in raters
            )
            for i in range(max_num_samples):
                current_super_rater = 0
                for rater in raters:
                    if (
                        len(rater2lp2sys2samples[rater][lp][system]) > 0
                        and rater2lp2sys2samples[rater][lp][system][i].human_evaluation
                        is not None
                    ):
                        super_rater = super_raters[current_super_rater]
                        super_rater2lp2sys2samples[super_rater][lp][system].append(
                            rater2lp2sys2samples[rater][lp][system][i]
                        )
                        current_super_rater += 1

                if current_super_rater > len(super_raters):
                    raise RuntimeError(
                        f"This sample has more than {len(super_raters)} annotations!"
                    )

                if current_super_rater < len(super_raters):
                    for idx in range(current_super_rater):
                        super_rater2lp2sys2samples[super_raters[idx]][lp][system].pop()

                    for super_rater in super_raters:
                        for rater in raters:
                            if len(rater2lp2sys2samples[rater][lp][system]) > 0:
                                super_rater2lp2sys2samples[super_rater][lp][
                                    system
                                ].append(rater2lp2sys2samples[rater][lp][system][i])
                                super_rater2lp2sys2samples[super_rater][lp][system][
                                    i
                                ].human_evaluation = None
                                break
                        else:
                            raise RuntimeError(
                                "Either there is no Rater with a None evaluation, "
                                "or No rater annotated this system, which should be impossible!"
                            )

    return super_rater2lp2sys2samples


def get_raters_evaluations(
    test_set: str,
    lps: list[str],
    use_merged_annotations: bool = False,
) -> Dict[str, Dict[str, Dict[str, List[Sample]]]]:
    """
    Load WMT evaluation data from all individual raters.

    Args:
        test_set: One of wmt22, wmt23, wmt24, wmt25
        lps: List of language pairs (e.g., en-de, en-zh, zh-en)
        use_merged_annotations: Whether to use merged evaluations

    Returns:
        Nested dictionary with samples containing human evaluations
    """
    rater2lp2sys2samples: Dict[str, Dict[str, Dict[str, List[Sample]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    if test_set == "wmt25":
        mqm_lps = [lp for lp in lps if lp in wmt25_lps_mqm]
        for lp in mqm_lps:
            json_path = Path(f"data/wmt25/data/mqm_generalMT2025_{lp}_with_errors.json")
            rater2lp2sys2samples["mqm.super.1"][lp] = (
                get_wmt25_mqm_rater_lp_data_from_raw_json(json_path)
            )
        return rater2lp2sys2samples

    for lp in lps:
        evalset = mt_metrics_eval_data.EvalSet(
            test_set, lp, read_stored_metric_scores=True, read_stored_ratings=True
        )

        src_lang = lang_code2lang.get(evalset.src_lang)
        tgt_lang = lang_code2lang.get(evalset.tgt_lang)

        if src_lang is None or tgt_lang is None:
            raise ValueError(
                f"Language code not found: {evalset.src_lang} or {evalset.tgt_lang}"
            )

        if use_merged_annotations:
            raters = [rater for rater in evalset._ratings.keys() if "merged" in rater]
            if len(raters) == 0:
                logger.warning(
                    f"No merged raters for test set {test_set} and lp {lp}. "
                    "Defaulting to individual raters."
                )
                raters = [
                    rater for rater in evalset._ratings.keys() if "merged" not in rater
                ]
        else:
            raters = [
                rater for rater in evalset._ratings.keys() if "merged" not in rater
            ]

        logger.info(
            f"Test set={test_set}, lp={lp}, raters={raters}, "
            f"use_merged_annotations={use_merged_annotations}"
        )

        systems = evalset.sys_outputs.keys()
        srcs = evalset.src

        for rater in raters:
            for sys in systems:
                tgts = evalset.sys_outputs[sys]
                mqm_ratings = evalset.Ratings(rater)[sys]
                docs_per_seg = evalset.DocsPerSeg()

                assert (
                    len(srcs) == len(tgts) == len(mqm_ratings) == len(docs_per_seg)
                ), (
                    f"len(srcs) = {len(srcs)} != len(tgts) = {len(tgts)} != "
                    f"len(mqm_ratings) = {len(mqm_ratings)} != "
                    f"len(docs_per_seg) = {len(docs_per_seg)}"
                )

                for seg_id, (src, tgt, mqm_rating, doc_id) in enumerate(
                    zip(srcs, tgts, mqm_ratings, docs_per_seg)
                ):
                    human_evaluation = None
                    if mqm_rating is not None:
                        errors = []
                        for error in mqm_rating.errors:
                            error_span = (
                                src[error.start : error.end]
                                if error.is_source_error
                                else tgt[error.start : error.end]
                            )

                            errors.append(
                                Error(
                                    span=error_span,
                                    category=error.category,
                                    severity=error.severity,
                                    start=error.start,
                                    end=error.end,
                                    score=-error.score,
                                    is_source_error=error.is_source_error,
                                )
                            )

                        human_evaluation = HumanEvaluation(
                            score=sum(error.score for error in errors),
                            errors=errors,
                            rater=rater,
                        )

                    sample = Sample(
                        src, tgt, src_lang, tgt_lang, doc_id=doc_id, seg_id=seg_id
                    )
                    sample.human_evaluation = human_evaluation

                    rater2lp2sys2samples[rater][lp][sys].append(sample)

    return rater2lp2sys2samples


def extract_list_of_errors_from_rating(
    rating: mt_metrics_eval_ratings.Rating, src: str, tgt: str
) -> List[Error]:
    """Extract Error objects from an mt-metrics-eval Rating."""
    errors = [
        Error(
            span=(
                src[error.start : error.end]
                if error.is_source_error
                else tgt[error.start : error.end]
            ),
            category=error.category,
            severity=error.severity,
            start=error.start,
            end=error.end,
            score=-error.score,
        )
        for error in rating.errors
    ]

    return errors
