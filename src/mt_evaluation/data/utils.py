# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
General data utilities for the MT Evaluation Framework.

This module provides general utilities for data handling used throughout
the evaluation framework, including document context reconstruction,
cache management, and sample flattening.
"""

import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from pathlib import Path

from mt_metrics_eval import meta_info as mt_metrics_eval_meta_info

from mt_evaluation.core import Sample
from mt_evaluation.data.cache import MTEvaluationCache
from mt_evaluation.utils import is_all_nones, get_metric_display_name

logger = logging.getLogger(__name__)


def reconstruct_document_context(
    rater2lp2sys2samples: Dict[str, Dict[str, Dict[str, List[Sample]]]],
    num_paragraphs_per_document: int,
) -> Dict[str, Dict[str, Dict[str, List[Sample]]]]:
    """
    Reconstruct document context for samples by grouping segments by document.

    Args:
        rater2lp2sys2samples: Dictionary mapping rater -> lp -> system -> samples
        num_paragraphs_per_document: Number of surrounding paragraphs to include

    Returns:
        Updated dictionary with src_doc field populated for each sample
    """
    # Collect unique samples to reconstruct document context
    unique_samples: Set[Tuple[str, int, str]] = set()
    for rater, lp2sys2samples in rater2lp2sys2samples.items():
        for lp, sys2samples in lp2sys2samples.items():
            for sys, samples in sys2samples.items():
                for sample in samples:
                    assert sample.doc_id is not None and type(sample.doc_id) is str
                    assert sample.seg_id is not None and type(sample.seg_id) is int
                    assert sample.src is not None and type(sample.src) is str

                    unique_samples.add((sample.doc_id, sample.seg_id, sample.src))

    unique_samples_sorted: List[Tuple[str, int, str]] = sorted(
        unique_samples, key=lambda x: (x[0], x[1])
    )

    # Build document ID to document text mapping
    prev_doc_id, doc_id = None, None
    i = 0
    doc = ""
    doc_id2doc = {}
    while i < len(unique_samples_sorted):
        doc_id, seg_id, src = unique_samples_sorted[i]

        # Starting
        if prev_doc_id is None:
            doc = src + "\n\n"
            prev_doc_id = doc_id
        # Completed one document, add concatenated src to the dictionary
        elif doc_id != prev_doc_id:
            doc_id2doc[prev_doc_id] = doc.strip()
            doc = src + "\n\n"
            prev_doc_id = doc_id
        else:
            doc += src + "\n\n"

        i += 1

    doc_id2doc[doc_id] = doc.strip()

    # Create new samples with document context
    new_rater2lp2sys2samples = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    n = num_paragraphs_per_document
    for rater, lp2sys2samples in rater2lp2sys2samples.items():
        for lp, sys2samples in lp2sys2samples.items():
            for sys, samples in sys2samples.items():
                new_samples = []
                for sample in samples:
                    new_sample = Sample.from_dict(sample.to_dict())
                    src_doc = doc_id2doc[new_sample.doc_id]

                    paragraphs = src_doc.split("\n\n")
                    par_idx = [
                        idx
                        for idx, par in enumerate(paragraphs)
                        if par == new_sample.src
                    ][0]
                    new_sample.src_doc = "\n\n".join(
                        (paragraphs[max(0, par_idx - n) : par_idx + n])[: n + 2]
                    )
                    new_samples.append(new_sample)
                new_rater2lp2sys2samples[rater][lp][sys] = new_samples

    return new_rater2lp2sys2samples


def get_autoeval2lp2sys2samples_with_automatic_evaluations(
    lp2sys2samples: Dict[str, Dict[str, List[Sample]]],
    metrics_to_evaluate_info: List[Dict],
    do_not_verify_completeness: bool = False,
) -> Dict[str, Dict[str, Dict[str, List[Sample]]]]:
    """
    Load automatic evaluations from cache for the given samples and metrics.

    Args:
        lp2sys2samples: Dictionary with keys lp, systems and values the samples
            with the gold human evaluation
        metrics_to_evaluate_info: List of dictionaries containing metric info
        do_not_verify_completeness: If True, don't raise error for missing evaluations

    Returns:
        Dictionary containing samples for each metric, language pair, and system.
    """
    autoeval2lp2samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for metric_to_evaluate in metrics_to_evaluate_info:
        metric_name = get_metric_display_name(
            metric_to_evaluate["autoeval"],
            metric_to_evaluate["model"],
            metric_to_evaluate["run_specific_info"],
        )

        # Remove the last part (run-specific-info) for consistent naming
        metric_name = "_".join(metric_name.split("_")[:-1])

        cache_dir = get_cache_dir(
            metric_to_evaluate["autoeval"],
            metric_to_evaluate["model"],
            metric_to_evaluate["outputs_path"],
            metric_to_evaluate["run_specific_info"],
        )
        cache_file = cache_dir / "cache.jsonl"
        cache = MTEvaluationCache(str(cache_file))

        for lp, sys2samples in lp2sys2samples.items():
            for sys, sys_samples in sys2samples.items():
                for sample in sys_samples:
                    sample_with_auto = Sample.from_dict(sample.to_dict())
                    sample_with_auto.human_evaluation = None

                    if sample.human_evaluation is None:
                        sample_with_auto.evaluation = None
                    elif cache.is_evaluated(sample_with_auto):
                        sample_with_auto.evaluation = cache.get_evaluation(
                            sample_with_auto
                        )
                    else:
                        if do_not_verify_completeness:
                            sample_with_auto.evaluation = None
                        else:
                            raise RuntimeError(
                                f"Sample {sample_with_auto} not found in cache. "
                                "Please run the evaluation first."
                            )
                    autoeval2lp2samples[metric_name][lp][sys].append(sample_with_auto)

    return autoeval2lp2samples


def flatten_samples_for_evaluation(
    super_rater2lp2sys2samples: Dict[str, Dict[str, Dict[str, List[Sample]]]],
) -> List[Sample]:
    """
    Flatten nested dictionary to list of unique samples for evaluation.

    Args:
        super_rater2lp2sys2samples: Nested dictionary with samples

    Returns:
        List of unique samples to evaluate
    """
    gold_super_rater = sorted(super_rater2lp2sys2samples.keys())[0]
    lp2sys2samples = super_rater2lp2sys2samples[gold_super_rater]
    
    all_data = []
    for lp, sys2samples in lp2sys2samples.items():
        # Only evaluate systems that have been human-annotated
        mqm_annotated_systems = [
            sys
            for sys, sys_samples in sys2samples.items()
            if not is_all_nones([sample.human_evaluation for sample in sys_samples])
        ]

        logger.debug(
            f"{lp}: annotated systems/total systems: "
            f"{len(mqm_annotated_systems)}/{len(sys2samples.keys())}"
        )

        lp_data = []
        for sys in mqm_annotated_systems:
            sys_samples = sys2samples[sys]
            lp_data.extend(sys_samples)

        lp_data_no_duplicates = list(dict.fromkeys(lp_data))
        logger.debug(
            f"{lp}: unique samples/total samples: "
            f"{len(lp_data_no_duplicates)}/{len(lp_data)}"
        )
        all_data += lp_data_no_duplicates

    logger.debug(f"Total samples: {len(all_data)}")
    return all_data


def get_cache_dir(
    autoeval_name: str,
    autoeval_model_name: str,
    outputs_path: str,
    run_specific_info: str,
) -> Path:
    """
    Get the cache directory path for a specific evaluator and model.

    Args:
        autoeval_name: Name of the automatic evaluator.
        autoeval_model_name: Name of the model used for evaluation.
        outputs_path: Base path for outputs.
        run_specific_info: Additional info to differentiate runs using same model and autoeval

    Returns:
        Path: The cache directory path.
    """
    # Replace forward slashes to avoid path issues
    autoeval_name = autoeval_name.replace("/", "_")
    autoeval_model_name = autoeval_model_name.replace("/", "_")

    cache_dir = (
        Path(outputs_path) / autoeval_name / autoeval_model_name / run_specific_info
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def parse_lps(
    lps: List[str],
    test_set: str,
) -> List[str]:
    """Parse language pairs, expanding 'all' to actual list."""
    if lps == "all":
        return list(mt_metrics_eval_meta_info.DATA[test_set].keys())
    else:
        return lps
