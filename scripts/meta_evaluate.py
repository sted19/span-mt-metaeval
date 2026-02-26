# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import logging
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path
import multiprocessing

from mt_evaluation.core import (
    Sample,
    wmt22_lps,
    wmt23_lps,
    wmt24_lps,
    wmt25_lps,
    wmt25_lps_mqm,
    wmt25_lps_esa,
    UNKNOWN_SEVERITY,
)
from mt_evaluation.meta_evaluation import MetricStats
from mt_evaluation.data import (
    parse_tsv_wmt25_submission,
    get_raters_evaluations,
    get_super_raters_from_raters,
)
from mt_evaluation.data.utils import (
    get_autoeval2lp2sys2samples_with_automatic_evaluations,
)
from mt_evaluation.data.wmt_loaders import enes_subcategory_to_category_mapping
from mt_evaluation.meta_evaluation.metrics_to_evaluate import (
    metrics_to_evaluate_info_wmt25_mqm,
    metrics_to_evaluate_info_wmt25_esa,
    metrics_to_evaluate_info_wmt24,
    metrics_to_evaluate_info_wmt23,
    metrics_to_evaluate_info_wmt22,
    metrics_to_merge_info,
)
from mt_evaluation.utils import setup_logging, convert_defaultdict_to_dict
from mt_evaluation.meta_evaluation.utils import (
    aggregate_stats,
    print_results_and_stats,
    remove_samples_with_none_human_evaluation,
    compute_sentinel_counts,
)
from mt_evaluation.meta_evaluation.span_level.preprocessing import (
    count_errors_by_severity,
    compute_evaluations_stats,
)
from mt_evaluation.meta_evaluation.span_level.preprocessing import (
    preprocess_samples_with_human_evaluations,
    preprocess_single_autoeval_wrapper,
)
from mt_evaluation.meta_evaluation.span_level.metrics import Metrics
from mt_evaluation.meta_evaluation.span_level.utils import (
    merge_autoevaluators_based_on_info,
    aggregate_metrics,
    compute_results_from_metrics,
    process_single_autoeval_wrapper,
)
from mt_evaluation.meta_evaluation.span_level.perturbations import (
    extract_perturbations_from_autoevals,
    ALL_PERTURBATIONS,
)

# NOTE: aspects of the meta-evaluation to remember
#   HANDLING OF HUMAN ERRORS
#  1. We are not considering neutral errors even for the span-based matching (to be coherent with their MQM score of 0)
#  2. We are not considering error categories that do not refer to actual translation errors such as source issue or creative reinterpretation
#   ............................
#   HANDLING OF AUTOMATIC ERRORS
#  1. Ill-formed errors (i.e., empty errors, or those whose text is not entirely contained in the translation or the source), are excluded from the evaluation
#   HANDLING OF MATCHING ALGORITHMS
#  1. By default, we use optimal matching (in terms of f1_with_partial_overlap_and_partial_credit). This is used also for exact_match and partial_overlap, despite it might (might it?) be not optimal for them. This behavior is incorrect and must be edited.


# NOTE: predominant LLM errors and parsing errors
#   1. Omission not marked in the source, but in the target
#   2. Omission has an empty span --> it is true that omitted text is NOT there, but the LLM should not mark an empty span. Rather, should mark the omitted span in the source
#   3. Non-omissions marked in the source for other categories (e.g., Fluency errors marked in the source)

logger = logging.getLogger(__name__)


def main():
    parser = read_arguments()
    args = parser.parse_args()

    setup_logging(args.logging_level)

    test_set2protocol2lps = {
        "wmt22": {"mqm": wmt22_lps},
        "wmt23": {"mqm": wmt23_lps},
        "wmt24": {"mqm": wmt24_lps},
        "wmt25": {"mqm": wmt25_lps_mqm, "esa": wmt25_lps_esa},
    }

    test_set2protocol2metrics_to_evaluate = {
        "wmt22": {"mqm": metrics_to_evaluate_info_wmt22},
        "wmt23": {"mqm": metrics_to_evaluate_info_wmt23},
        "wmt24": {"mqm": metrics_to_evaluate_info_wmt24},
        "wmt25": {
            "mqm": metrics_to_evaluate_info_wmt25_mqm,
            "esa": metrics_to_evaluate_info_wmt25_esa,
        },
    }

    test_sets = args.test_sets
    annotation_protocol = args.annotation_protocol

    if len(test_sets) > 1:
        metrics_to_evaluate = set(
            [
                autoeval_entry["autoeval"] + autoeval_entry["model"]
                for autoeval_entry in test_set2protocol2metrics_to_evaluate[
                    test_sets[0]
                ][annotation_protocol]
            ]
        )
        assert all(
            set(
                [
                    autoeval_entry["autoeval"] + autoeval_entry["model"]
                    for autoeval_entry in test_set2protocol2metrics_to_evaluate[
                        test_set
                    ][annotation_protocol]
                ]
            )
            == metrics_to_evaluate
            for test_set in test_sets
        ), "You must evaluate the same metrics across multiple test sets"

    autoeval2test_set2lp2sys2samples_with_automatic_evaluations = defaultdict(dict)
    test_set2lp2sys2samples_with_human_evaluations = dict()
    for test_set in test_sets:

        lps = test_set2protocol2lps[test_set][annotation_protocol]
        lps = args.lps if args.lps is not None else lps

        logger.info(f"Evaluating {test_set} on {lps}")

        super_raters = (
            [args.gold_rating_key]
            if test_set == "wmt24"
            or (test_set == "wmt25" and annotation_protocol == "mqm")
            else [args.gold_rating_key] + args.human_as_a_metric_rating_keys
        )

        rater2lp2sys2samples = get_raters_evaluations(
            test_set,
            lps,
            use_merged_annotations=args.use_merged_annotations,
            annotation_protocol=annotation_protocol,
        )

        super_rater2lp2sys2samples = get_super_raters_from_raters(
            rater2lp2sys2samples, super_raters
        )

        lp2sys2samples_with_human_evaluations = {
            lp: super_rater2lp2sys2samples[args.gold_rating_key][lp] for lp in lps
        }
        gold_lp2sys2samples = super_rater2lp2sys2samples.pop(args.gold_rating_key)

        # The meta-evaluation can be conducted only on samples that have been human-evaluated
        # Here, we filter out the samples not human-evaluated both from human and automatic samples, maintaining alignment
        (
            super_rater2lp2sys2samples,
            lp2sys2samples_with_human_evaluations,
        ) = remove_samples_with_none_human_evaluation(
            super_rater2lp2sys2samples,
            lp2sys2samples_with_human_evaluations,
        )

        autoeval2lp2sys2samples_with_automatic_evaluations: Dict[
            str, Dict[str, Dict[str, List[Sample]]]
        ] = get_autoeval2lp2sys2samples_with_automatic_evaluations(
            lp2sys2samples=lp2sys2samples_with_human_evaluations,
            metrics_to_evaluate_info=test_set2protocol2metrics_to_evaluate[test_set][
                annotation_protocol
            ],
            do_not_verify_completeness=args.do_not_verify_completeness,
        )

        if test_set == "wmt25" and not args.do_not_load_wmt25_submissions:
            submission_dir = Path("data/wmt25/submissions/")

            for wmt25_submission_path in submission_dir.iterdir():

                if wmt25_submission_path.name.startswith("."):
                    continue

                submission_type = wmt25_submission_path.name.split(".")[2]
                submission_name = wmt25_submission_path.name.split(".")[0]
                autoeval_entry = submission_name + "." + submission_type
                autoeval2lp2sys2samples_with_automatic_evaluations[autoeval_entry] = (
                    parse_tsv_wmt25_submission(
                        gold_lp2sys2samples,
                        autoeval_entry,
                        wmt25_submission_path,
                        lps=lps,
                        fix_indices_with_tgt_annotated=args.fix_wmt25_indices_with_tgt_annotated,
                    )
                )

        if len(test_sets) == 1:
            logger.info(
                f"Extending autoevals with human super-raters: {super_rater2lp2sys2samples.keys()}"
            )
            for (
                super_rater,
                lp2sys2samples,
            ) in super_rater2lp2sys2samples.items():
                autoeval2lp2sys2samples_with_automatic_evaluations[super_rater] = (
                    lp2sys2samples
                )
        else:
            logger.info(
                "Discarding super-rater annotations because you are using more than one test set."
            )

        # Merge metrics based on information in metrics_to_merge_info
        for merge_info in metrics_to_merge_info:
            merge_autoevaluators_based_on_info(
                merge_info,
                autoeval2lp2sys2samples_with_automatic_evaluations,
            )

        test_set2lp2sys2samples_with_human_evaluations[test_set] = (
            lp2sys2samples_with_human_evaluations
        )
        for autoeval_entry in autoeval2lp2sys2samples_with_automatic_evaluations:
            autoeval2test_set2lp2sys2samples_with_automatic_evaluations[autoeval_entry][
                test_set
            ] = autoeval2lp2sys2samples_with_automatic_evaluations[autoeval_entry]

    # Collapse the test_set dimension to merge language pairs that are identical across test sets
    lp2sys2samples_with_human_evaluations = defaultdict(lambda: defaultdict(list))
    autoeval2lp2sys2samples_with_automatic_evaluations = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for (
        test_set,
        _lp2sys2samples_with_human_evaluations,
    ) in test_set2lp2sys2samples_with_human_evaluations.items():
        for (
            lp,
            _sys2samples_with_human_evaluations,
        ) in _lp2sys2samples_with_human_evaluations.items():
            for sys, samples in _sys2samples_with_human_evaluations.items():
                lp2sys2samples_with_human_evaluations[lp][sys] += samples

                for (
                    autoeval_entry
                ) in autoeval2test_set2lp2sys2samples_with_automatic_evaluations:
                    autoeval2lp2sys2samples_with_automatic_evaluations[autoeval_entry][
                        lp
                    ][
                        sys
                    ] += autoeval2test_set2lp2sys2samples_with_automatic_evaluations[
                        autoeval_entry
                    ][
                        test_set
                    ][
                        lp
                    ][
                        sys
                    ]

    # Parse perturbations argument
    enabled_perturbations = set(args.perturbations) if args.perturbations else []

    logger.info(f"Enabled perturbations: {enabled_perturbations}")

    logger.info(
        f"Total metrics to evaluate: {len(autoeval2lp2sys2samples_with_automatic_evaluations)}"
    )

    metrics_stats = defaultdict(lambda: defaultdict(MetricStats))

    lp2preprocessed_samples_with_human_evaluations = dict()
    lp2sys2preprocessed_samples_with_human_evaluations = defaultdict(lambda: dict())
    all_srcs, all_tgts, all_samples_with_human_evaluations = [], [], []
    lps = list(lp2sys2samples_with_human_evaluations.keys())
    for lp in lps:
        human_errors_before_preprocessing = [
            error
            for sys, sys_samples in lp2sys2samples_with_human_evaluations[lp].items()
            for sample in sys_samples
            for error in sample.human_evaluation.errors
        ]

        original_human_counts = count_errors_by_severity(
            human_errors_before_preprocessing
        )
        if original_human_counts.get(UNKNOWN_SEVERITY, 0) > 0:
            raise RuntimeError(
                f"{original_human_counts.get(UNKNOWN_SEVERITY, 0)} unknown severities found in {lp} for human annotations."
            )

        (
            human_severity_filtered_errors,
            human_category_filtered_errors,
            human_ill_formed_errors,
            human_score_0_errors,
            human_overlapping_errors,
        ) = (0, 0, 0, 0, 0)

        # Preprocess samples with human evaluations
        for sys, sys_samples in lp2sys2samples_with_human_evaluations[lp].items():

            # In en-es, you want to map subcategories to the original MQM category to be able to filter later
            if lp == "en-es":
                mapping = enes_subcategory_to_category_mapping

                for sample in sys_samples:
                    for error in sample.human_evaluation.errors:
                        macro_category = mapping.get(error.category, "")
                        if macro_category:
                            error.category = macro_category + "\\" + error.category

            (
                lp2sys2preprocessed_samples_with_human_evaluations[lp][sys],
                sys_human_severity_filtered_errors,
                sys_human_category_filtered_errors,
                sys_human_ill_formed_errors,
                sys_human_score_0_errors,
                sys_human_overlapping_errors,
            ) = preprocess_samples_with_human_evaluations(
                sys_samples,
                included_severities=args.human_severities,
                included_categories=args.human_categories,
                remove_overlapping_errors=args.remove_overlapping_errors,
                transform_critical_into_major=args.transform_critical_into_major,
            )

            human_severity_filtered_errors += sys_human_severity_filtered_errors
            human_category_filtered_errors += sys_human_category_filtered_errors
            human_ill_formed_errors += sys_human_ill_formed_errors
            human_score_0_errors += sys_human_score_0_errors
            human_overlapping_errors += sys_human_overlapping_errors

        human_errors_after_preprocessing = [
            error
            for sys, sys_samples in lp2sys2preprocessed_samples_with_human_evaluations[
                lp
            ].items()
            for sample in sys_samples
            for error in sample.human_evaluation.errors
        ]

        final_human_counts = count_errors_by_severity(human_errors_after_preprocessing)

        srcs, tgts = (
            [],
            [],
        )

        # TODO: investigate how to shrink or optimize this
        if args.compute_auto_statistics_only_on_the_samples_with_human_errors:
            samples_with_human_evaluations = [
                sample
                for sys, sys_samples in lp2sys2preprocessed_samples_with_human_evaluations[
                    lp
                ].items()
                for sample in sys_samples
                if sample.human_evaluation.errors
            ]
        else:
            samples_with_human_evaluations = [
                sample
                for sys, sys_samples in lp2sys2preprocessed_samples_with_human_evaluations[
                    lp
                ].items()
                for sample in sys_samples
            ]

        for sample in samples_with_human_evaluations:
            srcs.append(sample.src)
            tgts.append(sample.tgt)

        all_srcs.extend(srcs)
        all_tgts.extend(tgts)
        all_samples_with_human_evaluations.extend(samples_with_human_evaluations)

        (
            num_samples_with_no_human_errors,
            num_human_errors,
            human_total_span_length,
        ) = compute_evaluations_stats(
            [sample.human_evaluation for sample in samples_with_human_evaluations]
        )

        metrics_stats[lp][args.gold_rating_key].update(
            len(samples_with_human_evaluations),
            num_human_errors,
            num_samples_with_no_human_errors,
            human_ill_formed_errors,
            0,
            0,
            0,
            human_severity_filtered_errors,
            human_category_filtered_errors,
            original_human_counts,
            final_human_counts,
            human_total_span_length,
            human_score_0_errors,
            human_overlapping_errors,
        )

        lp2preprocessed_samples_with_human_evaluations[lp] = [
            sample
            for sys, sys_samples in lp2sys2preprocessed_samples_with_human_evaluations[
                lp
            ].items()
            for sample in sys_samples
        ]

    # convert my defaultdicts to dicts before parallelization
    autoeval2lp2sys2samples_with_automatic_evaluations = convert_defaultdict_to_dict(
        autoeval2lp2sys2samples_with_automatic_evaluations
    )

    # Prepare arguments for each autoeval
    autoeval_args = [
        (
            autoeval,
            lp2sys2samples_with_automatic_evaluations,
            lp2preprocessed_samples_with_human_evaluations,
            args.human_as_a_metric_rating_keys,
            args.do_not_verify_completeness,
            args.remove_overlapping_errors,
            args.auto_severities,
            args.auto_categories,
            args.transform_critical_into_major,
            args.logging_level,
        )
        for autoeval, lp2sys2samples_with_automatic_evaluations in autoeval2lp2sys2samples_with_automatic_evaluations.items()
    ]

    num_workers = min(
        multiprocessing.cpu_count() - 1,
        len(autoeval_args),
    )

    logger.info(
        f"Preprocessing {len(autoeval_args)} autoevals with {num_workers} workers"
    )

    # PREPROCESS autoevals in parallel or sequentially
    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            preprocessing_results = []
            for i, result in enumerate(
                pool.imap_unordered(preprocess_single_autoeval_wrapper, autoeval_args),
                1,
            ):
                preprocessing_results.append(result)
                logger.info(f"Completed {i}/{len(autoeval_args)}: {result[0]}")
    else:
        # Sequential processing (useful for debugging)
        logger.info("Running in sequential mode (num_workers=1)")
        preprocessing_results = [
            preprocess_single_autoeval_wrapper(args) for args in autoeval_args
        ]

    logger.info("Preprocessing: All auto-evaluators completed!")

    # Merge results back into main data structures
    autoeval2lp2preprocessed_samples_with_automatic_evaluations = dict()
    for (
        autoeval,
        metric_stats,
        lp2preprocessed_samples_with_automatic_evaluations,
    ) in preprocessing_results:
        for lp in lps:
            metrics_stats[lp][autoeval] = metric_stats[lp]
            autoeval2lp2preprocessed_samples_with_automatic_evaluations[autoeval] = (
                lp2preprocessed_samples_with_automatic_evaluations
            )

    autoeval2lp2preprocessed_samples_with_automatic_evaluations = (
        extract_perturbations_from_autoevals(
            autoeval2lp2preprocessed_samples_with_automatic_evaluations,
            enabled_perturbations=enabled_perturbations,
        )
    )

    for (
        autoeval,
        _lp2preprocessed_samples_with_automatic_evaluations,
    ) in autoeval2lp2preprocessed_samples_with_automatic_evaluations.items():
        if not any(
            autoeval.endswith(perturbation) for perturbation in enabled_perturbations
        ):
            continue
        for (
            lp,
            _preprocessed_samples_with_automatic_evaluations,
        ) in _lp2preprocessed_samples_with_automatic_evaluations.items():
            _auto_evaluations = [
                sample.evaluation
                for sample in _preprocessed_samples_with_automatic_evaluations
            ]
            num_samples_with_no_errors, num_total_error_spans, total_span_length = (
                compute_evaluations_stats(_auto_evaluations)
            )
            metrics_stats[lp][autoeval].update(
                len(_auto_evaluations),
                num_total_error_spans,
                num_samples_with_no_errors,
                0,
                0,
                0,
                0,
                0,
                0,
                {},
                {},
                total_span_length,
                0,
                0,
            )

    # Prepare arguments for each autoeval
    autoeval_args = [
        (
            autoeval,
            lps,
            lp2preprocessed_samples_with_automatic_evaluations,
            lp2preprocessed_samples_with_human_evaluations,
            args.severity_penalty,
            args.remove_overlapping_errors,
            args.fix_edge_cases_in_precision,
            args.use_greedy_matching,
            args.logging_level,
        )
        for autoeval, lp2preprocessed_samples_with_automatic_evaluations in autoeval2lp2preprocessed_samples_with_automatic_evaluations.items()
    ]

    num_workers = min(
        multiprocessing.cpu_count() - 1,
        len(autoeval_args),
    )
    logger.info(f"Processing {len(autoeval_args)} autoevals with {num_workers} workers")

    # PROCESS autoevals in parallel or sequentially
    if num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as pool:
            processing_results = []
            for i, result in enumerate(
                pool.imap_unordered(process_single_autoeval_wrapper, autoeval_args),
                1,
            ):
                processing_results.append(result)
                logger.info(f"Completed {i}/{len(autoeval_args)}: {result[0]}")
    else:
        # Sequential processing (useful for debugging)
        logger.info("Running in sequential mode (num_workers=1)")
        processing_results = [
            process_single_autoeval_wrapper(args) for args in autoeval_args
        ]

    logger.info("Processing: All auto-evaluators (with sentinels) completed!")

    metrics: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Metrics]]]]] = dict()
    for (
        autoeval,
        autoeval_metrics,
    ) in processing_results:
        metrics[autoeval] = autoeval_metrics

    results = compute_results_from_metrics(metrics)

    # Print tables for each language pair
    for lp in lps:
        sentinel_counts = compute_sentinel_counts(results[lp])
        print_results_and_stats(
            lp,
            results[lp],
            metrics_stats[lp],
            sentinel_counts,
        )

    global_key = "global"
    global_stats: Dict[str, MetricStats] = aggregate_stats(metrics_stats)
    global_metrics: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Metrics]]]]] = (
        aggregate_metrics(metrics, global_key)
    )
    global_results = compute_results_from_metrics(global_metrics)
    global_counts_sentinels = compute_sentinel_counts(global_results[global_key])

    print_results_and_stats(
        global_key,
        global_results[global_key],
        global_stats,
        global_counts_sentinels,
    )


def read_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the meta-evaluation (at the span-level) for the given metrics"
    )
    parser.add_argument(
        "--test-sets",
        type=str,
        nargs="+",
        default=["wmt24"],
        help="The test sets to run the meta-evaluation on",
    )
    parser.add_argument(
        "--annotation-protocol",
        type=str,
        default="mqm",
        help="The annotation protocols to use for the evaluation, in {esa, mqm} (default: mqm)",
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="The logging level to use (default: INFO)",
    )
    parser.add_argument(
        "--lps",
        type=str,
        nargs="+",
        default=None,
        help="The language pairs to evaluate (default: wmt24_lps/wmt23_lps).",
    )
    parser.add_argument(
        "--gold-rating-key",
        type=str,
        default="mqm.super.1",
    )
    parser.add_argument(
        "--human-as-a-metric-rating-keys",
        type=str,
        nargs="*",
        default=["mqm.super.2", "mqm.super.3"],
    )
    parser.add_argument(
        "--auto-severities",
        type=str,
        nargs="+",
        default=["minor", "major", "critical"],
        help="The severities to consider in the evaluation (default: minor major critical).",
    )
    parser.add_argument(
        "--human-severities",
        type=str,
        nargs="+",
        default=["minor", "major", "critical"],
        help="The severities to consider in the evaluation (default: minor major critical).",
    )
    parser.add_argument(
        "--human-categories",
        type=str,
        nargs="+",
        default="All",
        help="The categories to consider in the human evaluations (default: All).",
    )
    parser.add_argument(
        "--auto-categories",
        type=str,
        nargs="+",
        default="All",
        help="The categories to consider in the automatic evaluations (default: All).",
    )
    parser.add_argument(
        "--do-not-verify-completeness",
        action="store_true",
        help="Do not verify that the automatic evaluations have annotated the FULL dataset. If this is specified, partial annotations are still loaded from the cache and evaluated",
    )

    parser.add_argument(
        "--compute-auto-statistics-only-on-the-samples-with-human-errors",
        action="store_true",
        help="If this flag is used, automatic statistics are computed only on the samples with human errors. As a consequence, total_errors will be the number of automatic errors in the samples with human errors. This is useful for comparing different autoevaluators when we are filtering to include only the samples that contain some specific category of human errors.",
    )

    parser.add_argument(
        "--use-merged-annotations",
        type=bool,
        default=True,
        help="If true, load mqm annotations directly from mqm.merged annotations, rather than from the individual raters.",
    )

    parser.add_argument(
        "--severity-penalty",
        type=float,
        default=0.0,
        help="Penalty on severity mismatches",
    )

    parser.add_argument(
        "--remove-overlapping-errors",
        action="store_true",
        help="Remove overlapping errors both in gold and automatic evaluations",
    )

    parser.add_argument(
        "--perturbations",
        type=str,
        nargs="*",
        default=None,
        choices=list(ALL_PERTURBATIONS) + [[]],
        help=f"Perturbations to apply to evaluators. Valid values: {', '.join(ALL_PERTURBATIONS)}. "
        f"If not specified or empty list, no perturbations are generated. "
        f"Example: --perturbations NO_EXT RAND_REMOVE_05",
    )

    parser.add_argument(
        "--fix-edge-cases-in-precision",
        action="store_true",
        help="Whether to return p=1 only when all tp, fp, and fn are 0, not only when tp and fp are so.",
    )
    parser.add_argument(
        "--fix-wmt25-indices-with-tgt-annotated",
        action="store_true",
        help="Whether to use the correct indices when loading submissions, different from the (wrong) indices used at wmt25",
    )
    parser.add_argument(
        "--transform-critical-into-major",
        type=bool,
        default=True,
        help="Whether to change critical severity into major to not get penalized when severity penalty is used",
    )
    parser.add_argument(
        "--do-not-load-wmt25-submissions",
        action="store_true",
    )
    parser.add_argument(
        "--use-greedy-matching",
        action="store_true",
        help="Whether to use greedy matching instead of optimal matching when computing the metrics.",
    )

    return parser


if __name__ == "__main__":
    main()
