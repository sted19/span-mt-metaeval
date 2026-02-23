# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
import logging
from typing import Literal
import scipy

from mt_metrics_eval import data as mt_metrics_eval_data, stats as mt_metrics_eval_stats

from mt_evaluation.meta_evaluation.utils import (
    get_autoeval2scores,
    compute_correlation,
)
from mt_evaluation.meta_evaluation.metrics_to_evaluate import (
    metrics_to_evaluate_info_wmt24,
)
from mt_evaluation.utils import setup_logging


logger = logging.getLogger(__name__)


def main():
    parser = read_arguments()
    args = parser.parse_args()

    setup_logging(args.logging_level)

    test_set: Literal["wmt23", "wmt24"] = "wmt24"
    lps = args.lps
    evalsets = {
        (test_set, lp): mt_metrics_eval_data.EvalSet(test_set, lp, True) for lp in lps
    }

    metric_name2scores = get_autoeval2scores(
        test_set=test_set, metrics_to_evaluate_info=metrics_to_evaluate_info_wmt24
    )

    for (test_set, lp), evalset in evalsets.items():

        metric_name2seg_sys_scores = {
            metric_name: metric_name2scores[metric_name][lp]
            for metric_name in metric_name2scores
        }

        # I'm still passing evalset.std_ref as ref_to_use. This way, I don't evaluate the translations of the
        # best reference, but, in exchange, I can compare with the results obtained at the WMT24 metrics shared task
        pce = compute_correlation(
            corr_fcn=mt_metrics_eval_stats.PairwiseConfidenceError,
            evalset=evalset,
            ref_to_use=evalset.std_ref,
            include_outliers=False,
            include_human=not args.exclude_human,
            gold_name="mqm",
            primary_metrics=True,
            level="seg",
            grouping_criteria=None,
            metric_name2seg_sys_scores=metric_name2seg_sys_scores,
        )

        kwt = compute_correlation(
            corr_fcn=mt_metrics_eval_stats.KendallWithTiesOpt,
            evalset=evalset,
            ref_to_use=evalset.std_ref,
            include_outliers=False,
            include_human=not args.exclude_human,
            gold_name="mqm",
            primary_metrics=True,
            level="seg",
            grouping_criteria="item",
            metric_name2seg_sys_scores=metric_name2seg_sys_scores,
        )

        seg_level_pearson_with_item_grouping = compute_correlation(
            corr_fcn=scipy.stats.pearsonr,
            evalset=evalset,
            ref_to_use=evalset.std_ref,
            include_outliers=False,
            include_human=not args.exclude_human,
            gold_name="mqm",
            primary_metrics=True,
            level="seg",
            grouping_criteria="item",
            metric_name2seg_sys_scores=metric_name2seg_sys_scores,
        )

        # print results
        print(f"Test Set: {test_set}, LP: {lp}")
        print(f"PCE: {pce:.3f}")
        print(f"KWT: {kwt:.3f}")
        print(
            f"Seg Level Pearson with Item Grouping: {seg_level_pearson_with_item_grouping:.3f}"
        )


def read_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the meta-evaluation (sys, seg, and span-level) for the given metrics"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="The number of resampling runs for statistical significance. Default: 0.",
    )
    parser.add_argument(
        "--exclude-human",
        action="store_true",
        help="Whether to include 'human' systems (i.e., reference translations) among the evaluated systems.",
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
        default=["en-de", "en-es", "ja-zh"],
        help="The language pairs to evaluate (default: en-de en-es ja-zh).",
    )

    return parser


if __name__ == "__main__":
    main()
