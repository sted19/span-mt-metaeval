# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from argparse import ArgumentParser
import logging

from mt_metrics_eval import (
    data as mt_metrics_eval_data,
    tasks,
    meta_info as mt_metrics_eval_meta_info,
)
from typing import Dict, List, Literal

from mt_evaluation.core import Sample, Error, AutomaticEvaluation, HumanEvaluation
from mt_evaluation.meta_evaluation.metrics_to_evaluate import (
    metrics_to_evaluate_info_wmt24,
    metrics_to_evaluate_info_wmt23,
    metrics_to_evaluate_info_wmt22,
    metrics_to_evaluate_info_wmt25_mqm,
)
from mt_evaluation.data import (
    get_raters_evaluations,
    get_super_raters_from_raters,
    get_autoeval2lp2sys2samples_with_automatic_evaluations,
)
from mt_evaluation.meta_evaluation.utils import (
    get_autoeval2ref_to_use,
    remove_samples_with_none_human_evaluation,
)
from mt_evaluation.utils import setup_logging
from mt_evaluation.core import (
    all_severities as all_severities,
    wmt22_lps,
    wmt23_lps,
    wmt24_lps,
    wmt25_lps,
    non_translation,
    unintelligible,
)

logger = logging.getLogger(__name__)


# This function is used to deal with ill-formed errors or errors that do not have a score/severity/other. I'm introducing it to keep compatibility with gemba prompts that didn't collect much info
def preprocess_error(error: Error, included_severities: List[str]):
    if error is None:
        return None

    # Ensure we have a category and a severity
    if error.category is None or len(error.category.strip()) == 0:
        raise ValueError(f"Error {error} has no category")
    if error.severity is None or len(error.severity.strip()) == 0:
        raise ValueError(f"Error {error} has no severity")

    # Make source error consistent with the category
    is_source_error = error.is_source_error
    if is_source_error is None:
        if "source error" in error.category:
            is_source_error = True
        else:
            is_source_error = False

    # Assign scores based on category or severity
    score = error.score
    if score is None:
        if (
            non_translation in error.category.lower()
            or unintelligible in error.category.lower()
        ):
            score = -25
        elif "source error" in error.category.lower():
            score = 0.0
            is_source_error = True
        elif "major" in error.severity.lower():
            score = -5.0
        elif "minor" in error.severity.lower():
            score = -1.0
        elif "neutral" in error.severity.lower():
            score = 0.0
        elif "critical" in error.severity.lower():
            score = -10.0
        else:
            logger.error(
                f"Unknown error severity: {error.severity}. This error be removed"
            )
            return None

    # make sure score and category/severity match
    assert (
        (
            score == 0
            and (
                "neutral" in error.severity.lower()
                or "source error" in error.category.lower()
            )
        )
        or (score == -1 and "minor" in error.severity.lower())
        or (score == -5 and "major" in error.severity.lower())
        or (score == -10 and "critical" in error.severity.lower())
        or (
            score == -25
            and non_translation in error.category.lower()
            or unintelligible in error.category.lower()
        )
    ), f"Error {error} has score {score} but severity {error.severity}"

    # Filter errors based on severity
    if error.severity.lower() not in included_severities:
        return None

    return Error(
        span=error.span,
        category=error.category,
        severity=error.severity,
        start=error.start,
        end=error.end,
        is_source_error=is_source_error,
        score=score,
        explanation=error.explanation,
    )


def preprocess_automatic_evaluation(
    evaluation: AutomaticEvaluation,
    included_severities: List[str],
) -> AutomaticEvaluation | None:
    if evaluation is None:
        return evaluation

    new_errors = []
    for error in evaluation.errors:
        new_error = preprocess_error(error, included_severities)
        if new_error is not None:
            new_errors.append(new_error)

    new_score = sum(error.score for error in new_errors)

    return AutomaticEvaluation(
        score=new_score,
        errors=new_errors,
        annotation=evaluation.annotation,
        user_prompt=evaluation.user_prompt,
        system_prompt=evaluation.system_prompt,
        few_shots=evaluation.few_shots,
        cost=evaluation.cost,
        parsing_error=evaluation.parsing_error,
    )


def preprocess_human_evaluation(
    evaluation: HumanEvaluation, included_severities: List[str]
) -> HumanEvaluation | None:
    if evaluation is None:
        return evaluation

    new_errors = []
    for error in evaluation.errors:
        if error.severity.lower() in included_severities:
            new_errors.append(
                Error(
                    span=error.span,
                    category=error.category,
                    severity=error.severity,
                    start=error.start,
                    end=error.end,
                    is_source_error=error.is_source_error,
                    score=error.score,
                    explanation=error.explanation,
                )
            )

    new_score = sum(error.score for error in new_errors)

    return HumanEvaluation(
        score=new_score,
        errors=new_errors,
    )


# NOTE: the number of samples has to stay exactly the same. This function must modify them to apply the desired changes (e.g., removing errors with neutral severity)
def preprocess_samples_with_human_evaluations(
    samples: List[Sample], human_severities: List[str]
) -> List[Sample]:
    new_samples = []

    for sample in samples:
        new_samples.append(
            Sample(
                src=sample.src,
                tgt=sample.tgt,
                src_lang=sample.src_lang,
                tgt_lang=sample.tgt_lang,
                evaluation=sample.evaluation,
                human_evaluation=preprocess_human_evaluation(
                    sample.human_evaluation, human_severities
                ),
            )
        )

    return new_samples


def preprocess_samples_with_automatic_evaluations(
    samples: List[Sample], automatic_severities: List[str]
) -> List[Sample]:
    new_samples = []

    for sample in samples:
        new_samples.append(
            Sample(
                src=sample.src,
                tgt=sample.tgt,
                src_lang=sample.src_lang,
                tgt_lang=sample.tgt_lang,
                evaluation=preprocess_automatic_evaluation(
                    sample.evaluation, automatic_severities
                ),
                human_evaluation=sample.human_evaluation,
            )
        )

    return new_samples


def overwrite_mt_metrics_eval_mqm_scores(
    evs: mt_metrics_eval_data.EvalSet,
    seg_scores: Dict[str, List[float | None]],
    sys_scores: Dict[str, List[float | None]],
):
    """
    NOTE: this function modifies the EvalSet in place. You cannot use the same evalset for other purposes.

    :param evs: the EvalSet whose scores you want to overwrite
    :param seg_scores: a dictionary mapping sys_names to list of scores (or None) in the same for of that used by the EvalSet
    :param sys_scores: a dictionary mapping sys_names to list of scores (or None) in the same for of that used by the EvalSet
    """

    # First, make sure we are overwriting lists of the same length
    orig_seg_mqm_scores = evs._scores["seg"]["mqm"]
    for sys, sys_scores_list in orig_seg_mqm_scores.items():

        # All None lists trigger errors because I don't have them in my computed scores. Also, they don't need to be overwritten
        if all(val is None for val in sys_scores_list):
            continue

        assert len(seg_scores[sys]) == len(
            sys_scores_list
        ), f"Lengths between original and new segment scores differ. len(seg_scores[sys]) = {len(seg_scores[sys])} != len(sys_scores) {len(sys_scores_list)}"

    evs._scores["seg"]["mqm"] = seg_scores
    evs._scores["sys"]["mqm"] = sys_scores


def WMT24OnWMT23(lps: list[str] | None = None, primary=True, k=0, gold=None):
    """Generate the WMT24 task set for WMT23 and associated weight vector."""

    # Not strictly necessary to declare this, because setting human=True will
    # only score human outputs if any are available, but we want to make the
    # human attribute reflect what actually got used, and also want to avoid
    # having to load the EvalSets at this point to get this info automatically.

    lps_with_multiple_refs = {"en-he", "he-en"}

    def Add(lp, level, corr_fcn, human, gold, **kw_args):
        tasks_list.Append(
            tasks.Task(
                "wmt23",
                lp,
                level=level,
                corr_fcn=corr_fcn,
                human=human,
                gold=gold,
                primary=primary,
                k=k,
                **kw_args,
            )
        )

    if lps is None:
        lps = ["en-de", "he-en", "zh-en"]
    lps = sorted(lps)

    tasks_list = tasks.TaskSet()

    # For each language pair: PCE at the system-level.
    for lp in lps:
        human = lp in lps_with_multiple_refs
        Add(
            lp,
            "sys",
            "pce",
            human=human,
            gold=[gold] * len(lps) if gold else None,
        )

    weights = [1] * (len(tasks_list))
    weights = [w / sum(weights) for w in weights]

    return tasks_list, weights


def WMT24_tasks_with_pdp(lps: list[str] | None = None, primary=True, k=0, gold=None):
    """Generate the WMT24 task set associated weight vector."""

    # Not strictly necessary to declare this, because setting human=True will
    # only score human outputs if any are available, but we want to make the
    # human attribute reflect what actually got used, and also want to avoid
    # having to load the EvalSets at this point to get this info automatically.
    lps_with_multiple_refs = {"en-de"}

    def Add(lp, level, corr_fcn, human, gold, **kw_args):
        local_tasks.Append(
            tasks.Task(
                "wmt24",
                lp,
                level=level,
                corr_fcn=corr_fcn,
                human=human,
                gold=gold,
                primary=primary,
                k=k,
                **kw_args,
            )
        )

    if lps is None:
        lps = ["en-de", "en-es", "ja-zh"]
    lps = sorted(lps)

    local_tasks = tasks.TaskSet()

    # For each language pair: PCE at the system-level and accuracy at the
    # segment-level.
    for lp in lps:
        human = lp in lps_with_multiple_refs
        Add(
            lp,
            "sys",
            "pce",
            human=human,
            gold=[gold] * len(lps) if gold else None,
        )
        Add(
            lp,
            "seg",
            "pearson",
            human,
            gold,
            avg_by="item",
            perm_test="pairs",
            corr_fcn_args={"pdp": True},
        )

    weights = [1] * len(local_tasks)
    weights = [w / sum(weights) for w in weights]

    return local_tasks, weights


def my_tasks(lps: list[str] | None, primary=True, k=0, gold=None):

    def Add(lp, level, corr_fcn, human, gold, **kw_args):
        tasks_list.Append(
            tasks.Task(
                "wmt24",
                lp,
                level=level,
                corr_fcn=corr_fcn,
                human=human,
                gold=gold,
                primary=primary,
                k=k,
                **kw_args,
            )
        )

    tasks_list = tasks.TaskSet()

    # NOTE: this is false! I'm putting it here just to make it work. Check correctness.
    lps_with_multiple_refs = {"en-de", "en-es"}

    for lp in lps:
        human = lp in lps_with_multiple_refs
        Add(
            lp,
            "sys",
            "accuracy",  # this should be Pairwise Accuracy and be across all lps at the same time
            human=human,
            gold=[gold] * len(lps) if gold else None,
        )

    weights = [1] * (len(tasks_list))
    weights = [w / sum(weights) for w in weights]

    return tasks_list, weights


def compute_final_wmt_ranking(
    test_set: Literal["wmt23", "wmt24"],
    metric_name2lp_samples: Dict[str, Dict[str, Dict[str, List[Sample]]]],
    lp2sys2samples_human: Dict[str, Dict[str, List[Sample]]],
    metric_name2ref_to_use: Dict[str, Dict[str, str]],
    k: int,
    pce: bool,
    human_severities: List[str],
    auto_severities: List[str],
    use_my_tasks: bool,
    use_pdp_instead_of_acceq: bool,
) -> None:
    """Compute the final WMT ranking.

    Args:
        test_set (Literal["wmt23", "wmt24"]): Name of the WMT test set to use. Allowed values: 'wmt23', 'wmt24'.
        metric_name2lp_samples (Dict[str, Dict[str, Dict[str, List[float]]]]): Dictionary with the samples for each metric, lp, system_name.
        lp2sys2samples_human
        metric_name2ref_to_use (Dict[str, Dict[str, str]]): Dictionary with the used refs.
        k (int): The number of resampling runs for statistical significance.
        pce (bool): Whether to run only the PCE System-Level Meta-Evaluation measure.
        human_severities (List[str]): Filter human errors not included in the provided severities
        auto_severities (List[str]): Filter automatic errors not included in the provided severities
        use_my_tasks (bool): Whether to use My Tasks instead of WMT24 ones.
        use_pdp_instead_of_acceq (bool): Whether to use PDP instead of ACCEQ.
    """

    lps = wmt24_lps if test_set == "wmt24" else wmt23_lps
    evs_dict = {
        (test_set, lp): mt_metrics_eval_data.EvalSet(test_set, lp, True) for lp in lps
    }

    for lp in lps:
        evs = evs_dict[(test_set, lp)]

        sys2samples_human = lp2sys2samples_human[lp]

        """
        # Overwrite gold scores
        mqm_seg_scores, mqm_sys_scores = {}, {}
        for sys_name, sys_samples in sys2samples_human.items():
            preprocessed_sys_samples = preprocess_samples_with_human_evaluations(
                sys_samples, human_severities
            )

            mqm_seg_level_sys_scores = [
                sample.human_evaluation.score if sample.human_evaluation else None
                for sample in preprocessed_sys_samples
            ]
            non_none_mqm_seg_level_sys_scores = [
                score for score in mqm_seg_level_sys_scores if score is not None
            ]
            mqm_sys_level_sys_scores = [
                sum(non_none_mqm_seg_level_sys_scores)
                / len(non_none_mqm_seg_level_sys_scores)
            ]

            mqm_seg_scores[sys_name] = mqm_seg_level_sys_scores
            mqm_sys_scores[sys_name] = mqm_sys_level_sys_scores

        overwrite_mt_metrics_eval_mqm_scores(evs, mqm_seg_scores, mqm_sys_scores)
        """

        # collect metric scores
        for metric_name, lp2samples in metric_name2lp_samples.items():
            refs = {metric_name2ref_to_use[metric_name][lp]}
            refs = refs if refs != {"src"} else set()

            lp_samples = lp2samples[lp]

            # The number of samples has to stay exactly the same
            preprocessed_lp_samples = {
                sys: preprocess_samples_with_automatic_evaluations(
                    sys_samples, auto_severities
                )
                for sys, sys_samples in lp_samples.items()
            }

            seg_scores, sys_scores = {}, {}
            for sys, sys_samples in preprocessed_lp_samples.items():
                seg_scores[sys] = [
                    sample.evaluation.score if sample.evaluation else 1000
                    for sample in sys_samples
                ]

                if all(score == 1000 for score in seg_scores[sys]):
                    sys_scores[sys] = [1000]
                else:
                    non_none_seg_sys_scores = [
                        score for score in seg_scores[sys] if score != 1000
                    ]
                    sys_scores[sys] = [
                        sum(non_none_seg_sys_scores) / len(non_none_seg_sys_scores)
                    ]

            evs.AddMetric(
                metric_name, refs, "seg", seg_scores, replace=True, repair=True
            )
            evs.AddMetric(
                metric_name, refs, "sys", sys_scores, replace=True, repair=True
            )

    for evs in evs_dict.values():
        evs.SetPrimaryMetrics(evs.primary_metrics | set(metric_name2lp_samples))

    if test_set == "wmt24":
        if use_my_tasks:
            wmt_tasks, wts = my_tasks(wmt24_lps, k=k)
            baselines_metainfo = mt_metrics_eval_meta_info.WMT24
        elif use_pdp_instead_of_acceq:
            wmt_tasks, wts = WMT24_tasks_with_pdp(wmt24_lps, k=k)
            baselines_metainfo = mt_metrics_eval_meta_info.WMT24
        else:
            wmt_tasks, wts = tasks.WMT24(wmt24_lps, k=k)
            baselines_metainfo = mt_metrics_eval_meta_info.WMT24

    elif test_set == "wmt23":
        wmt_tasks, wts = (
            WMT24OnWMT23(wmt23_lps, k=k) if pce else tasks.WMT23(wmt23_lps, k=k)
        )
        baselines_metainfo = mt_metrics_eval_meta_info.WMT23

    new_results = wmt_tasks.Run(eval_set_dict=evs_dict)
    matrix = None
    if k > 0:
        avg_corrs, matrix = new_results.AverageCorrMatrix(wts)
    else:
        avg_corrs = new_results.AverageCorrs(wts)

    table = new_results.Table(
        metrics=list(avg_corrs),
        initial_column=avg_corrs,
        initial_column_header="avg-corr",
        attr_list=["lang", "level", "corr_fcn"],
        nicknames={"KendallWithTiesOpt": "acc-t"},
        fmt="text",
        baselines_metainfo=baselines_metainfo,
    )
    print(table)

    print("\n\n\n")

    if k > 0:
        # Print the p-value matrix for the pairwise comparisons used to assign significance clusters.
        print(tasks.MatrixString(avg_corrs, matrix, probs=True))
        print("\n\n\n")


def main() -> None:
    """Command to compute the final WMT-23/24 ranking."""
    parser = read_arguments()
    args = parser.parse_args()

    test_set = args.test_set

    test_set2lps = {
        "wmt22": wmt22_lps,
        "wmt23": wmt23_lps,
        "wmt24": wmt24_lps,
        "wmt25": wmt25_lps,
    }
    lps = test_set2lps[test_set]
    lps = args.lps if args.lps is not None else lps

    super_raters = (
        [args.gold_rating_key]
        if test_set == "wmt24" or test_set == "wmt25"
        else [args.gold_rating_key] + args.human_as_a_metric_rating_keys
    )

    setup_logging(args.logging_level)

    rater2lp2sys2samples = get_raters_evaluations(
        args.test_set,
        lps,
        use_merged_annotations=True,
    )

    super_rater2lp2sys2samples = get_super_raters_from_raters(
        rater2lp2sys2samples, super_raters
    )

    lp2sys2samples_with_human_evaluations = {
        lp: super_rater2lp2sys2samples[args.gold_rating_key][lp] for lp in lps
    }
    gold_lp2sys2samples = super_rater2lp2sys2samples.pop(args.gold_rating_key)

    test_set2metrics_to_evaluate = {
        "wmt22": metrics_to_evaluate_info_wmt22,
        "wmt23": metrics_to_evaluate_info_wmt23,
        "wmt24": metrics_to_evaluate_info_wmt24,
        "wmt25": metrics_to_evaluate_info_wmt25_mqm,
    }
    autoeval2lp2sys2samples_with_automatic_evaluations = (
        get_autoeval2lp2sys2samples_with_automatic_evaluations(
            lp2sys2samples=lp2sys2samples_with_human_evaluations,
            metrics_to_evaluate_info=test_set2metrics_to_evaluate[test_set],
            do_not_verify_completeness=False,
        )
    )
    autoeval2ref_to_use = get_autoeval2ref_to_use(
        test_set=args.test_set,
        lps=lps,
        metrics_to_evaluate_info=metrics_to_evaluate_info_wmt24,
    )

    compute_final_wmt_ranking(
        args.test_set,
        autoeval2lp2sys2samples_with_automatic_evaluations,
        gold_lp2sys2samples,
        autoeval2ref_to_use,
        args.k,
        args.pce,
        human_severities=args.human_severities,
        auto_severities=args.auto_severities,
        use_my_tasks=args.use_my_tasks,
        use_pdp_instead_of_acceq=args.use_pdp_instead_of_acceq,
    )


def read_arguments() -> ArgumentParser:
    parser = ArgumentParser(description="Command to compute the WMT final ranking.")
    parser.add_argument(
        "--test-set",
        type=str,
        choices=["wmt23", "wmt24"],
        default="wmt24",
        help="Name of the WMT test set to use. Allowed values: 'wmt23', 'wmt24'. Default: 'wmt24'.",
    )
    parser.add_argument(
        "--lps",
        type=str,
        nargs="+",
        default=None,
        help="The language pairs to evaluate (default: wmt24_lps/wmt23_lps).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="The number of resampling runs for statistical significance. Default: 1000.",
    )
    parser.add_argument(
        "--pce",
        action="store_true",
        help="Whether to run only the PCE System-Level Meta-Evaluation measure.",
    )
    parser.add_argument(
        "--human-severities",
        type=str,
        nargs="+",
        default=all_severities,
        help="Filter human errors with severity not included in the provided severities",
    )
    parser.add_argument(
        "--auto-severities",
        type=str,
        nargs="+",
        default=all_severities,
        help="Filter automatic errors with severity not included in the provided severities",
    )
    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="The logging level to use (default: INFO)",
    )

    parser.add_argument(
        "--use-my-tasks",
        action="store_true",
        help="Whether to use My Tasks instead of WMT Tasks.",
        default=False,
    )
    parser.add_argument(
        "--use-pdp-instead-of-acceq",
        action="store_true",
        help="Whether to use PDP in the place of acc_eq",
        default=False,
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

    return parser


if __name__ == "__main__":
    main()
