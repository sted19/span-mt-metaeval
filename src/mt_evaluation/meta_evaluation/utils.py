# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import Literal, Dict, List, Tuple, Callable
from dataclasses import asdict
import logging
from tabulate import tabulate
from collections import defaultdict

from mt_metrics_eval import (
    data as mt_metrics_eval_data,
    stats as mt_metrics_eval_stats,
)

from mt_evaluation.meta_evaluation import (
    wmt23_lps,
    wmt24_lps,
    UNKNOWN_SEVERITY,
    MetricResults,
    MetricStats,
    SentinelCounts,
    METRIC_TYPES,
)
from mt_evaluation.meta_evaluation.span_level.perturbations import (
    PERTURBATION_EXT_ONLY,
    PERTURBATION_REMOVE_ERRORS_IN_SAMPLES_WITH_1,
)
from mt_evaluation.data import MTEvaluationCache, lang_code2lang, get_cache_dir
from mt_evaluation.utils import get_metric_display_name
from mt_evaluation.core import Sample
from mt_evaluation.utils import is_all_nones


logger = logging.getLogger(__name__)


def remove_samples_with_none_human_evaluation(
    metric_name2lp2sys2samples_with_automatic_evaluations: Dict[
        str, Dict[str, Dict[str, List[Sample]]]
    ],
    lp2sys2samples_with_human_evaluations: Dict[str, Dict[str, List[Sample]]],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, List[Sample]]]], Dict[str, Dict[str, List[Sample]]]
]:
    """
    Given a dictionary containing automatic evaluations of the form
    {
        metric_name: {
            lp : {
                mt_system: {
                    samples: List[Sample]
                }
            }
        }
    }
    and a similar dictionary containing human evaluations (only lacking the layer of metric_name), return two
    dictionaries of the same form, where all samples corresponding to None human evaluations (i.e., samples that
    have not been human-evaluated at the corresponding WMT edition) have been removed.

    List of samples remain aligned.
    """

    valid_samples_with_automatic_evaluations = {}
    for (
        metric_name,
        lp2sys2samples_with_auto_evaluations,
    ) in metric_name2lp2sys2samples_with_automatic_evaluations.items():
        valid_samples_with_automatic_evaluations[metric_name] = {}
        for (
            lp,
            sys2samples_with_auto_evaluations,
        ) in lp2sys2samples_with_auto_evaluations.items():
            valid_samples_with_automatic_evaluations[metric_name][lp] = {}
            for (
                sys,
                samples_with_auto_evaluations,
            ) in sys2samples_with_auto_evaluations.items():
                samples_with_human_evaluations = lp2sys2samples_with_human_evaluations[
                    lp
                ][sys]

                assert len(samples_with_human_evaluations) == len(
                    samples_with_auto_evaluations
                )
                assert all(
                    sample1.get_input_hash() == sample2.get_input_hash()
                    for sample1, sample2 in zip(
                        samples_with_auto_evaluations, samples_with_human_evaluations
                    )
                )

                valid_samples_with_automatic_evaluations[metric_name][lp][sys] = [
                    sample
                    for i, sample in enumerate(samples_with_auto_evaluations)
                    if samples_with_human_evaluations[i].human_evaluation is not None
                ]

    valid_samples_with_human_evaluations = {
        lp: {
            sys: [
                sample for sample in sys_samples if sample.human_evaluation is not None
            ]
            for sys, sys_samples in lp2sys2samples_with_human_evaluations[lp].items()
        }
        for lp in lp2sys2samples_with_human_evaluations
    }

    return (
        valid_samples_with_automatic_evaluations,
        valid_samples_with_human_evaluations,
    )


class MetricLogFilter(logging.Filter):
    def __init__(self, excluded_metrics=None):
        super().__init__()
        self.excluded_metrics = excluded_metrics or []
        self.current_metric = None

    def set_current_metric(self, metric_name):
        self.current_metric = metric_name

    def filter(self, record):
        if self.current_metric:
            return not any(
                excluded in self.current_metric for excluded in self.excluded_metrics
            )
        return True


def compute_correlation(
    corr_fcn: Callable,
    evalset: mt_metrics_eval_data.EvalSet,
    ref_to_use: str,
    include_outliers: bool,
    include_human: bool,
    gold_name: str,
    primary_metrics: bool,
    level: str,
    grouping_criteria: str | None,
    metric_name2seg_sys_scores: Dict[str, Dict[str, Dict[str, List[float | None]]]],
) -> float:
    """
    Function to compute the correlation between metric scores and human ratings provided by WMT.

    Args:
        corr_fcn (Callable): A correlation object from mt-metrics-eval (e.g., mt_metrics_eval_stats.PairwiseConfidenceError)
        evalset (data.EvalSet): A WMT EvalSet
        ref_to_use (str): Human reference to use in the evaluation.
        include_outliers (bool): Whether to include outlier systems.
        include_human (bool): Whether to include 'human' systems (i.e., reference translations) among the evaluated systems.
        gold_name (str): Type of human rating used as gold (e.g., 'mqm')
        primary_metrics (bool): Whether to compare only metrics that have been designated as primary submissions.
        level (str): The level of analysis (granularity), either 'seg' (segment) or 'sys' (system).
        grouping_criteria (str | None): The grouping criteria for metrics comparison.
        metric_name2seg_sys_scores (Dict[str, Dict[str, Dict[str, List[float | None ]]]]): Dictionary from 'seg', 'sys' to, sys_names,
            to metric scores.

    Returns:
        float: The correlation score for the specified `mt_metric`.
    """
    assert level in {"seg", "sys"}, "level must be either 'seg' or 'sys'."
    assert (
        len(list(metric_name2seg_sys_scores.keys())) == 1
    ), "Only one metric should be provided in `metric_name2seg_sys_scores`."

    extern_metrics = {
        metric_name: metric_name2seg_sys_scores[metric_name][level]
        for metric_name in metric_name2seg_sys_scores
    }

    # Get correlations for the given parameters
    metric2corr: Dict[str, mt_metrics_eval_stats.Correlation] = (
        mt_metrics_eval_data.GetCorrelations(
            evs=evalset,
            level=level,
            main_refs={ref_to_use},
            close_refs=set(),
            include_human=include_human,
            include_outliers=include_outliers,
            gold_name=gold_name,
            primary_metrics=primary_metrics,
            domain=None,
            extern_metrics=extern_metrics,
        )
    )

    if corr_fcn == mt_metrics_eval_stats.KendallWithTiesOpt:
        (
            metric2corrs_and_ranks,
            sig_matrix,
            draws_index,
            draws_list,
        ) = mt_metrics_eval_data.CompareMetrics(
            metric_corrs=metric2corr,
            corr_fcn=corr_fcn,
            average_by=grouping_criteria,
            k=0,
            perm_test="pairs",
            sample_rate=1.0,
        )

    elif corr_fcn == mt_metrics_eval_stats.PairwiseConfidenceError:
        (
            metric2corrs_and_ranks,
            sig_matrix,
            draws_index,
            draws_list,
        ) = mt_metrics_eval_data.CompareMetricsWithPairwiseConfidenceError(
            metric_corrs=metric2corr,
            k=0,
        )

    else:
        (
            metric2corrs_and_ranks,
            sig_matrix,
            draws_index,
            draws_list,
        ) = mt_metrics_eval_data.CompareMetrics(
            metric_corrs=metric2corr,
            corr_fcn=corr_fcn,
            average_by=grouping_criteria,
            k=0,
            perm_test="pairs",
        )

    metric_name = list(metric_name2seg_sys_scores.keys())[0]

    # Ensure the requested mt_metric exists in the correlations
    assert (
        metric_name in metric2corrs_and_ranks
    ), f"MT Metric '{metric_name}' not found in computed correlations."

    return metric2corrs_and_ranks[metric_name][0]


def get_autoeval2ref_to_use(
    test_set: Literal["wmt23", "wmt24"],
    lps: List[str],
    metrics_to_evaluate_info: List[Dict],
) -> Dict[str, Dict[str, str]]:
    """Return a dictionary with the reference to use for each autoeval, for each language pair.

    Args:
        test_set (Literal["wmt23", "wmt24"]): Name of the WMT test set to use. Allowed values: 'wmt23', 'wmt24'.
        lps (List[str]): List of language pairs to use.
        metrics_to_evaluate_info: List of dictionaries containing the information of the metrics to evaluate, as
            follows: {
                    "autoeval": "gemba-mqm",
                    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "outputs_path": Path("outputs"),
                }

    Returns:
        Dict[str, Dict[str, str]]: Dictionary with the reference to use for each autoeval and language pair.
    """

    autoeval2ref_to_use = dict()

    for metric_to_evaluate in metrics_to_evaluate_info:

        reference_free = metric_to_evaluate["reference_free"]

        metric_name = get_metric_display_name(
            metric_to_evaluate["autoeval"],
            metric_to_evaluate["model"],
            metric_to_evaluate["run_specific_info"],
        )

        if reference_free:
            autoeval2ref_to_use[metric_name] = {
                lps[0]: "src",
                lps[1]: "src",
                lps[2]: "src",
            }
        else:
            raise NotImplementedError(
                "Only reference-free metrics are supported at the moment."
            )

    return autoeval2ref_to_use


def get_autoeval2scores(
    test_set: Literal["wmt23", "wmt24"],
    metrics_to_evaluate_info: List[Dict],
) -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:
    """Read the input files and return dictionary with the scores for each metric and a dictionary with the used refs.

    Args:
        test_set (Literal["wmt23", "wmt24"]): Name of the WMT test set to use. Allowed values: 'wmt23', 'wmt24'.
        metrics_to_evaluate_info: List of dictionaries containing the information of the metrics to evaluate, as
            follows: {
                    "autoeval": "gemba-mqm",
                    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "outputs_path": Path("outputs"),
                }

    Returns:
        Dict[str, Dict[str, Dict[str, List[float]]]]: Dict containing metric scores for each metric, language pair, level, and system
    """

    autoeval2lp_scores = dict()
    lps = wmt24_lps if test_set == "wmt24" else wmt23_lps

    for metric_to_evaluate in metrics_to_evaluate_info:

        metric_name = get_metric_display_name(
            metric_to_evaluate["autoeval"],
            metric_to_evaluate["model"],
            metric_to_evaluate["run_specific_info"],
        )

        """
        Here, I retrieve from the cache the scores assigned by my autoevals to the (src, tgt, src_lang, tgt_lang 
        contained in the test set
        """

        cache_dir = get_cache_dir(
            metric_to_evaluate["autoeval"],
            metric_to_evaluate["model"],
            metric_to_evaluate["outputs_path"],
            metric_to_evaluate["run_specific_info"],
        )
        cache_file = cache_dir / "cache.jsonl"
        cache = MTEvaluationCache(str(cache_file))

        evalsets = {lp: mt_metrics_eval_data.EvalSet(test_set, lp, True) for lp in lps}
        autoeval2lp_scores[metric_name] = dict()
        for lp in lps:
            autoeval2lp_scores[metric_name][lp] = dict()
            autoeval2lp_scores[metric_name][lp]["seg"] = dict()
            autoeval2lp_scores[metric_name][lp]["sys"] = dict()
            evalset = evalsets[lp]
            src_lang = lang_code2lang.get(evalset.src_lang)
            tgt_lang = lang_code2lang.get(evalset.tgt_lang)

            if src_lang is None or tgt_lang is None:
                raise ValueError(
                    f"Language code not found: {evalset.src_lang} or {evalset.tgt_lang}"
                )

            evalset_mqm_annotated_systems = (
                [
                    sys
                    for sys, scores in evalset.Scores("sys", "mqm").items()
                    if not is_all_nones(scores)
                ]
                if evalset.Scores("sys", "mqm") is not None
                else []
            )

            srcs = evalset.src
            for sys in evalset_mqm_annotated_systems:

                autoeval2lp_scores[metric_name][lp]["seg"][sys] = []

                tgts = evalset.sys_outputs[sys]
                evalset_data = [
                    Sample(src, tgt, src_lang, tgt_lang)
                    for src, tgt in zip(srcs, tgts)
                    # if not src.startswith("CANARY")
                ]

                for sample in evalset_data:
                    if cache.is_evaluated(sample):
                        score = cache.get_evaluation(sample).score
                        autoeval2lp_scores[metric_name][lp]["seg"][sys].append(score)
                    else:
                        raise ValueError(
                            f"Sample {sample} not found in cache. Please run the evaluation first."
                        )

                autoeval2lp_scores[metric_name][lp]["sys"][sys] = [
                    sum(autoeval2lp_scores[metric_name][lp]["seg"][sys])
                    / len(autoeval2lp_scores[metric_name][lp]["seg"][sys])
                ]

    return autoeval2lp_scores


def aggregate_stats(
    metrics_stats: Dict[str, Dict[str, MetricStats]],
) -> Dict[str, MetricStats]:
    """

    :param metrics_stats: dictionary from language pairs to another dictionary of the form {metric name: Metric Stats}
    :param key: the key of the dictionary where to save global stats

    adds a field of name 'key' to `metrics_stats`, saving global statistics there
    """
    metric_names = list(metrics_stats[next(iter(metrics_stats))].keys())
    global_stats = defaultdict(MetricStats)

    for metric_name in metric_names:
        for lp in metrics_stats:
            metric_stats = metrics_stats[lp][metric_name]

            global_stats[metric_name].update(**asdict(metric_stats))

    return global_stats


def determine_perturbation_type(metric_name: str) -> Tuple[bool, bool, bool, bool]:
    normal, ext_only, no_ext, remove_all = False, False, False, False
    if metric_name.endswith(PERTURBATION_EXT_ONLY):
        ext_only = True
    elif metric_name.endswith(PERTURBATION_REMOVE_ERRORS_IN_SAMPLES_WITH_1):
        remove_all = True
    else:
        normal = True

    return normal, ext_only, no_ext, remove_all


def compute_sentinel_counts(
    metrics_results: Dict[str, Dict[str, Dict[str, MetricResults]]],
) -> Dict[str, Dict[str, SentinelCounts]]:

    sentinel_counts = defaultdict(lambda: defaultdict(SentinelCounts))

    for aggr_type, match_type2metric_results in metrics_results.items():
        for match_type, results in match_type2metric_results.items():
            for metric_type in METRIC_TYPES:

                num_ext_only_greater_than_normal: int = 0
                num_ext_only_smaller_or_equal_than_normal: int = 0
                num_no_ext_greater_than_normal: int = 0
                num_no_ext_smaller_or_equal_than_normal: int = 0
                num_remove_all_1_greater_than_normal: int = 0
                num_remove_all_1_smaller_or_equal_than_normal: int = 0

                for i, (metric_name_i, metric_results_i) in enumerate(
                    list(results.items())[:-1]
                ):
                    for j, (metric_name_j, metric_results_j) in enumerate(
                        list(results.items())[i + 1 :]
                    ):

                        # I'm only interested in comparing normal variants with sentinels of THE SAME auto-evaluator. So, the metric names should match
                        if not metric_name_i.startswith(
                            metric_name_j
                        ) and not metric_name_j.startswith(metric_name_i):
                            continue

                        normal_i, ext_only_i, no_ext_i, remove_all_i = (
                            determine_perturbation_type(metric_name_i)
                        )
                        normal_j, ext_only_j, no_ext_j, remove_all_j = (
                            determine_perturbation_type(metric_name_j)
                        )

                        if normal_i and ext_only_j:
                            if (
                                metric_results_j.results[metric_type].f1
                                > metric_results_i.results[metric_type].f1
                            ):
                                num_ext_only_greater_than_normal += 1
                            else:
                                num_ext_only_smaller_or_equal_than_normal += 1
                        elif normal_j and ext_only_i:
                            if (
                                metric_results_i.results[metric_type].f1
                                > metric_results_j.results[metric_type].f1
                            ):
                                num_ext_only_greater_than_normal += 1
                            else:
                                num_ext_only_smaller_or_equal_than_normal += 1
                        if normal_i and no_ext_j:
                            if (
                                metric_results_j.results[metric_type].f1
                                > metric_results_i.results[metric_type].f1
                            ):
                                num_no_ext_greater_than_normal += 1
                            else:
                                num_no_ext_smaller_or_equal_than_normal += 1
                        elif normal_j and no_ext_i:
                            if (
                                metric_results_i.results[metric_type].f1
                                > metric_results_j.results[metric_type].f1
                            ):
                                num_no_ext_greater_than_normal += 1
                            else:
                                num_no_ext_smaller_or_equal_than_normal += 1
                        elif normal_i and remove_all_j:
                            if (
                                metric_results_j.results[metric_type].f1
                                > metric_results_i.results[metric_type].f1
                            ):
                                num_remove_all_1_greater_than_normal += 1
                            else:
                                num_remove_all_1_smaller_or_equal_than_normal += 1
                        elif normal_j and remove_all_i:
                            if (
                                metric_results_i.results[metric_type].f1
                                > metric_results_j.results[metric_type].f1
                            ):
                                num_remove_all_1_greater_than_normal += 1
                            else:
                                num_remove_all_1_smaller_or_equal_than_normal += 1
                        else:
                            pass

                sentinel_counts[aggr_type][match_type].update(
                    metric_type,
                    num_ext_only_greater_than_normal,
                    num_ext_only_smaller_or_equal_than_normal,
                    num_no_ext_greater_than_normal,
                    num_no_ext_smaller_or_equal_than_normal,
                    num_remove_all_1_greater_than_normal,
                    num_remove_all_1_smaller_or_equal_than_normal,
                )

    return sentinel_counts


def print_results(
    metrics_results_matching: Dict[str, MetricResults],
    metrics_results_no_matching: Dict[str, MetricResults],
    metrics_results_matching_macro: Dict[str, MetricResults],
    metrics_results_no_matching_macro: Dict[str, MetricResults],
) -> List[str]:
    if not metrics_results_matching or not metrics_results_no_matching:
        print("No results available for this language pair.")
        return []

    # sort results according to f1_pc (micro-averaged with matching)
    sorted_results = sorted(
        metrics_results_matching.items(),
        key=lambda x: x[1].results["Character\nProportion"].f1,
        reverse=True,
    )

    # Table 1: Micro-averaged metrics
    print("MICRO-AVERAGED METRICS:")
    print("-" * 50)

    for i, metrics_results in enumerate(
        [metrics_results_matching, metrics_results_no_matching]
    ):

        print("GREEDY BIPARTITE MATCHING") if i == 0 else print("NO MATCHING")
        print("-" * 50)

        micro_table_data = []
        micro_headers = ["Metric"]

        for metric_type in METRIC_TYPES:
            micro_headers.extend(
                [
                    metric_type + "\nPrecision",
                    metric_type + "\nRecall",
                    metric_type + "\nF1",
                ]
            )

        for metric_name, _ in sorted_results:
            metric_results = metrics_results[metric_name]
            row = [metric_name]
            for metric_type in METRIC_TYPES:
                row.extend(
                    [
                        f"{100*metric_results.results[metric_type].precision:.2f}",
                        f"{100*metric_results.results[metric_type].recall:.2f}",
                        f"{100*metric_results.results[metric_type].f1:.2f}",
                    ]
                )
            micro_table_data.append(row)

        print(
            tabulate(
                micro_table_data,
                headers=micro_headers,
                tablefmt="grid",
                stralign="center",
                numalign="center",
            )
        )
        print()

    # Table 2: Macro-averaged metrics
    print("MACRO-AVERAGED METRICS:")
    print("-" * 50)

    for i, metrics_results_macro in enumerate(
        [metrics_results_matching_macro, metrics_results_no_matching_macro]
    ):

        print("GREEDY BIPARTITE MATCHING") if i == 0 else print("NO MATCHING")
        print("-" * 50)

        macro_table_data = []
        macro_headers = ["Metric"]

        for metric_type in METRIC_TYPES:
            macro_headers.extend(
                [
                    metric_type + "\nPrecision",
                    metric_type + "\nRecall",
                    metric_type + "\nF1",
                ]
            )

        for metric_name, _ in sorted_results:
            metric_results = metrics_results_macro[metric_name]
            row = [metric_name]
            for metric_type in METRIC_TYPES:
                row.extend(
                    [
                        f"{100*metric_results.results[metric_type].precision:.2f}",
                        f"{100*metric_results.results[metric_type].recall:.2f}",
                        f"{100*metric_results.results[metric_type].f1:.2f}",
                    ]
                )
            macro_table_data.append(row)

        print(
            tabulate(
                macro_table_data,
                headers=macro_headers,
                tablefmt="grid",
                stralign="center",
                numalign="center",
            )
        )
        print()

    return [metric_name for metric_name, metric_results in sorted_results]


def print_stats(
    metrics_stats: Dict[str, MetricStats],
):
    if not metrics_stats:
        print("No stats available for this language pair.")
        return

    # Table 3: General statistics with severity breakdown
    print("GENERAL STATISTICS:")
    print("-" * 50)

    stats_table_data = []
    stats_headers = [
        "Metric",
        "Total\nDetected\nErrors",
        "Remaining\nErrors",
        "Avg\nErrors\nper\nSample",
        "Avg\nSpan\nLength\n(chars)",
        "Ill\nFormed\nErrors",
        "Ill\nFormed\nExtended\nSpans",
        "Errors\nAmbiguous\nMatch",
        "Errors\nAmbiguous\nMatch\nExtended",
        "Overlapping\nErrors",
        "Severity\nFiltered\nErrors",
        "Category\nFiltered\nErrors",
        "Score 0\nErrors",
        "Neutral\n(Orig/Final)",
        "Minor\n(Orig/Final)",
        "Major\n(Orig/Final)",
        "Critical\n(Orig/Final)",
        "Samples\nNo Errors",
        "Total\nSamples",
    ]

    metric_names = list(metrics_stats.keys())

    for metric_name in metric_names:
        metric_stats = metrics_stats[metric_name]
        row = [
            metric_name,
            f"{metric_stats.num_errors + metric_stats.num_removed_errors}",
            f"{metric_stats.num_errors}",
            f"{metric_stats.avg_errors_per_sample:.2f}",
            f"{metric_stats.avg_span_length:.1f}",
            f"{metric_stats.num_ill_formed_errors}",
            f"{metric_stats.num_ill_formed_extended_span}",
            f"{metric_stats.num_errors_with_ambiguous_match}",
            f"{metric_stats.num_errors_with_ambiguous_match_with_extended_span}",
            f"{metric_stats.num_overlapping_errors}",
            f"{metric_stats.num_severity_filtered_errors}",
            f"{metric_stats.num_category_filtered_errors}",
            f"{metric_stats.num_score_0_errors}",
            metric_stats.severity_breakdown.get("neutral", "0 / 0"),
            metric_stats.severity_breakdown.get("minor", "0 / 0"),
            metric_stats.severity_breakdown.get("major", "0 / 0"),
            metric_stats.severity_breakdown.get("critical", "0 / 0"),
            f"{metric_stats.num_samples_with_no_errors} ({metric_stats.num_samples_with_no_errors/metric_stats.num_samples*100:.1f}%)",
            f"{metric_stats.num_samples}",
        ]
        stats_table_data.append(row)

    print(
        tabulate(
            stats_table_data,
            headers=stats_headers,
            tablefmt="grid",
            stralign="center",
            numalign="center",
        )
    )
    print()


def print_sentinel_counts(
    sentinel_counts: Dict[str, Dict[str, SentinelCounts]],
):
    print(f"\n{'='*100}")
    print("SENTINEL COUNTS")
    print(f"{'='*100}")

    for aggr_type in ["micro", "macro"]:
        print(f"\n{aggr_type.upper()}-AVERAGED:")
        print("-" * 50)

        for match_type in ["matching", "not_matching"]:
            (
                print("GREEDY BIPARTITE MATCHING")
                if match_type == "matching"
                else print("NO MATCHING")
            )
            print("-" * 50)

            counts = sentinel_counts[aggr_type][match_type]

            table_data = []
            headers = [
                "Metric\nType",
                "ext_only\n> normal",
                "ext_only\n≤ normal",
                "no_ext\n> normal",
                "no_ext\n≤ normal",
                "remove_all\n> normal",
                "remove_all\n≤ normal",
            ]

            for metric_type in METRIC_TYPES:
                type_counts = counts.get(metric_type)
                row = [
                    metric_type,
                    type_counts.num_ext_only_greater_than_normal,
                    type_counts.num_ext_only_smaller_or_equal_than_normal,
                    type_counts.num_no_ext_greater_than_normal,
                    type_counts.num_no_ext_smaller_or_equal_than_normal,
                    type_counts.num_remove_all_1_greater_than_normal,
                    type_counts.num_remove_all_1_smaller_or_equal_than_normal,
                ]
                table_data.append(row)

            print(
                tabulate(
                    table_data,
                    headers=headers,
                    tablefmt="grid",
                    stralign="center",
                    numalign="center",
                )
            )
            print()


def print_results_and_stats(
    lp: str,
    metrics_results: Dict[str, Dict[str, Dict[str, MetricResults]]],
    metrics_stats: Dict[str, MetricStats],
    sentinel_counts: Dict[str, Dict[str, SentinelCounts]],
):

    metrics_results_matching: Dict[str, MetricResults] = metrics_results["micro"][
        "matching"
    ]
    metrics_results_no_matching: Dict[str, MetricResults] = metrics_results["micro"][
        "not_matching"
    ]
    metrics_results_matching_macro: Dict[str, MetricResults] = metrics_results["macro"][
        "matching"
    ]
    metrics_results_no_matching_macro: Dict[str, MetricResults] = metrics_results[
        "macro"
    ]["not_matching"]

    print(f"\n{'='*100}")
    print(f"LANGUAGE PAIR: {lp.upper()}")
    print(f"{'='*100}")

    metric_names_in_order = print_results(
        metrics_results_matching,
        metrics_results_no_matching,
        metrics_results_matching_macro,
        metrics_results_no_matching_macro,
    )
    print_sentinel_counts(sentinel_counts)
    print_stats(metrics_stats)
