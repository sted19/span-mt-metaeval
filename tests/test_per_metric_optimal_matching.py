# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Tests for per-metric-type optimal matching in compute_tp_fp_fn_with_error_matching.

Verifies that when use_greedy_matching=False, each metric type (Exact Match,
Partial Overlap, Character Counts, Character Proportion) uses its own optimal
matching, potentially producing different TP/FP/FN values than a single shared
matching would.
"""

import pytest
from mt_evaluation.core import (
    AutomaticEvaluation,
    HumanEvaluation,
    Error,
)
from mt_evaluation.meta_evaluation.span_level.utils import (
    compute_tp_fp_fn_with_error_matching,
)


def make_auto_eval(errors):
    """Helper to create an AutomaticEvaluation with given errors."""
    return AutomaticEvaluation(
        score=0.0,
        errors=errors,
        annotation="",
        parsing_error=False,
        user_prompt="",
        system_prompt="",
        cost=0.0,
    )


def make_human_eval(errors):
    """Helper to create a HumanEvaluation with given errors."""
    return HumanEvaluation(
        score=0.0,
        errors=errors,
    )


def unpack_result(res):
    """Unpack the 14-tuple into a readable dict."""
    (
        tp_em, fp_em, fn_em,
        tp_for_precision, tp_for_recall, fp, fn,
        tpc, fpc, fnc,
        tppc_for_precision, tppc_for_recall, fppc, fnpc,
    ) = res
    return {
        "tp_em": tp_em, "fp_em": fp_em, "fn_em": fn_em,
        "tp_for_precision": tp_for_precision, "tp_for_recall": tp_for_recall,
        "fp": fp, "fn": fn,
        "tpc": tpc, "fpc": fpc, "fnc": fnc,
        "tppc_for_precision": tppc_for_precision, "tppc_for_recall": tppc_for_recall,
        "fppc": fppc, "fnpc": fnpc,
    }


class TestOptimalMatchingNoErrors:
    """Edge case: no errors on either side."""

    def test_no_errors_optimal(self):
        auto_eval = make_auto_eval([])
        human_eval = make_human_eval([])
        res = compute_tp_fp_fn_with_error_matching(
            auto_eval, human_eval, "src", "tgt", use_greedy_matching=False
        )
        d = unpack_result(res)
        for key, val in d.items():
            assert val == 0.0, f"{key} should be 0.0, got {val}"


class TestOptimalMatchingPerfectMatch:
    """Single auto error exactly matching single human error."""

    def test_perfect_single_match_optimal(self):
        src = "source text"
        tgt = "this is a wrong translation"

        auto_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=10, end=15, is_source_error=False)
        ]
        human_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=10, end=15, is_source_error=False)
        ]

        res = compute_tp_fp_fn_with_error_matching(
            make_auto_eval(auto_errors),
            make_human_eval(human_errors),
            src, tgt,
            use_greedy_matching=False,
        )
        d = unpack_result(res)

        # All metric types should show a perfect match
        assert d["tp_em"] == 1.0
        assert d["fp_em"] == 0.0
        assert d["fn_em"] == 0.0
        assert d["tp_for_precision"] == 1.0
        assert d["fp"] == 0.0
        assert d["fn"] == 0.0
        assert d["tpc"] == 5.0  # 5 chars overlap
        assert d["fpc"] == 0.0
        assert d["fnc"] == 0.0
        assert d["tppc_for_precision"] == 1.0
        assert d["fppc"] == 0.0
        assert d["fnpc"] == 0.0


class TestOptimalMatchingGreedyEquivalence:
    """When there's no conflict, greedy and optimal should agree."""

    def test_simple_case_greedy_equals_optimal(self):
        """Two non-overlapping error pairs — same result either way."""
        src = "source"
        tgt = "aaa bbb ccc ddd eee"

        auto_errors = [
            Error(span="aaa", category="accuracy", severity="major",
                  start=0, end=3, is_source_error=False),
            Error(span="ddd", category="accuracy", severity="major",
                  start=12, end=15, is_source_error=False),
        ]
        human_errors = [
            Error(span="aaa", category="accuracy", severity="major",
                  start=0, end=3, is_source_error=False),
            Error(span="ddd", category="accuracy", severity="major",
                  start=12, end=15, is_source_error=False),
        ]

        res_greedy = compute_tp_fp_fn_with_error_matching(
            make_auto_eval(auto_errors),
            make_human_eval(human_errors),
            src, tgt,
            use_greedy_matching=True,
        )
        res_optimal = compute_tp_fp_fn_with_error_matching(
            make_auto_eval(auto_errors),
            make_human_eval(human_errors),
            src, tgt,
            use_greedy_matching=False,
        )

        for g, o in zip(res_greedy, res_optimal):
            assert g == pytest.approx(o), (
                f"Greedy and optimal should match: greedy={res_greedy}, optimal={res_optimal}"
            )


class TestOptimalMatchingDifferentPerMetric:
    """
    Scenario where different objectives produce different optimal matchings.

    Setup:
        auto_0: [0, 10)   "aaaaaaaaaa"    len=10
        auto_1: [10, 20)  "bbbbbbbbbb"    len=10

        human_0: [0, 10)   "aaaaaaaaaa"    len=10  (exact match with auto_0)
        human_1: [5, 20)   "aaaaabbbbb..."  len=15  (overlaps auto_0 by 5, auto_1 by 10)

    Exact Match objective:
        auto_0 <-> human_0: exact match (score 1.0)
        auto_0 <-> human_1: no exact match (score 0.0)
        auto_1 <-> human_0: no exact match (score 0.0)
        auto_1 <-> human_1: no exact match (score 0.0)
        → Optimal: auto_0↔human_0 (only match). auto_1 unmatched.
          tp_em=1.0, fp_em=1.0, fn_em=1.0

    Partial Overlap objective:
        auto_0 <-> human_0: overlap (score 1.0)
        auto_0 <-> human_1: overlap (score 1.0)
        auto_1 <-> human_0: no overlap (score 0.0)
        auto_1 <-> human_1: overlap (score 1.0)
        → Optimal: auto_0↔human_0, auto_1↔human_1 (both matched).
          tp_for_precision=2.0, fp=0.0

    Character Counts objective:
        auto_0 <-> human_0: overlap=10
        auto_0 <-> human_1: overlap=5
        auto_1 <-> human_0: overlap=0
        auto_1 <-> human_1: overlap=10
        → Optimal: auto_0↔human_0 (10), auto_1↔human_1 (10). Total=20.
          tpc=20.0, fpc=0.0

    Character Proportion objective (harmonic mean of overlap/len):
        auto_0 <-> human_0: 2*10/(10+10) = 1.0  (F1=1.0: perfect)
        auto_0 <-> human_1: 2*5/(10+15) = 0.4
        auto_1 <-> human_0: 0
        auto_1 <-> human_1: 2*10/(10+15) = 0.8
        → Optimal: auto_0↔human_0 (1.0), auto_1↔human_1 (0.8). Total=1.8.
          Same matching as char counts here.

    Key difference: Exact Match matching only matches 1 pair, while the other
    three objectives match both pairs.
    """

    def test_exact_match_differs_from_partial_overlap(self):
        src = "source text"
        tgt = "a" * 20  # 20 chars

        auto_errors = [
            Error(span="a" * 10, category="accuracy", severity="major",
                  start=0, end=10, is_source_error=False),
            Error(span="a" * 10, category="accuracy", severity="major",
                  start=10, end=20, is_source_error=False),
        ]
        human_errors = [
            Error(span="a" * 10, category="accuracy", severity="major",
                  start=0, end=10, is_source_error=False),
            Error(span="a" * 15, category="accuracy", severity="major",
                  start=5, end=20, is_source_error=False),
        ]

        res = compute_tp_fp_fn_with_error_matching(
            make_auto_eval(auto_errors),
            make_human_eval(human_errors),
            src, tgt,
            use_greedy_matching=False,
        )
        d = unpack_result(res)

        # Exact Match: only auto_0↔human_0 matches (the only exact pair)
        # auto_1 is unmatched → fp_em=1
        # human_1 is unmatched → fn_em=1
        assert d["tp_em"] == 1.0
        assert d["fp_em"] == 1.0
        assert d["fn_em"] == 1.0

        # Partial Overlap: auto_0↔human_0, auto_1↔human_1 both matched
        assert d["tp_for_precision"] == 2.0
        assert d["fp"] == 0.0
        assert d["tp_for_recall"] == 2.0
        assert d["fn"] == 0.0

        # Character Counts: auto_0↔human_0 (10 chars), auto_1↔human_1 (10 chars)
        assert d["tpc"] == 20.0
        assert d["fpc"] == 0.0
        assert d["fnc"] == 5.0  # human_1 has 15 chars, 10 matched → 5 unmatched

        # Character Proportion: auto_0↔human_0, auto_1↔human_1
        # auto_0: overlap=10, span=10 → portion=10/10=1.0
        assert d["tppc_for_precision"] == pytest.approx(1.0 + 10 / 10)  # 1.0 + 1.0 = 2.0
        # human_1: overlap=10, span=15 → portion=10/15
        assert d["tppc_for_recall"] == pytest.approx(1.0 + 10 / 15)

    def test_invariants_hold_per_metric_with_optimal(self):
        """Verify structural invariants hold for each metric type."""
        src = "source text"
        tgt = "a" * 20

        auto_errors = [
            Error(span="a" * 10, category="accuracy", severity="major",
                  start=0, end=10, is_source_error=False),
            Error(span="a" * 10, category="accuracy", severity="major",
                  start=10, end=20, is_source_error=False),
        ]
        human_errors = [
            Error(span="a" * 10, category="accuracy", severity="major",
                  start=0, end=10, is_source_error=False),
            Error(span="a" * 15, category="accuracy", severity="major",
                  start=5, end=20, is_source_error=False),
        ]

        res = compute_tp_fp_fn_with_error_matching(
            make_auto_eval(auto_errors),
            make_human_eval(human_errors),
            src, tgt,
            use_greedy_matching=False,
        )
        d = unpack_result(res)

        # Exact Match: tp_em + fp_em == num_auto (precision side)
        assert d["tp_em"] + d["fp_em"] == pytest.approx(len(auto_errors))
        # fn_em: each human error contributes 1-exact_match_reward or 1
        # so fn_em <= num_human
        assert d["fn_em"] <= len(human_errors) + 1e-9

        # Partial Overlap: tp + fp == num_auto, tp_r + fn == num_human
        assert d["tp_for_precision"] + d["fp"] == pytest.approx(len(auto_errors))
        assert d["tp_for_recall"] + d["fn"] == pytest.approx(len(human_errors))

        # Character Proportion: tppc + fppc == num_auto, tppc_r + fnpc == num_human
        assert d["tppc_for_precision"] + d["fppc"] == pytest.approx(len(auto_errors))
        assert d["tppc_for_recall"] + d["fnpc"] == pytest.approx(len(human_errors))


class TestOptimalMatchingSeverityPenalty:
    """Per-metric optimal matching with severity penalties."""

    def test_severity_mismatch_with_optimal(self):
        """Different severities with penalty=0.5 using optimal matching."""
        src = "source"
        tgt = "this is wrong text here"

        auto_errors = [
            Error(span="wrong", category="accuracy", severity="minor",
                  start=8, end=13, is_source_error=False),
        ]
        human_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=8, end=13, is_source_error=False),
        ]

        res = compute_tp_fp_fn_with_error_matching(
            make_auto_eval(auto_errors),
            make_human_eval(human_errors),
            src, tgt,
            severity_penalty=0.5,
            use_greedy_matching=False,
        )
        d = unpack_result(res)

        # Exact match: spans are identical but severity differs
        # binary_reward = 1 - 0.5 = 0.5
        assert d["tp_em"] == pytest.approx(0.5)
        assert d["fp_em"] == pytest.approx(0.5)

        # Partial overlap: same match, binary_reward=0.5
        assert d["tp_for_precision"] == pytest.approx(0.5)
        assert d["fp"] == pytest.approx(0.5)

        # Invariants
        assert d["tp_for_precision"] + d["fp"] == pytest.approx(1.0)
        assert d["tp_for_recall"] + d["fn"] == pytest.approx(1.0)

    def test_severity_full_penalty_blocks_match(self):
        """With penalty=1.0, severity mismatch blocks matching entirely."""
        src = "source"
        tgt = "this is wrong text here"

        auto_errors = [
            Error(span="wrong", category="accuracy", severity="minor",
                  start=8, end=13, is_source_error=False),
        ]
        human_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=8, end=13, is_source_error=False),
        ]

        res = compute_tp_fp_fn_with_error_matching(
            make_auto_eval(auto_errors),
            make_human_eval(human_errors),
            src, tgt,
            severity_penalty=1.0,
            use_greedy_matching=False,
        )
        d = unpack_result(res)

        # With full penalty, the matching score is 0 → no match
        assert d["tp_em"] == 0.0
        assert d["fp_em"] == 1.0
        assert d["fn_em"] == 1.0
        assert d["tp_for_precision"] == 0.0
        assert d["fp"] == 1.0
        assert d["fn"] == 1.0


class TestOptimalMatchingSourceVsTarget:
    """Errors on different sides (source vs target) should not match."""

    def test_source_target_separation_optimal(self):
        src = "source wrong"
        tgt = "target wrong"

        auto_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=7, end=12, is_source_error=True),
        ]
        human_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=7, end=12, is_source_error=False),
        ]

        res = compute_tp_fp_fn_with_error_matching(
            make_auto_eval(auto_errors),
            make_human_eval(human_errors),
            src, tgt,
            use_greedy_matching=False,
        )
        d = unpack_result(res)

        # No match because different sides
        assert d["fp_em"] == 1.0
        assert d["fn_em"] == 1.0
        assert d["fp"] == 1.0
        assert d["fn"] == 1.0
        assert d["tpc"] == 0.0
        assert d["fpc"] == 5.0
        assert d["fnc"] == 5.0
