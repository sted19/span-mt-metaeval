# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Tests for compute_tp_fp_fn_with_error_matching() with overlapping error spans.

This test file investigates how the function behaves when:
1. Two automatic error spans overlap
2. Two gold/human error spans overlap
3. Both automatic and gold spans have overlaps
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
from mt_evaluation.meta_evaluation.span_level.matching import (
    compute_overlap_length,
    find_greedy_bipartite_matching,
    MatchInfo,
)


class TestComputeOverlapLength:
    """Test the overlap length computation helper function."""

    def test_exact_match(self):
        """Test when spans are exactly the same."""
        overlap_len = compute_overlap_length((0, 5), (0, 5))
        assert overlap_len == 5

    def test_span1_contains_span2(self):
        """Test when span1 fully contains span2."""
        overlap_len = compute_overlap_length((0, 10), (2, 7))
        assert overlap_len == 5  # end2 - start2 = 7 - 2

    def test_span2_contains_span1(self):
        """Test when span2 fully contains span1."""
        overlap_len = compute_overlap_length((2, 7), (0, 10))
        assert overlap_len == 5  # end1 - start1 = 7 - 2

    def test_partial_overlap_span1_first(self):
        """Test partial overlap where span1 starts before span2."""
        overlap_len = compute_overlap_length((0, 5), (3, 8))
        assert overlap_len == 2  # end1 - start2 = 5 - 3

    def test_partial_overlap_span2_first(self):
        """Test partial overlap where span2 starts before span1."""
        overlap_len = compute_overlap_length((3, 8), (0, 5))
        assert overlap_len == 2  # end2 - start1 = 5 - 3

    def test_no_overlap(self):
        """Test when spans don't overlap."""
        overlap_len = compute_overlap_length((0, 5), (6, 10))
        assert overlap_len == 0

    def test_adjacent_spans(self):
        """Test when spans are adjacent (end of one equals start of other)."""
        overlap_len = compute_overlap_length((0, 5), (5, 10))
        assert overlap_len == 0


class TestFindGreedyBipartiteMatching:
    """Test the bipartite matching function."""

    def test_single_auto_single_human_exact_match(self):
        """Test with one auto error and one human error at same position."""
        src = "source text"
        tgt = "this is wrong text"

        auto_errors = [
            Error(
                span="wrong",
                category="accuracy",
                severity="major",
                start=8,
                end=13,
                is_source_error=False,
            )
        ]
        human_errors = [
            Error(
                span="wrong",
                category="accuracy",
                severity="major",
                start=8,
                end=13,
                is_source_error=False,
            )
        ]

        auto_matches, human_matches = find_greedy_bipartite_matching(
            auto_errors, human_errors, src, tgt
        )

        assert 0 in auto_matches
        assert 0 in human_matches
        match_info = auto_matches[0]
        assert match_info.avg_overlap == 1.0  # Full overlap (same span)
        assert match_info.overlap_length == 5
        assert match_info.matched_index == 0  # Should match human error at index 0

    def test_two_overlapping_auto_errors_one_human(self):
        """
        Test when two auto errors overlap and both could match one human error.

        Scenario:
        - Human error: positions 5-15
        - Auto error 1: positions 5-10 (overlaps with human)
        - Auto error 2: positions 8-15 (overlaps with human AND auto error 1)

        The function should match greedily, preferring containment then overlap.
        """
        src = "source text"
        tgt = "this is wrong error text"

        # Auto error 1: "s wr" (positions 6-10 in "this is wrong error text")
        # Auto error 2: "wrong err" (positions 8-17)
        # Human error: "is wrong error" (positions 5-19)

        auto_errors = [
            Error(
                span="s wr",
                category="accuracy",
                severity="major",
                start=6,
                end=10,
                is_source_error=False,
            ),
            Error(
                span="wrong err",
                category="accuracy",
                severity="major",
                start=8,
                end=17,
                is_source_error=False,
            ),
        ]
        human_errors = [
            Error(
                span="is wrong error",
                category="accuracy",
                severity="major",
                start=5,
                end=19,
                is_source_error=False,
            )
        ]

        auto_matches, human_matches = find_greedy_bipartite_matching(
            auto_errors, human_errors, src, tgt
        )

        # One human error, so at most one auto error should be matched
        assert len(human_matches) == 1
        # The greedy algorithm should pick the auto error with better overlap
        # Let's see which one gets matched
        print(f"auto_matches: {auto_matches}")
        print(f"human_matches: {human_matches}")

        # Both auto errors should have some match info computed
        # but only one should be in the final matching
        matched_auto_indices = [i for i in auto_matches.keys()]
        assert len(matched_auto_indices) == 1  # Only one auto error can match

    def test_two_overlapping_human_errors_one_auto(self):
        """
        Test when two human errors overlap and both could be matched by one auto error.

        Scenario:
        - Auto error: positions 5-15
        - Human error 1: positions 5-10
        - Human error 2: positions 8-15

        The function should match only one human error to the auto error.
        """
        src = "source text"
        tgt = "this is wrong error text"

        auto_errors = [
            Error(
                span="is wrong error",
                category="accuracy",
                severity="major",
                start=5,
                end=19,
                is_source_error=False,
            )
        ]
        human_errors = [
            Error(
                span="s wr",
                category="accuracy",
                severity="major",
                start=5,
                end=9,
                is_source_error=False,
            ),
            Error(
                span="wrong err",
                category="accuracy",
                severity="major",
                start=8,
                end=17,
                is_source_error=False,
            ),
        ]

        auto_matches, human_matches = find_greedy_bipartite_matching(
            auto_errors, human_errors, src, tgt
        )

        print(f"auto_matches: {auto_matches}")
        print(f"human_matches: {human_matches}")

        # One auto error, so at most one human error should be matched
        assert len(auto_matches) == 1
        matched_human_indices = [i for i in human_matches.keys()]
        assert len(matched_human_indices) == 1  # Only one human error can match


class TestComputeTpFpFnWithErrorMatching:
    """Test the main function with overlapping spans."""

    def test_no_errors(self):
        """Test when there are no errors on either side."""
        src = "source text"
        tgt = "target text"

        auto_eval = AutomaticEvaluation(
            score=0.0,
            errors=[],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=0.0,
            errors=[],
        )

        result = compute_tp_fp_fn_with_error_matching(auto_eval, human_eval, src, tgt)

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
        ) = result

        assert tp_for_precision == 0
        assert tp_for_recall == 0
        assert fp == 0
        assert fn == 0

    def test_perfect_single_match(self):
        """Test with one auto error perfectly matching one human error."""
        src = "source text"
        tgt = "this is wrong text"

        auto_eval = AutomaticEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(auto_eval, human_eval, src, tgt)

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
        ) = result

        assert tp_for_precision == 1
        assert tp_for_recall == 1
        assert fp == 0
        assert fn == 0
        assert tpc == 5  # full overlap
        assert fpc == 0
        assert fnc == 0

    def test_two_overlapping_auto_errors_one_human_error(self):
        """
        Test scenario: Two overlapping auto errors, one human error.

        The greedy bipartite matching should match only one auto error
        to the human error (1-to-1 matching), leaving the other as FP.

        This tests whether the function handles overlapping auto spans correctly.
        """
        src = "source text"
        tgt = "this is a wrong error in the text"

        # Create two overlapping auto errors
        # Auto error 1: "wrong" at positions 10-15
        # Auto error 2: "wrong error" at positions 10-21 (overlaps with auto error 1)
        # Human error: "wrong error" at positions 10-21

        auto_eval = AutomaticEvaluation(
            score=-10.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",
                    start=10,
                    end=15,
                    is_source_error=False,
                ),
                Error(
                    span="wrong error",
                    category="accuracy",
                    severity="major",
                    start=10,
                    end=21,
                    is_source_error=False,
                ),
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong error",
                    category="accuracy",
                    severity="major",
                    start=10,
                    end=21,
                    is_source_error=False,
                )
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(auto_eval, human_eval, src, tgt)

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
        ) = result

        print(f"tp_for_precision={tp_for_precision}, tp_for_recall={tp_for_recall}")
        print(f"fp={fp}, fn={fn}")
        print(f"tpc={tpc}, fpc={fpc}, fnc={fnc}")

        # With 1-to-1 matching:
        # - One auto error matches the human error (TP=1)
        # - One auto error is unmatched (FP=1)
        # - Human error is matched (FN=0)
        assert tp_for_precision == 1, "One auto error should match"
        assert tp_for_recall == 1, "Human error should be matched"
        assert fp == 1, "One auto error should be unmatched (FP)"
        assert fn == 0, "Human error should not be unmatched"

    def test_two_overlapping_human_errors_one_auto_error(self):
        """
        Test scenario: One auto error, two overlapping human errors.

        The greedy bipartite matching should match only one human error
        to the auto error (1-to-1 matching), leaving the other as FN.

        This tests whether the function handles overlapping human spans correctly.
        """
        src = "source text"
        tgt = "this is a wrong error in the text"

        # Auto error: "wrong error" at positions 10-21
        # Human error 1: "wrong" at positions 10-15 (overlaps with human error 2)
        # Human error 2: "wrong error" at positions 10-21

        auto_eval = AutomaticEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong error",
                    category="accuracy",
                    severity="major",
                    start=10,
                    end=21,
                    is_source_error=False,
                )
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-10.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",
                    start=10,
                    end=15,
                    is_source_error=False,
                ),
                Error(
                    span="wrong error",
                    category="accuracy",
                    severity="major",
                    start=10,
                    end=21,
                    is_source_error=False,
                ),
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(auto_eval, human_eval, src, tgt)

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
        ) = result

        print(f"tp_for_precision={tp_for_precision}, tp_for_recall={tp_for_recall}")
        print(f"fp={fp}, fn={fn}")
        print(f"tpc={tpc}, fpc={fpc}, fnc={fnc}")

        # With 1-to-1 matching:
        # - Auto error matches one human error (TP=1)
        # - One human error is unmatched (FN=1)
        # - Auto error is matched (FP=0)
        assert tp_for_precision == 1, "Auto error should match"
        assert tp_for_recall == 1, "One human error should be matched"
        assert fp == 0, "Auto error should not be unmatched"
        assert fn == 1, "One human error should be unmatched (FN)"

    def test_source_and_target_errors_separate(self):
        """
        Test that source errors and target errors are matched separately.

        An auto error in source should NOT match a human error in target.
        """
        src = "source with error text"
        tgt = "target with error text"

        # Auto error is in TARGET
        # Human error is in SOURCE
        auto_eval = AutomaticEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="error",
                    category="accuracy",
                    severity="major",
                    start=12,
                    end=17,
                    is_source_error=False,  # TARGET error
                )
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="error",
                    category="omission",
                    severity="major",
                    start=12,
                    end=17,
                    is_source_error=True,  # SOURCE error
                )
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(auto_eval, human_eval, src, tgt)

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
        ) = result

        # Source and target errors should NOT match
        assert tp_for_precision == 0, "Source/target errors should not match"
        assert tp_for_recall == 0
        assert fp == 1, "Auto target error is unmatched"
        assert fn == 1, "Human source error is unmatched"


class TestOverlappingSpansConsistency:
    """
    Test consistency and correctness when spans overlap.

    Key invariants that should hold:
    1. tp_for_precision + fp == len(auto_errors)
    2. tp_for_recall + fn == len(human_errors)
    3. Each auto error contributes either to TP or FP
    4. Each human error contributes either to TP or FN
    """

    def test_invariants_with_multiple_overlapping_auto_errors(self):
        """
        Test invariants hold when multiple auto errors overlap.
        """
        src = "source text here"
        tgt = "this has multiple overlapping errors here"

        # Create 3 overlapping auto errors
        auto_eval = AutomaticEvaluation(
            score=-15.0,
            errors=[
                Error(
                    span="multiple",
                    category="accuracy",
                    severity="major",
                    start=9,
                    end=17,
                    is_source_error=False,
                ),
                Error(
                    span="multiple overlapping",
                    category="accuracy",
                    severity="major",
                    start=9,
                    end=29,
                    is_source_error=False,
                ),
                Error(
                    span="overlapping errors",
                    category="accuracy",
                    severity="major",
                    start=18,
                    end=36,
                    is_source_error=False,
                ),
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-10.0,
            errors=[
                Error(
                    span="multiple overlapping errors",
                    category="accuracy",
                    severity="major",
                    start=9,
                    end=36,
                    is_source_error=False,
                ),
                Error(
                    span="here",
                    category="accuracy",
                    severity="minor",
                    start=37,
                    end=41,
                    is_source_error=False,
                ),
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(auto_eval, human_eval, src, tgt)

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
        ) = result

        print(
            f"Result: tp_p={tp_for_precision}, tp_r={tp_for_recall}, fp={fp}, fn={fn}"
        )

        # Invariant 1: tp_for_precision + fp == len(auto_errors)
        assert tp_for_precision + fp == len(
            auto_eval.errors
        ), f"Invariant violated: {tp_for_precision} + {fp} != {len(auto_eval.errors)}"

        # Invariant 2: tp_for_recall + fn == len(human_errors)
        assert tp_for_recall + fn == len(
            human_eval.errors
        ), f"Invariant violated: {tp_for_recall} + {fn} != {len(human_eval.errors)}"

    def test_invariants_with_multiple_overlapping_human_errors(self):
        """
        Test invariants hold when multiple human errors overlap.
        """
        src = "source text here"
        tgt = "this has multiple overlapping errors here"

        auto_eval = AutomaticEvaluation(
            score=-10.0,
            errors=[
                Error(
                    span="multiple overlapping errors",
                    category="accuracy",
                    severity="major",
                    start=9,
                    end=36,
                    is_source_error=False,
                ),
                Error(
                    span="here",
                    category="accuracy",
                    severity="minor",
                    start=37,
                    end=41,
                    is_source_error=False,
                ),
            ],
            annotation="",
            parsing_error=False,
        )
        # Create 3 overlapping human errors
        human_eval = HumanEvaluation(
            score=-15.0,
            errors=[
                Error(
                    span="multiple",
                    category="accuracy",
                    severity="major",
                    start=9,
                    end=17,
                    is_source_error=False,
                ),
                Error(
                    span="multiple overlapping",
                    category="accuracy",
                    severity="major",
                    start=9,
                    end=29,
                    is_source_error=False,
                ),
                Error(
                    span="overlapping errors",
                    category="accuracy",
                    severity="major",
                    start=18,
                    end=36,
                    is_source_error=False,
                ),
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(auto_eval, human_eval, src, tgt)

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
        ) = result

        print(
            f"Result: tp_p={tp_for_precision}, tp_r={tp_for_recall}, fp={fp}, fn={fn}"
        )

        # Invariant 1: tp_for_precision + fp == len(auto_errors)
        assert tp_for_precision + fp == len(
            auto_eval.errors
        ), f"Invariant violated: {tp_for_precision} + {fp} != {len(auto_eval.errors)}"

        # Invariant 2: tp_for_recall + fn == len(human_errors)
        assert tp_for_recall + fn == len(
            human_eval.errors
        ), f"Invariant violated: {tp_for_recall} + {fn} != {len(human_eval.errors)}"


class TestSeverityPenalty:
    """Test the severity_penalty parameter in compute_tp_fp_fn_with_error_matching()."""

    def test_severity_penalty_zero_no_effect(self):
        """Test that severity_penalty=0.0 has no effect (backward compatible)."""
        src = "source text"
        tgt = "this is wrong text"

        # Auto: minor severity, Human: major severity (mismatch)
        auto_eval = AutomaticEvaluation(
            score=-1.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="minor",  # Different from human
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",  # Different from auto
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(
            auto_eval, human_eval, src, tgt, severity_penalty=0.0
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
        ) = result

        # With penalty=0.0, severity mismatch doesn't matter - full TP
        assert tp_for_precision == 1.0
        assert tp_for_recall == 1.0
        assert fp == 0.0
        assert fn == 0.0

    def test_severity_penalty_half_partial_match(self):
        """Test that severity_penalty=0.5 gives half credit for severity mismatch."""
        src = "source text"
        tgt = "this is wrong text"

        # Auto: minor severity, Human: major severity (mismatch)
        auto_eval = AutomaticEvaluation(
            score=-1.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="minor",
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(
            auto_eval, human_eval, src, tgt, severity_penalty=0.5
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
        ) = result

        # With penalty=0.5, severity mismatch gives 50% TP, 50% FP/FN
        assert tp_for_precision == 0.5
        assert tp_for_recall == 0.5
        assert fp == 0.5
        assert fn == 0.5

        # Character-level: 5 chars, 50% penalty
        assert tpc == 2.5  # 5 * 0.5
        assert fpc == 2.5  # 5 - 2.5
        assert fnc == 2.5  # 5 - 2.5

    def test_severity_penalty_full_treated_as_no_match(self):
        """Test that severity_penalty=1.0 treats severity mismatch as no match."""
        src = "source text"
        tgt = "this is wrong text"

        auto_eval = AutomaticEvaluation(
            score=-1.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="minor",
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(
            auto_eval, human_eval, src, tgt, severity_penalty=1.0
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
        ) = result

        # With penalty=1.0, severity mismatch is treated as no match
        assert tp_for_precision == 0.0
        assert tp_for_recall == 0.0
        assert fp == 1.0
        assert fn == 1.0

    def test_severity_match_no_penalty(self):
        """Test that matching severities have no penalty regardless of penalty value."""
        src = "source text"
        tgt = "this is wrong text"

        # Both auto and human have same severity
        auto_eval = AutomaticEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",  # Same as auto
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
        )

        # Test with various penalty values
        for penalty in [0.0, 0.5, 1.0]:
            result = compute_tp_fp_fn_with_error_matching(
                auto_eval, human_eval, src, tgt, severity_penalty=penalty
            )

            # Result now has 14 values: tp_em, fp_em, fn_em, then the rest
            tp_em, fp_em, fn_em, tp_for_precision, tp_for_recall, fp, fn = result[:7]

            # With matching severity, full TP regardless of penalty
            assert tp_for_precision == 1.0, f"Failed with penalty={penalty}"
            assert tp_for_recall == 1.0, f"Failed with penalty={penalty}"
            assert fp == 0.0, f"Failed with penalty={penalty}"
            assert fn == 0.0, f"Failed with penalty={penalty}"

    def test_severity_penalty_with_partial_overlap(self):
        """Test severity penalty with partial span overlap."""
        src = "source text"
        tgt = "this is a wrong error here"

        # Auto error: "wrong" (positions 10-15)
        # Human error: "wrong error" (positions 10-21)
        # Different severities
        auto_eval = AutomaticEvaluation(
            score=-1.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="minor",
                    start=10,
                    end=15,
                    is_source_error=False,
                )
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong error",
                    category="accuracy",
                    severity="major",
                    start=10,
                    end=21,
                    is_source_error=False,
                )
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(
            auto_eval, human_eval, src, tgt, severity_penalty=0.5
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
        ) = result

        # Partial overlap + severity penalty
        # - They overlap (auto contained in human)
        # - Severity mismatch with 0.5 penalty
        assert tp_for_precision == 0.5
        assert tp_for_recall == 0.5
        assert fp == 0.5
        assert fn == 0.5

        # Character-level:
        # - Overlap is 5 chars (the "wrong" part)
        # - With 0.5 penalty: tpc = 5 * 0.5 = 2.5
        # - Auto span is 5 chars: fpc = 5 - 2.5 = 2.5
        # - Human span is 11 chars: fnc = 11 - 2.5 = 8.5
        assert tpc == 2.5
        assert fpc == 2.5
        assert fnc == 8.5

    def test_severity_penalty_invariants(self):
        """Test that invariants hold with severity penalty."""
        src = "source text"
        tgt = "this is wrong text"

        auto_eval = AutomaticEvaluation(
            score=-1.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="minor",
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",
                    start=8,
                    end=13,
                    is_source_error=False,
                )
            ],
        )

        for penalty in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = compute_tp_fp_fn_with_error_matching(
                auto_eval, human_eval, src, tgt, severity_penalty=penalty
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
            ) = result

            # Invariant 1: tp_for_precision + fp == len(auto_errors)
            assert abs(tp_for_precision + fp - 1) < 1e-5, \
                f"Invariant 1 violated with penalty={penalty}: {tp_for_precision} + {fp} != 1"

            # Invariant 2: tp_for_recall + fn == len(human_errors)
            assert abs(tp_for_recall + fn - 1) < 1e-5, \
                f"Invariant 2 violated with penalty={penalty}: {tp_for_recall} + {fn} != 1"

            # Invariant 3: tppc_for_precision + fppc == len(auto_errors)
            assert abs(tppc_for_precision + fppc - 1) < 1e-5, \
                f"Invariant 3 violated with penalty={penalty}"

            # Invariant 4: tppc_for_recall + fnpc == len(human_errors)
            assert abs(tppc_for_recall + fnpc - 1) < 1e-5, \
                f"Invariant 4 violated with penalty={penalty}"

    def test_severity_penalty_multiple_errors_mixed_severities(self):
        """Test severity penalty with multiple errors having different severity matches."""
        src = "source text"
        tgt = "this is wrong text with error"

        # Two auto errors: one matches severity, one doesn't
        auto_eval = AutomaticEvaluation(
            score=-6.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",  # Matches human
                    start=8,
                    end=13,
                    is_source_error=False,
                ),
                Error(
                    span="error",
                    category="accuracy",
                    severity="minor",  # Doesn't match human
                    start=24,
                    end=29,
                    is_source_error=False,
                ),
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-10.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="major",  # Matches auto
                    start=8,
                    end=13,
                    is_source_error=False,
                ),
                Error(
                    span="error",
                    category="accuracy",
                    severity="major",  # Doesn't match auto
                    start=24,
                    end=29,
                    is_source_error=False,
                ),
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(
            auto_eval, human_eval, src, tgt, severity_penalty=0.5
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
        ) = result

        # First error: exact match (severity matches) -> TP=1
        # Second error: severity mismatch with 0.5 penalty -> TP=0.5, FP/FN=0.5
        assert tp_for_precision == 1.5  # 1.0 + 0.5
        assert tp_for_recall == 1.5
        assert fp == 0.5  # Only from second error
        assert fn == 0.5  # Only from second error


class TestSeverityPenaltyWithOverlaps:
    """Test severity penalty in combination with overlapping spans."""

    def test_overlapping_auto_errors_severity_mismatch(self):
        """
        Test: Two overlapping auto errors compete for one human error,
        and the matched one has wrong severity.
        """
        src = "source text"
        tgt = "this is a wrong error in the text"

        # Two overlapping auto errors, both minor severity
        # Human error is major severity
        auto_eval = AutomaticEvaluation(
            score=-2.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="minor",
                    start=10,
                    end=15,
                    is_source_error=False,
                ),
                Error(
                    span="wrong error",
                    category="accuracy",
                    severity="minor",
                    start=10,
                    end=21,
                    is_source_error=False,
                ),
            ],
            annotation="",
            parsing_error=False,
        )
        human_eval = HumanEvaluation(
            score=-5.0,
            errors=[
                Error(
                    span="wrong error",
                    category="accuracy",
                    severity="major",  # Different from both auto errors
                    start=10,
                    end=21,
                    is_source_error=False,
                )
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(
            auto_eval, human_eval, src, tgt, severity_penalty=0.5
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
        ) = result

        print(f"Result: tp_p={tp_for_precision}, tp_r={tp_for_recall}, fp={fp}, fn={fn}")

        # Invariants should still hold
        assert abs(tp_for_precision + fp - 2) < 1e-5, \
            f"Invariant violated: {tp_for_precision} + {fp} != 2"
        assert abs(tp_for_recall + fn - 1) < 1e-5, \
            f"Invariant violated: {tp_for_recall} + {fn} != 1"

        # One auto error is matched (but with severity penalty), one is FP
        # Best match: "wrong error" auto (exact position) -> but severity mismatch
        # So matched auto gets 0.5 TP, unmatched gets full FP
        assert tp_for_precision == 0.5, "Matched auto error should get partial credit"
        assert fp == 1.5, "One full FP (unmatched) + 0.5 from severity penalty"

    def test_overlapping_human_errors_one_has_matching_severity(self):
        """
        Test: Two overlapping human errors compete for one auto error.
        One human error has matching severity, one doesn't.
        Greedy matching might not pick the one with matching severity.
        """
        src = "source text"
        tgt = "this is a wrong error in the text"

        # Auto error: minor severity
        auto_eval = AutomaticEvaluation(
            score=-1.0,
            errors=[
                Error(
                    span="wrong error",
                    category="accuracy",
                    severity="minor",
                    start=10,
                    end=21,
                    is_source_error=False,
                )
            ],
            annotation="",
            parsing_error=False,
        )
        # Two overlapping human errors with different severities
        human_eval = HumanEvaluation(
            score=-6.0,
            errors=[
                Error(
                    span="wrong",
                    category="accuracy",
                    severity="minor",  # Matches auto!
                    start=10,
                    end=15,
                    is_source_error=False,
                ),
                Error(
                    span="wrong error",
                    category="accuracy",
                    severity="major",  # Doesn't match auto
                    start=10,
                    end=21,
                    is_source_error=False,
                ),
            ],
        )

        result = compute_tp_fp_fn_with_error_matching(
            auto_eval, human_eval, src, tgt, severity_penalty=0.5
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
        ) = result

        print(f"Result: tp_p={tp_for_precision}, tp_r={tp_for_recall}, fp={fp}, fn={fn}")

        # Invariants should still hold
        assert abs(tp_for_precision + fp - 1) < 1e-5
        assert abs(tp_for_recall + fn - 2) < 1e-5

        # Greedy matching prefers containment, so "wrong error" human error
        # (which exactly matches auto position) should be picked
        # But that one has major severity (mismatch) -> partial credit
        # The "wrong" human error is unmatched -> FN


if __name__ == "__main__":
    # Run specific tests with verbose output
    pytest.main([__file__, "-v", "-s"])
