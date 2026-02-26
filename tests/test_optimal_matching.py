# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Tests for find_optimal_bipartite_matching().

Verifies correctness of the Hungarian-algorithm-based optimal matching,
including edge cases and scenarios where optimal differs from greedy.
"""

import pytest
from mt_evaluation.core import Error
from mt_evaluation.meta_evaluation.span_level.matching import (
    find_optimal_bipartite_matching,
    find_greedy_bipartite_matching,
    compute_overlap_length,
    MatchInfo,
)


from typing import Tuple


def avg_overlap_objective(
    span1: Tuple[int, int], span2: Tuple[int, int], src: str, tgt: str
) -> float:
    """Objective that mirrors the avg_overlap used in greedy matching."""
    overlap_len = compute_overlap_length(span1, span2)
    if overlap_len == 0:
        return 0.0
    len1 = span1[1] - span1[0]
    len2 = span2[1] - span2[0]
    total_span_len = len1 + len2
    if total_span_len == 0:
        return 0.0
    return 2 * overlap_len / total_span_len


class TestOptimalMatchingEmptyInputs:
    """Edge cases with empty error lists."""

    def test_both_empty(self):
        auto_matches, human_matches = find_optimal_bipartite_matching(
            [], [], "src", "tgt", avg_overlap_objective
        )
        assert auto_matches == {}
        assert human_matches == {}

    def test_empty_auto(self):
        human_errors = [
            Error(span="word", category="accuracy", severity="major",
                  start=0, end=4, is_source_error=False)
        ]
        auto_matches, human_matches = find_optimal_bipartite_matching(
            [], human_errors, "src", "tgt word", avg_overlap_objective
        )
        assert auto_matches == {}
        assert human_matches == {}

    def test_empty_human(self):
        auto_errors = [
            Error(span="word", category="accuracy", severity="major",
                  start=0, end=4, is_source_error=False)
        ]
        auto_matches, human_matches = find_optimal_bipartite_matching(
            auto_errors, [], "src", "tgt word", avg_overlap_objective
        )
        assert auto_matches == {}
        assert human_matches == {}


class TestOptimalMatchingExactMatch:
    """Single pair, exact match."""

    def test_single_exact_match(self):
        src = "source text"
        tgt = "this is wrong text"

        auto_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=8, end=13, is_source_error=False)
        ]
        human_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=8, end=13, is_source_error=False)
        ]

        auto_matches, human_matches = find_optimal_bipartite_matching(
            auto_errors, human_errors, src, tgt, avg_overlap_objective
        )

        assert 0 in auto_matches
        assert 0 in human_matches
        assert auto_matches[0].avg_overlap == 1.0
        assert auto_matches[0].overlap_length == 5
        assert auto_matches[0].matched_index == 0


class TestOptimalMatchingNoOverlap:
    """Pairs with no overlap should not be matched."""

    def test_no_overlap(self):
        src = "source"
        tgt = "aaa bbb ccc ddd"

        auto_errors = [
            Error(span="aaa", category="accuracy", severity="major",
                  start=0, end=3, is_source_error=False)
        ]
        human_errors = [
            Error(span="ddd", category="accuracy", severity="major",
                  start=12, end=15, is_source_error=False)
        ]

        auto_matches, human_matches = find_optimal_bipartite_matching(
            auto_errors, human_errors, src, tgt, avg_overlap_objective
        )

        assert auto_matches == {}
        assert human_matches == {}


class TestOptimalMatchingSourceTargetMismatch:
    """Errors on different sides (source vs target) should not match."""

    def test_source_vs_target(self):
        src = "wrong source"
        tgt = "wrong target"

        auto_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=0, end=5, is_source_error=True)
        ]
        human_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=0, end=5, is_source_error=False)
        ]

        auto_matches, human_matches = find_optimal_bipartite_matching(
            auto_errors, human_errors, src, tgt, avg_overlap_objective
        )

        assert auto_matches == {}
        assert human_matches == {}


class TestOptimalMatchingSeverityPenalty:
    """Severity penalty reduces score for mismatched severities."""

    def test_severity_penalty_applied(self):
        src = "source"
        tgt = "this is wrong text"

        auto_errors = [
            Error(span="wrong", category="accuracy", severity="minor",
                  start=8, end=13, is_source_error=False)
        ]
        human_errors = [
            Error(span="wrong", category="accuracy", severity="major",
                  start=8, end=13, is_source_error=False)
        ]

        # With no penalty: should match
        auto_matches, _ = find_optimal_bipartite_matching(
            auto_errors, human_errors, src, tgt, avg_overlap_objective,
            severity_penalty=0.0,
        )
        assert 0 in auto_matches

        # With full penalty: score becomes 0 → no match
        auto_matches, _ = find_optimal_bipartite_matching(
            auto_errors, human_errors, src, tgt, avg_overlap_objective,
            severity_penalty=1.0,
        )
        assert auto_matches == {}


class TestOptimalBeatsGreedy:
    """
    The key scenario: optimal matching outperforms greedy.

    Consider three errors where the greedy first pick blocks a globally
    better assignment.

    Setup (all target-side, same severity):
        human_error_0: positions 0-10   (span length 10)
        human_error_1: positions 8-18   (span length 10)
        auto_error_0:  positions 0-12   (span length 12)  — overlaps both humans
        auto_error_1:  positions 9-18   (span length 9)   — overlaps only human_1

    Greedy picks: auto_0 <-> human_0  (highest single-pair overlap),
        then auto_1 <-> human_1.
    But the optimal is: auto_0 <-> human_1, auto_1 ... wait, auto_1 only overlaps
    human_1. Let me engineer this properly.

    Better setup:
        human_0: [0, 10)   "aaaaaaaaaa"    len=10
        human_1: [10, 20)  "bbbbbbbbbb"    len=10

        auto_0: [5, 15)    "aaaaabbbbb"    len=10   — overlaps human_0 by 5, human_1 by 5
        auto_1: [0, 10)    "aaaaaaaaaa"    len=10   — overlaps human_0 by 10

    Pairwise avg_overlap (Dice coeff = 2*overlap / (len_a + len_h)):
        auto_0 <-> human_0: 2*5/(10+10) = 0.5
        auto_0 <-> human_1: 2*5/(10+10) = 0.5
        auto_1 <-> human_0: 2*10/(10+10) = 1.0
        auto_1 <-> human_1: 0  (no overlap)

    Greedy: picks (auto_1, human_0) score=1.0, then (auto_0, human_1) score=0.5  → total = 1.5
    Optimal: same here — total = 1.5. Both agree.

    Let me try a different layout where they disagree:
        human_0: [0, 10)   len=10
        human_1: [8, 18)   len=10

        auto_0: [0, 14)    len=14   — overlaps human_0 by 10, human_1 by 6
        auto_1: [7, 18)    len=11   — overlaps human_0 by 3, human_1 by 10

    Pairwise:
        auto_0 <-> human_0: 2*10/(14+10) = 20/24 ≈ 0.833
        auto_0 <-> human_1: 2*6/(14+10)  = 12/24  = 0.5
        auto_1 <-> human_0: 2*3/(11+10)  = 6/21  ≈ 0.286
        auto_1 <-> human_1: 2*10/(11+10) = 20/21 ≈ 0.952

    Greedy: picks (auto_1, human_1) score≈0.952, then (auto_0, human_0) score≈0.833 → total ≈ 1.786
    Optimal: same — total ≈ 1.786.
    Alternative: (auto_0, human_1)=0.5 + (auto_1, human_0)=0.286 = 0.786 → worse.

    Hmm, it's hard to beat greedy with Dice overlap on real spans. Let me use a
    custom objective that makes it easy to demonstrate.
    """

    def test_optimal_outperforms_greedy_with_custom_objective(self):
        """
        Custom objective where greedy is sub-optimal.

        Three errors each side:
            auto_0, auto_1, auto_2
            human_0, human_1, human_2

        Custom objective matrix (engineered so greedy picks badly):
            obj(auto_0, human_0) = 0.9   ← greedy picks this first
            obj(auto_0, human_1) = 0.8
            obj(auto_0, human_2) = 0.0
            obj(auto_1, human_0) = 0.85
            obj(auto_1, human_1) = 0.0
            obj(auto_1, human_2) = 0.0
            obj(auto_2, human_0) = 0.0
            obj(auto_2, human_1) = 0.7
            obj(auto_2, human_2) = 0.1

        Greedy: picks (auto_0, human_0)=0.9, then (auto_2, human_1)=0.7,
                then (auto_1, -)=unmatched, (-, human_2)=unmatched
                → total = 0.9 + 0.7 + 0.1 = 1.7

        Optimal: (auto_0, human_1)=0.8, (auto_1, human_0)=0.85, (auto_2, human_2)=0.1
                → total = 0.8 + 0.85 + 0.1 = 1.75
                OR: (auto_1, human_0)=0.85, (auto_0, human_1)=0.8, (auto_2, human_2)=0.1
                → total = 1.75

        Greedy total = 1.7 < Optimal total = 1.75
        """
        # All errors at the same position so source/target filtering doesn't interfere.
        # We rely on the custom objective for scoring.

        def make_error(label: str, start: int, end: int) -> Error:
            return Error(
                span=label, category="accuracy", severity="major",
                start=start, end=end, is_source_error=False,
            )

        auto_errors = [
            make_error("a0", 0, 1),
            make_error("a1", 0, 1),
            make_error("a2", 0, 1),
        ]
        human_errors = [
            make_error("h0", 0, 1),
            make_error("h1", 0, 1),
            make_error("h2", 0, 1),
        ]

        # Encode the score matrix using (auto_index, human_index) pairs.
        # All errors share position (0, 1), so we build a lookup from
        # auto_errors/human_errors lists to identify them by index.
        score_matrix = {
            (0, 0): 0.9,
            (0, 1): 0.8,
            (0, 2): 0.0,
            (1, 0): 0.85,
            (1, 1): 0.0,
            (1, 2): 0.0,
            (2, 0): 0.0,
            (2, 1): 0.7,
            (2, 2): 0.1,
        }

        # Build span-label -> index maps for the lookup
        auto_label_to_idx = {e.span: i for i, e in enumerate(auto_errors)}
        human_label_to_idx = {e.span: i for i, e in enumerate(human_errors)}

        def custom_objective(
            span1: Tuple[int, int], span2: Tuple[int, int], src: str, tgt: str
        ) -> float:
            # Since all errors share the same (0,1) span, find_optimal passes
            # identical tuples. We need to map back to indices via the outer
            # loop in find_optimal_bipartite_matching. However, the objective
            # only receives spans, not indices. We work around this by
            # returning 1.0 (any overlap) and relying on the score_matrix
            # structure. Instead, we embed the score_matrix directly in the
            # cost matrix via a closure that captures (i, j) from the caller.
            #
            # Actually, we can't do that since find_optimal passes spans.
            # Since ALL spans are (0,1), every pair gets the same objective
            # value — we can't distinguish them by span alone.
            # So we return 1.0 for all overlapping pairs and let severity
            # handle the rest. This test needs restructuring.
            return 1.0  # placeholder — see restructured version below

        # Restructure: give each error a unique position so the objective
        # can distinguish them.
        auto_errors_v2 = [
            Error(span="a0", category="accuracy", severity="major",
                  start=0, end=1, is_source_error=False),
            Error(span="a1", category="accuracy", severity="major",
                  start=10, end=11, is_source_error=False),
            Error(span="a2", category="accuracy", severity="major",
                  start=20, end=21, is_source_error=False),
        ]
        human_errors_v2 = [
            Error(span="h0", category="accuracy", severity="major",
                  start=0, end=1, is_source_error=False),
            Error(span="h1", category="accuracy", severity="major",
                  start=10, end=11, is_source_error=False),
            Error(span="h2", category="accuracy", severity="major",
                  start=20, end=21, is_source_error=False),
        ]

        # Map (start, start) pairs to scores
        pos_score_map = {
            (0, 0): 0.9,
            (0, 10): 0.8,
            (0, 20): 0.0,
            (10, 0): 0.85,
            (10, 10): 0.0,
            (10, 20): 0.0,
            (20, 0): 0.0,
            (20, 10): 0.7,
            (20, 20): 0.1,
        }

        def custom_objective_v2(
            span1: Tuple[int, int], span2: Tuple[int, int], src: str, tgt: str
        ) -> float:
            return pos_score_map.get((span1[0], span2[0]), 0.0)

        auto_errors = auto_errors_v2
        human_errors = human_errors_v2
        custom_objective = custom_objective_v2

        auto_matches, human_matches = find_optimal_bipartite_matching(
            auto_errors, human_errors, "", "", custom_objective
        )

        # Optimal should match all three
        assert len(auto_matches) == 3
        assert len(human_matches) == 3

        # Check optimal assignment: auto_0->human_1, auto_1->human_0, auto_2->human_2
        assert auto_matches[0].matched_index == 1  # auto_0 -> human_1
        assert auto_matches[1].matched_index == 0  # auto_1 -> human_0
        assert auto_matches[2].matched_index == 2  # auto_2 -> human_2

        # Verify total score
        total = sum(
            pos_score_map[(auto_errors[i].start, human_errors[m.matched_index].start)]
            for i, m in auto_matches.items()
        )
        assert total == pytest.approx(1.75)


class TestOptimalMatchesGreedyOnSimpleCases:
    """When greedy is already optimal, both should agree."""

    def test_two_perfect_matches(self):
        """Two auto errors perfectly matching two human errors (no conflict)."""
        src = "source"
        tgt = "aaa bbb ccc ddd"

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

        opt_auto, opt_human = find_optimal_bipartite_matching(
            auto_errors, human_errors, src, tgt, avg_overlap_objective
        )
        greedy_auto, greedy_human = find_greedy_bipartite_matching(
            auto_errors, human_errors, src, tgt
        )

        # Same number of matches
        assert len(opt_auto) == len(greedy_auto)
        # Same assignments
        for idx in opt_auto:
            assert opt_auto[idx].matched_index == greedy_auto[idx].matched_index


class TestOptimalMatchingAsymmetricSizes:
    """More auto errors than human errors and vice versa."""

    def test_more_auto_than_human(self):
        src = "source"
        tgt = "aaa bbb ccc"

        auto_errors = [
            Error(span="aaa", category="accuracy", severity="major",
                  start=0, end=3, is_source_error=False),
            Error(span="bbb", category="accuracy", severity="major",
                  start=4, end=7, is_source_error=False),
            Error(span="ccc", category="accuracy", severity="major",
                  start=8, end=11, is_source_error=False),
        ]
        human_errors = [
            Error(span="aaa", category="accuracy", severity="major",
                  start=0, end=3, is_source_error=False),
        ]

        auto_matches, human_matches = find_optimal_bipartite_matching(
            auto_errors, human_errors, src, tgt, avg_overlap_objective
        )

        # Only one human error → at most one match
        assert len(auto_matches) == 1
        assert len(human_matches) == 1
        assert auto_matches[0].matched_index == 0
        assert auto_matches[0].avg_overlap == 1.0

    def test_more_human_than_auto(self):
        src = "source"
        tgt = "aaa bbb ccc"

        auto_errors = [
            Error(span="bbb", category="accuracy", severity="major",
                  start=4, end=7, is_source_error=False),
        ]
        human_errors = [
            Error(span="aaa", category="accuracy", severity="major",
                  start=0, end=3, is_source_error=False),
            Error(span="bbb", category="accuracy", severity="major",
                  start=4, end=7, is_source_error=False),
            Error(span="ccc", category="accuracy", severity="major",
                  start=8, end=11, is_source_error=False),
        ]

        auto_matches, human_matches = find_optimal_bipartite_matching(
            auto_errors, human_errors, src, tgt, avg_overlap_objective
        )

        # Only one auto error → at most one match
        assert len(auto_matches) == 1
        assert len(human_matches) == 1
        assert auto_matches[0].matched_index == 1  # matches human "bbb"
        assert auto_matches[0].avg_overlap == 1.0
