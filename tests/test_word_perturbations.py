# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Tests for progressive length perturbation with language-aware character scaling.

Run with: python -m pytest tests/test_word_perturbations.py -v
"""

import pytest

from mt_evaluation.core import Sample, AutomaticEvaluation, Error
from mt_evaluation.meta_evaluation.span_level.perturbations import (
    _get_base_lang_code,
    _get_error_lang_code,
    _is_character_level_lang,
    _SPACE_DELIMITED_CHAR_SCALE,
    increase_spans_length_by_n,
)


# ---------------------------------------------------------------------------
# Helper to build a minimal Sample with one error
# ---------------------------------------------------------------------------

def _make_sample(src: str, tgt: str, errors: list[Error]) -> Sample:
    score = sum(e.score for e in errors if e.score)
    return Sample(
        src=src,
        tgt=tgt,
        src_lang="English",
        tgt_lang="German",
        evaluation=AutomaticEvaluation(
            score=score,
            errors=errors,
            annotation="",
            parsing_error=False,
        ),
    )


# ===========================================================================
# Unit tests for helper functions
# ===========================================================================


class TestGetBaseLangCode:
    def test_plain(self):
        assert _get_base_lang_code("en") == "en"

    def test_with_region(self):
        assert _get_base_lang_code("zh_CN") == "zh"
        assert _get_base_lang_code("ko_KR") == "ko"


class TestGetErrorLangCode:
    def test_source_side(self):
        assert _get_error_lang_code("en-de", is_source_error=True) == "en"

    def test_target_side(self):
        assert _get_error_lang_code("en-de", is_source_error=False) == "de"

    def test_target_with_region(self):
        assert _get_error_lang_code("en-zh_CN", is_source_error=False) == "zh_CN"

    def test_ja_zh(self):
        assert _get_error_lang_code("ja-zh_CN", is_source_error=True) == "ja"
        assert _get_error_lang_code("ja-zh_CN", is_source_error=False) == "zh_CN"


class TestIsCharacterLevelLang:
    def test_chinese(self):
        assert _is_character_level_lang("zh") is True
        assert _is_character_level_lang("zh_CN") is True

    def test_japanese(self):
        assert _is_character_level_lang("ja") is True
        assert _is_character_level_lang("ja_JP") is True

    def test_space_delimited(self):
        assert _is_character_level_lang("en") is False
        assert _is_character_level_lang("de") is False
        assert _is_character_level_lang("es") is False
        assert _is_character_level_lang("ko") is False
        assert _is_character_level_lang("ko_KR") is False


class TestScaleConstant:
    def test_scale_is_positive_int(self):
        assert isinstance(_SPACE_DELIMITED_CHAR_SCALE, int)
        assert _SPACE_DELIMITED_CHAR_SCALE > 0

    def test_scale_value(self):
        assert _SPACE_DELIMITED_CHAR_SCALE == 5


# ===========================================================================
# Integration tests for increase_spans_length_by_n
# ===========================================================================


class TestIncreaseSpansLengthByN:
    SCALE = _SPACE_DELIMITED_CHAR_SCALE  # 5

    def test_english_target_expand_by_1(self):
        # n=1 for English → delta = 1*5 = 5 characters each side
        tgt = "The quick brown fox jumps over the lazy dog"
        error = Error(
            span="fox",
            category="accuracy",
            severity="major",
            start=16,
            end=19,
            is_source_error=False,
            score=-5.0,
        )
        sample = _make_sample("source text", tgt, [error])

        result = increase_spans_length_by_n(sample, n=1, lp="en-de")

        new_error = result.evaluation.errors[0]
        # start: 16-5=11, end: 19+5=24
        assert new_error.start == 11
        assert new_error.end == 24
        assert new_error.span == tgt[11:24]
        assert new_error.span in tgt

    def test_english_source_error_expand(self):
        # n=2 for English source → delta = 2*5 = 10 characters each side
        src = "The quick brown fox jumps over the lazy dog"
        error = Error(
            span="fox",
            category="accuracy",
            severity="major",
            start=16,
            end=19,
            is_source_error=True,
            score=-5.0,
        )
        sample = _make_sample(src, "target text", [error])

        result = increase_spans_length_by_n(sample, n=2, lp="en-de")

        new_error = result.evaluation.errors[0]
        # start: 16-10=6, end: 19+10=29
        assert new_error.start == 6
        assert new_error.end == 29
        assert new_error.span == src[6:29]
        assert new_error.span in src

    def test_chinese_character_level_expansion(self):
        # Chinese target: n=2 → delta = 2 (no scaling)
        tgt = "这是一个测试翻译句子"
        #       0123456789
        error = Error(
            span="测试",
            category="accuracy",
            severity="major",
            start=4,
            end=6,
            is_source_error=False,
            score=-5.0,
        )
        sample = _make_sample("source text", tgt, [error])

        result = increase_spans_length_by_n(sample, n=2, lp="en-zh_CN")

        new_error = result.evaluation.errors[0]
        assert new_error.start == 2
        assert new_error.end == 8
        assert new_error.span == "一个测试翻译"
        assert new_error.span in tgt

    def test_japanese_character_level_expansion(self):
        # Japanese target: n=1 → delta = 1 (no scaling)
        tgt = "これはテスト翻訳文です"
        #       01234567890
        error = Error(
            span="テスト",
            category="fluency",
            severity="minor",
            start=3,
            end=6,
            is_source_error=False,
            score=-1.0,
        )
        sample = _make_sample("source text", tgt, [error])

        # Error is on target side → zh_CN → character-level
        result = increase_spans_length_by_n(sample, n=1, lp="ja-zh_CN")

        new_error = result.evaluation.errors[0]
        assert new_error.start == 2
        assert new_error.end == 7
        assert new_error.span == "はテスト翻"
        assert new_error.span in tgt

    def test_korean_scaled_expansion(self):
        # Korean is space-delimited → n=1 → delta = 5
        tgt = "이것은 테스트 번역 문장입니다"
        error = Error(
            span="번역",
            category="accuracy",
            severity="major",
            start=8,
            end=10,
            is_source_error=False,
            score=-5.0,
        )
        sample = _make_sample("source text", tgt, [error])

        result = increase_spans_length_by_n(sample, n=1, lp="en-ko_KR")

        new_error = result.evaluation.errors[0]
        # start: 8-5=3, end: 10+5=15
        assert new_error.start == 3
        assert new_error.end == 15
        assert new_error.span == tgt[3:15]
        assert new_error.span in tgt

    def test_german_scaled_expansion(self):
        # German: n=1 → delta = 5
        tgt = "Der schnelle braune Fuchs springt über den faulen Hund"
        # Fuchs is at index 20:25
        error = Error(
            span="Fuchs",
            category="fluency",
            severity="minor",
            start=20,
            end=25,
            is_source_error=False,
            score=-1.0,
        )
        sample = _make_sample("source text", tgt, [error])

        result = increase_spans_length_by_n(sample, n=1, lp="en-de")

        new_error = result.evaluation.errors[0]
        # start: 20-5=15, end: 25+5=30
        assert new_error.start == 15
        assert new_error.end == 30
        assert new_error.span == tgt[15:30]
        assert new_error.span in tgt

    def test_clamps_at_start(self):
        tgt = "Hello world test"
        error = Error(
            span="Hello",
            category="accuracy",
            severity="major",
            start=0,
            end=5,
            is_source_error=False,
            score=-5.0,
        )
        sample = _make_sample("source", tgt, [error])

        result = increase_spans_length_by_n(sample, n=1, lp="en-de")

        new_error = result.evaluation.errors[0]
        # start: max(0-5, 0)=0, end: min(5+5, 16)=10
        assert new_error.start == 0
        assert new_error.end == 10
        assert new_error.span == tgt[0:10]

    def test_clamps_at_end(self):
        tgt = "Hello world test"
        error = Error(
            span="test",
            category="accuracy",
            severity="major",
            start=12,
            end=16,
            is_source_error=False,
            score=-5.0,
        )
        sample = _make_sample("source", tgt, [error])

        result = increase_spans_length_by_n(sample, n=1, lp="en-de")

        new_error = result.evaluation.errors[0]
        # start: 12-5=7, end: min(16+5, 16)=16
        assert new_error.start == 7
        assert new_error.end == 16
        assert new_error.span == tgt[7:16]

    def test_does_not_mutate_original(self):
        tgt = "The quick brown fox jumps"
        error = Error(
            span="fox",
            category="accuracy",
            severity="major",
            start=16,
            end=19,
            is_source_error=False,
            score=-5.0,
        )
        sample = _make_sample("source text", tgt, [error])

        result = increase_spans_length_by_n(sample, n=2, lp="en-de")

        # Original sample unchanged
        assert sample.evaluation.errors[0].span == "fox"
        assert sample.evaluation.errors[0].start == 16
        assert sample.evaluation.errors[0].end == 19
        # Result changed
        assert result.evaluation.errors[0].span != "fox"

    def test_multiple_errors(self):
        tgt = "The quick brown fox jumps over the lazy dog"
        errors = [
            Error(
                span="quick",
                category="accuracy",
                severity="major",
                start=4,
                end=9,
                is_source_error=False,
                score=-5.0,
            ),
            Error(
                span="lazy",
                category="fluency",
                severity="minor",
                start=35,
                end=39,
                is_source_error=False,
                score=-1.0,
            ),
        ]
        sample = _make_sample("source text", tgt, errors)

        result = increase_spans_length_by_n(sample, n=1, lp="en-de")

        e0 = result.evaluation.errors[0]
        # start: max(4-5, 0)=0, end: 9+5=14
        assert e0.start == 0
        assert e0.end == 14
        assert e0.span == tgt[0:14]
        assert e0.span in tgt

        e1 = result.evaluation.errors[1]
        # start: 35-5=30, end: min(39+5, 43)=43
        assert e1.start == 30
        assert e1.end == 43
        assert e1.span == tgt[30:43]  # " the lazy dog" (with leading space)
        assert e1.span in tgt
