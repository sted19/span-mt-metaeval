# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Smoke tests for MT Evaluation Framework.

These tests verify basic functionality to catch obvious breakages during refactoring.
Run with: python -m pytest tests/test_smoke.py -v
"""

import pytest
import json
import tempfile
import os
from pathlib import Path


class TestCoreDataStructures:
    """Test core data structures serialize and deserialize correctly."""

    def test_error_roundtrip(self):
        """Test Error serialization roundtrip."""
        from mt_evaluation.core import Error

        error = Error(
            span="test span",
            category="accuracy-mistranslation",
            severity="major",
            start=0,
            end=9,
            is_source_error=False,
            score=-5.0,
            explanation="This is a test error",
            extended_span="extended test span",
        )

        # Serialize
        error_dict = error.to_dict()
        assert error_dict["span"] == "test span"
        assert error_dict["severity"] == "major"

        # Deserialize
        restored_error = Error.from_dict(error_dict)
        assert restored_error.span == error.span
        assert restored_error.category == error.category
        assert restored_error.severity == error.severity
        assert restored_error.start == error.start
        assert restored_error.end == error.end

    def test_sample_roundtrip(self):
        """Test Sample serialization roundtrip."""
        from mt_evaluation.core import Sample, HumanEvaluation, AutomaticEvaluation, Error

        # Create sample with evaluations
        error = Error(
            span="fehler",
            category="fluency-grammar",
            severity="minor",
            start=0,
            end=6,
            is_source_error=False,
            score=-1.0,
        )

        human_eval = HumanEvaluation(score=-1.0, errors=[error], rater="rater1")
        auto_eval = AutomaticEvaluation(
            score=-1.0,
            errors=[error],
            annotation="test annotation",
            parsing_error=False,
        )

        sample = Sample(
            src="Hello world",
            tgt="Hallo Welt",
            src_lang="English",
            tgt_lang="German",
            evaluation=auto_eval,
            human_evaluation=human_eval,
            doc_id="doc1",
            seg_id=1,
        )

        # Serialize
        sample_dict = sample.to_dict()
        assert sample_dict["src"] == "Hello world"
        assert sample_dict["tgt"] == "Hallo Welt"

        # Deserialize
        restored_sample = Sample.from_dict(sample_dict)
        assert restored_sample.src == sample.src
        assert restored_sample.tgt == sample.tgt
        assert restored_sample.evaluation.score == sample.evaluation.score
        assert len(restored_sample.evaluation.errors) == 1

    def test_sample_hash_equality(self):
        """Test Sample hashing based on input fields only."""
        from mt_evaluation.core import Sample

        sample1 = Sample(
            src="Hello", tgt="Hallo", src_lang="English", tgt_lang="German"
        )
        sample2 = Sample(
            src="Hello", tgt="Hallo", src_lang="English", tgt_lang="German"
        )
        sample3 = Sample(
            src="Hello!", tgt="Hallo", src_lang="English", tgt_lang="German"
        )

        # Same inputs should be equal
        assert sample1 == sample2
        assert hash(sample1) == hash(sample2)

        # Different inputs should not be equal
        assert sample1 != sample3

    def test_few_shots_roundtrip(self):
        """Test FewShots serialization roundtrip."""
        from mt_evaluation.core import FewShots

        few_shots = FewShots(
            user_prompts=["prompt1", "prompt2"],
            assistant_responses=["response1", "response2"],
        )

        # Serialize
        few_shots_dict = few_shots.to_dict()
        assert len(few_shots_dict["user_prompts"]) == 2

        # Deserialize
        restored = FewShots.from_dict(few_shots_dict)
        assert restored.user_prompts == few_shots.user_prompts
        assert restored.assistant_responses == few_shots.assistant_responses


class TestFactoryFunctions:
    """Test factory functions for models and autoevals."""

    def test_get_autoeval_returns_class(self):
        """Test that get_autoeval returns a valid class."""
        from mt_evaluation.autoevals.factory import get_autoeval, list_available_evaluators

        # Should return a class
        evaluator_class = get_autoeval("gemba-mqm")
        assert evaluator_class is not None

        # List should return available evaluators
        available = list_available_evaluators()
        assert "gemba-mqm" in available
        assert "unified-mqm-boosted-v5" in available

    def test_get_autoeval_invalid_raises(self):
        """Test that invalid schema raises ValueError."""
        from mt_evaluation.autoevals.factory import get_autoeval

        with pytest.raises(ValueError):
            get_autoeval("nonexistent-schema")

    def test_get_model_returns_class(self):
        """Test that get_model returns a valid class."""
        from mt_evaluation.models.factory import get_model, list_available_models

        # Should return a class for a valid model
        model_class = get_model("us.anthropic.claude-3-5-haiku-20241022-v1:0")
        assert model_class is not None

        # List should return available models
        available = list_available_models()
        assert len(available) > 0

    def test_get_model_invalid_raises(self):
        """Test that invalid model raises ValueError."""
        from mt_evaluation.models.factory import get_model

        with pytest.raises(ValueError):
            get_model("nonexistent-model")


class TestCache:
    """Test MTEvaluationCache functionality."""

    def test_cache_add_and_retrieve(self):
        """Test adding and retrieving evaluations from cache."""
        from mt_evaluation.data.cache import MTEvaluationCache
        from mt_evaluation.core import Sample, AutomaticEvaluation, Error

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.jsonl")
            cache = MTEvaluationCache(cache_file, save_frequency=1)

            # Create sample and evaluation
            sample = Sample(
                src="Test source",
                tgt="Test target",
                src_lang="English",
                tgt_lang="German",
            )
            evaluation = AutomaticEvaluation(
                score=-5.0,
                errors=[
                    Error(
                        span="target",
                        category="accuracy",
                        severity="major",
                        score=-5.0,
                    )
                ],
                annotation="test",
                parsing_error=False,
            )

            # Should not be evaluated initially
            assert not cache.is_evaluated(sample)

            # Add evaluation
            cache.add_evaluation(sample, evaluation)

            # Should now be evaluated
            assert cache.is_evaluated(sample)

            # Retrieved evaluation should match
            retrieved = cache.get_evaluation(sample)
            assert retrieved.score == evaluation.score
            assert len(retrieved.errors) == 1

            # Finalize cache
            cache.finalize()

            # Verify file was written
            assert os.path.exists(cache_file)

    def test_cache_persistence(self):
        """Test that cache persists across instances."""
        from mt_evaluation.data.cache import MTEvaluationCache
        from mt_evaluation.core import Sample, AutomaticEvaluation

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "test_cache.jsonl")

            # First instance - add evaluation
            cache1 = MTEvaluationCache(cache_file, save_frequency=1)
            sample = Sample(
                src="Persistent test",
                tgt="Persistenter Test",
                src_lang="English",
                tgt_lang="German",
            )
            evaluation = AutomaticEvaluation(
                score=0.0, errors=[], annotation="", parsing_error=False
            )
            cache1.add_evaluation(sample, evaluation)
            cache1.finalize()

            # Second instance - should find the evaluation
            cache2 = MTEvaluationCache(cache_file, save_frequency=1)
            assert cache2.is_evaluated(sample)
            assert len(cache2) == 1


class TestLanguageCodes:
    """Test language code mappings."""

    def test_get_language_name(self):
        """Test getting language names from codes."""
        from mt_evaluation.data.language_codes import get_language_name, get_language_name_safe

        assert get_language_name("en") == "English"
        assert get_language_name("de") == "German"
        assert get_language_name("zh_CN") == "Chinese"

        # Safe version should return default for unknown
        assert get_language_name_safe("xx", "Unknown") == "Unknown"

        # Regular version should raise for unknown
        with pytest.raises(ValueError):
            get_language_name("xx")


class TestParsingUtils:
    """Test JSON parsing utilities."""

    def test_extract_json_response_valid(self):
        """Test extracting JSON from markdown code blocks."""
        from mt_evaluation.autoevals.utils import extract_json_response

        text = '''Here is my analysis:

```json
[{"span": "test", "category": "accuracy", "severity": "major"}]
```

That's all.'''

        result = extract_json_response(text)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["span"] == "test"

    def test_extract_json_response_no_json(self):
        """Test that missing JSON raises ValueError."""
        from mt_evaluation.autoevals.utils import extract_json_response

        with pytest.raises(ValueError):
            extract_json_response("No JSON here")


class TestConstants:
    """Test that constants are properly defined."""

    def test_wmt_language_pairs(self):
        """Test WMT language pair constants."""
        from mt_evaluation.core import wmt22_lps, wmt23_lps, wmt24_lps, wmt25_lps

        assert "en-de" in wmt22_lps
        assert "en-de" in wmt23_lps
        assert "en-de" in wmt24_lps
        assert len(wmt25_lps) > 0

    def test_severity_constants(self):
        """Test severity constants."""
        from mt_evaluation.core import all_severities

        assert "minor" in all_severities
        assert "major" in all_severities
        assert "critical" in all_severities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
