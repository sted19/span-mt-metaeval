# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Core data structures for the MT Evaluation Framework.

This module defines the fundamental data structures used throughout the MT evaluation
framework, including prompts, evaluations, errors, and samples.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from mt_evaluation.core.model_io import FewShots


@dataclass
class Prompt:
    """
    A container for system and user prompt templates.

    This class holds the prompt templates that are used to format evaluation requests
    to language models. It supports both system and user prompts, as well as few-shot
    examples for in-context learning.

    Attributes:
        user_prompt: The user prompt template string with placeholders for formatting.
        system_prompt: Optional system prompt template string.
        few_shots: List of few-shot examples for in-context learning.

    Example:
        >>> prompt = Prompt(
        ...     user_prompt="Evaluate this translation from {src_lang} to {tgt_lang}:\\nSource: {src}\\nTarget: {tgt}",
        ...     system_prompt="You are an expert translation evaluator."
        ... )
    """

    user_prompt: str
    system_prompt: Optional[str] = field(default=None)
    few_shots: List[Dict[str, str]] = field(default_factory=list)

    def format_system_prompt(self, **kwargs) -> str:
        """
        Format the system prompt with provided keyword arguments.

        Args:
            **kwargs: Keyword arguments to format the system prompt template.

        Returns:
            str: The formatted system prompt.

        Raises:
            ValueError: If system_prompt is None.
        """
        if self.system_prompt is None:
            raise ValueError("System prompt is None")
        return self.system_prompt.format(**kwargs)

    def format_user_prompt(
        self, src_lang: str, tgt_lang: str, src: str, tgt: str, **kwargs
    ) -> str:
        """
        Format the user prompt with translation data and additional arguments.

        Args:
            src_lang: Source language name.
            tgt_lang: Target language name.
            src: Source text to be evaluated.
            tgt: Target text to be evaluated.
            **kwargs: Additional keyword arguments for formatting.

        Returns:
            str: The formatted user prompt.
        """
        return self.user_prompt.format(
            src_lang=src_lang, tgt_lang=tgt_lang, src=src, tgt=tgt, **kwargs
        )

    def format_few_shots(self) -> FewShots:
        """
        Format few-shot examples into a FewShots object.

        Returns:
            FewShots: Formatted few-shot examples ready for model input.
        """
        user_prompts, responses = [], []
        for shot in self.few_shots:
            src_lang, tgt_lang, src, tgt = (
                shot["src_lang"],
                shot["tgt_lang"],
                shot["src"],
                shot["tgt"],
            )

            shot_user_prompt = self.format_user_prompt(src_lang, tgt_lang, src, tgt)

            user_prompts.append(shot_user_prompt)
            responses.append(shot["response"])

        return FewShots(user_prompts=user_prompts, assistant_responses=responses)


@dataclass
class Error:
    """
    Represents a translation error identified during evaluation.

    This class encapsulates information about a specific error found in a translation,
    including the problematic text span, error category, severity level, and optional
    explanation.

    Attributes:
        span: The text span containing the error.
        category: The category/type of error (e.g., "accuracy", "fluency").
        severity: The severity level (e.g., "critical", "major", "minor").
        start: Optional start index of the error span in the text.
        end: Optional end index of the error span in the text.
        is_source_error: Whether the error is in the source (vs target) text.
        score: Optional score for this error.
        explanation: Optional detailed explanation of the error.
        extended_span: Optional extended span for context disambiguation.
    """

    span: str
    category: str
    severity: str
    # start, end, is_source_error, and score are all optional. The reason is that
    # I always have them for human annotations, but I might have issues filling them
    # with automatic annotations. Specifically, I might not be able to retrieve the
    # correct start and end indices starting from the span. As a consequence, I might
    # not know whether I'm dealing with an error in the source or the target. Finally,
    # scores-per-error are provided by human annotations, but I might prefer computing
    # them post annotation for automatic ones
    start: Optional[int] = None
    end: Optional[int] = None
    is_source_error: Optional[bool] = None
    score: Optional[float] = None
    explanation: Optional[str] = None
    # Could be useful to derive start/end positions of errors when there are multiple
    # matches for the same span
    extended_span: Optional[str] = None

    def __str__(self) -> str:
        """Return string representation of the error."""
        return (
            f"Error("
            f"span={self.span}, "
            f"start={self.start}, "
            f"end={self.end}, "
            f"category={self.category}, "
            f"severity={self.severity}, "
            f"is_source_error={self.is_source_error}, "
            f"score={self.score}, "
            f"explanation={self.explanation}"
            f"extended_span={self.extended_span}"
            f")"
        )

    def __repr__(self) -> str:
        """Return string representation of the error."""
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert error to dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the error.
        """
        return {
            "start": self.start,
            "end": self.end,
            "is_source_error": self.is_source_error,
            "span": self.span,
            "category": self.category,
            "severity": self.severity,
            "score": self.score,
            "explanation": self.explanation,
            "extended_span": self.extended_span,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Error":
        """
        Create Error instance from dictionary.

        Args:
            data: Dictionary containing error data.

        Returns:
            Error: New Error instance.
        """
        return cls(**data)


@dataclass
class Evaluation:
    """Base class for evaluation results."""
    score: float
    errors: List[Error]


@dataclass
class HumanEvaluation(Evaluation):
    """
    Human evaluation result with optional rater information.

    Attributes:
        score: The overall evaluation score.
        errors: List of errors identified in the evaluation.
        rater: Optional identifier for the human rater.
    """
    score: float
    errors: List[Error]
    rater: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert evaluation to dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the evaluation.
        """
        return {
            "score": self.score,
            "errors": [error.to_dict() for error in self.errors],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanEvaluation":
        """
        Create HumanEvaluation instance from dictionary.

        Args:
            data: Dictionary containing evaluation data.

        Returns:
            HumanEvaluation: New HumanEvaluation instance.
        """
        errors = [Error.from_dict(error) for error in data["errors"]]
        return cls(score=data["score"], errors=errors)


@dataclass
class AutomaticEvaluation(Evaluation):
    """
    Represents the complete evaluation result for a translation sample.

    This class contains all information about an evaluation, including the raw
    annotation, parsed errors, computed score, and metadata about the evaluation
    process.

    Attributes:
        score: Numerical score for the translation.
        errors: List of identified errors.
        annotation: Raw annotation text from the evaluator.
        parsing_error: Whether there was an error parsing the response.
        user_prompt: The user prompt used for evaluation.
        system_prompt: The system prompt used for evaluation.
        few_shots: Few-shot examples used for evaluation.
        cost: Cost of the evaluation (e.g., API cost).
    """

    score: float
    errors: List[Error]
    annotation: str
    parsing_error: bool | None
    user_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    few_shots: Optional[FewShots] = None
    cost: Optional[float] = None

    def __str__(self) -> str:
        """Return string representation of the evaluation."""
        return f"Evaluation(annotation={self.annotation}, errors={self.errors}, score={self.score})"

    def __repr__(self) -> str:
        """Return string representation of the evaluation."""
        return self.__str__()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert evaluation to dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the evaluation.
        """
        return {
            "annotation": self.annotation,
            "errors": [error.to_dict() for error in self.errors],
            "score": self.score,
            "user_prompt": self.user_prompt,
            "system_prompt": self.system_prompt,
            "few_shots": self.few_shots.to_dict() if self.few_shots else None,
            "cost": self.cost,
            "parsing_error": self.parsing_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutomaticEvaluation":
        """
        Create Evaluation instance from dictionary.

        Args:
            data: Dictionary containing evaluation data.

        Returns:
            AutomaticEvaluation: New Evaluation instance.
        """
        errors = [Error.from_dict(error) for error in data["errors"]]
        return cls(
            annotation=data.get("annotation"),
            errors=errors,
            score=data.get("score"),
            parsing_error=data.get("parsing_error", None),
            user_prompt=data.get("user_prompt"),
            system_prompt=data.get("system_prompt"),
            few_shots=(
                FewShots.from_dict(data["few_shots"]) if data.get("few_shots") else None
            ),
            cost=data.get("cost"),
        )


@dataclass
class Sample:
    """
    Represents a translation sample for evaluation.

    This class encapsulates a source text, its translation, language information,
    and optionally the evaluation result.

    Attributes:
        src: Source text.
        tgt: Target (translated) text.
        src_lang: Source language name.
        tgt_lang: Target language name.
        evaluation: Optional automatic evaluation result.
        human_evaluation: Optional human evaluation result.
        doc_id: Optional document identifier.
        seg_id: Optional segment identifier within the document.
        src_doc: Optional source document context.
        tgt_annotated: Optional annotated target text (for WMT2025 data).
    """

    src: str
    tgt: str
    src_lang: str
    tgt_lang: str
    evaluation: Optional[AutomaticEvaluation] = None
    human_evaluation: Optional[HumanEvaluation] = None
    doc_id: Optional[str] = None
    seg_id: Optional[int] = None
    src_doc: Optional[str] = None
    # to store additional fields from the wmt2025 data
    tgt_annotated: Optional[str] = None

    def _get_input_tuple(self) -> Tuple[str, str, str, str]:
        """
        Get tuple of input fields for hashing and comparison.

        Returns:
            Tuple[str, str, str, str]: Tuple of (src, tgt, src_lang, tgt_lang).
        """
        return self.src, self.tgt, self.src_lang, self.tgt_lang

    def get_input_hash(self) -> str:
        """
        Generate a string hash for cache keys.

        Returns:
            str: Hash string based on input fields.
        """
        return str(hash(self._get_input_tuple()))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert sample to dictionary for JSON serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the sample.
        """
        return {
            "src_doc": self.src_doc,
            "src": self.src,
            "tgt": self.tgt,
            "src_lang": self.src_lang,
            "tgt_lang": self.tgt_lang,
            "evaluation": (self.evaluation.to_dict() if self.evaluation else None),
            "human_evaluation": (
                self.human_evaluation.to_dict() if self.human_evaluation else None
            ),
            "doc_id": self.doc_id,
            "seg_id": self.seg_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        """
        Create Sample instance from dictionary for JSON deserialization.

        Args:
            data: Dictionary containing sample data.

        Returns:
            Sample: New Sample instance.
        """
        evaluation = None
        if data.get("evaluation"):
            evaluation = AutomaticEvaluation.from_dict(data["evaluation"])

        human_evaluation = None
        if data.get("human_evaluation"):
            human_evaluation = HumanEvaluation.from_dict(data["human_evaluation"])

        return cls(
            src_doc=data.get("src_doc"),
            src=data["src"],
            tgt=data["tgt"],
            src_lang=data["src_lang"],
            tgt_lang=data["tgt_lang"],
            evaluation=evaluation,
            human_evaluation=human_evaluation,
            doc_id=data.get("doc_id"),
            seg_id=data.get("seg_id"),
        )

    def __repr__(self) -> str:
        """Readable multi-line representation for easy inspection."""
        trunc = lambda s, n=80: (s[:n] + "…") if s and len(s) > n else s
        lines = [
            "Sample(",
            f"  src_lang={self.src_lang!r} → tgt_lang={self.tgt_lang!r},",
            f"  doc_id={self.doc_id!r}, seg_id={self.seg_id!r},",
            f"  src={trunc(self.src)!r},",
            f"  tgt={trunc(self.tgt)!r},",
        ]
        if self.evaluation:
            e = self.evaluation
            lines.append(
                f"  evaluation=AutomaticEvaluation(score={e.score}, "
                f"errors={len(e.errors)}, parsing_error={e.parsing_error}),"
            )
        else:
            lines.append("  evaluation=None,")
        if self.human_evaluation:
            h = self.human_evaluation
            lines.append(
                f"  human_evaluation=HumanEvaluation(score={h.score}, "
                f"errors={len(h.errors)}, rater={h.rater!r}),"
            )
        else:
            lines.append("  human_evaluation=None,")
        lines.append(")")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return the same readable representation as repr."""
        return self.__repr__()

    def __hash__(self) -> int:
        """
        Hash for use in sets/dicts based on input fields only.

        Returns:
            int: Hash value.
        """
        return hash(self._get_input_tuple())

    def __eq__(self, other) -> bool:
        """
        Check equality based on input fields only (not evaluation).

        Args:
            other: Object to compare with.

        Returns:
            bool: True if input fields are equal, False otherwise.
        """
        if not isinstance(other, Sample):
            return False
        return self._get_input_tuple() == other._get_input_tuple()
