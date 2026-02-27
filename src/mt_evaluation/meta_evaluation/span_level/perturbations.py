# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from collections import defaultdict
from typing import Dict, List, Set, Optional, Tuple
import numpy as np

from mt_evaluation.core import Sample, Error
from mt_evaluation.meta_evaluation.metrics_to_evaluate import (
    autoevals_with_no_extended_spans,
)

# Available perturbation types
PERTURBATION_EXT_ONLY = "EXT_ONLY"
PERTURBATION_RAND_REMOVE = "RAND_REMOVE"
PERTURBATION_REMOVE_ERRORS_IN_SAMPLES_WITH_1 = "REMOVE_ALL_1"
PERTURBATION_PROGRESSIVE_LENGTH = "PROGRESSIVE_LENGTH"

ALL_PERTURBATIONS = {
    PERTURBATION_EXT_ONLY,
    PERTURBATION_RAND_REMOVE,
    PERTURBATION_REMOVE_ERRORS_IN_SAMPLES_WITH_1,
    PERTURBATION_PROGRESSIVE_LENGTH,
}


# Languages where each character maps roughly to one "word" unit.
# For these scripts n characters ≈ n words, so the expansion step is 1.
# For space-delimited languages the step is scaled by
# _SPACE_DELIMITED_CHAR_SCALE (≈ average word length) so that
# "expand by n" is comparable across scripts.
_CHARACTER_LEVEL_LANGS = {"zh", "ja"}

# Average word length (in characters) for space-delimited languages
# (English ≈ 4.7, Spanish ≈ 5.2, German ≈ 5.8 → mean ≈ 5).
_SPACE_DELIMITED_CHAR_SCALE = 5


def _get_base_lang_code(lang_code: str) -> str:
    """Strip regional suffix (e.g. 'ko_KR' -> 'ko', 'zh_CN' -> 'zh')."""
    return lang_code.split("_")[0]


def _get_error_lang_code(lp: str, is_source_error: bool) -> str:
    """Return the language code for the side of the error.

    Args:
        lp: Language pair string, e.g. 'en-de', 'ja-zh_CN'.
        is_source_error: Whether the error is on the source side.
    """
    src_code, tgt_code = lp.split("-", 1)
    return src_code if is_source_error else tgt_code


def _is_character_level_lang(lang_code: str) -> bool:
    """Return True for CJK languages where each character ≈ one word (Chinese, Japanese)."""
    return _get_base_lang_code(lang_code) in _CHARACTER_LEVEL_LANGS


def increase_spans_length_by_n(sample: Sample, n: int, lp: str) -> Sample:
    """Expand every error span by *n* units on each side.

    For character-level languages (zh, ja) the span grows by *n* characters
    on each side.  For space-delimited languages (en, es, de, ko, …) the
    span grows by ``n * _SPACE_DELIMITED_CHAR_SCALE`` characters on each
    side so that one unit of *n* roughly corresponds to one word.
    """
    assert sample.evaluation is not None

    new_sample = sample.from_dict(sample.to_dict())

    for error in new_sample.evaluation.errors:
        start, end = error.start, error.end
        lang_code = _get_error_lang_code(lp, error.is_source_error)
        text = sample.src if error.is_source_error else sample.tgt

        if _is_character_level_lang(lang_code):
            delta = n
        else:
            delta = n * _SPACE_DELIMITED_CHAR_SCALE

        new_start = max(start - delta, 0)
        new_end = min(end + delta, len(text))

        error.start = new_start
        error.end = new_end
        error.span = text[new_start:new_end]

    return new_sample


def remove_errors_from_samples_with_1_error(sample: Sample) -> Tuple[Sample, int, bool]:
    assert sample.evaluation is not None or sample.human_evaluation is not None

    new_sample = sample.from_dict(sample.to_dict())

    removed_1 = False
    if new_sample.evaluation is not None:
        num_errors = len(new_sample.evaluation.errors)
        if num_errors == 1:
            new_sample.evaluation.errors = []
            new_sample.evaluation.score = 0
            removed_1 = True
    else:
        num_errors = len(new_sample.human_evaluation.errors)
        if num_errors == 1:
            new_sample.human_evaluation.errors = []
            new_sample.human_evaluation.score = 0
            removed_1 = True

    return new_sample, num_errors, removed_1


def remove_extended_span_from_sample_errors(sample: Sample) -> Sample:
    assert sample.evaluation is not None

    new_sample = sample.from_dict(sample.to_dict())

    for error in new_sample.evaluation.errors:
        error.extended_span = None

    return new_sample


def switch_spans_for_extended_spans(sample: Sample) -> Sample:
    assert sample.evaluation is not None

    new_sample = sample.from_dict(sample.to_dict())

    for error in new_sample.evaluation.errors:
        if not error.extended_span:
            continue

        if error.is_source_error:
            new_start = sample.src.find(error.extended_span)
            if new_start != -1:
                new_end = min(new_start + len(error.extended_span), len(sample.src))
            else:
                continue
        else:
            new_start = sample.tgt.find(error.extended_span)
            if new_start != -1:
                new_end = min(new_start + len(error.extended_span), len(sample.tgt))
            else:
                continue

        error.span = error.extended_span
        error.start = new_start
        error.end = new_end
        error.extended_span = None

    return new_sample


def remove_spans_with_probability_p(
    sample: Sample, p: float, random_numbers: List[float]
) -> Sample:
    assert sample.evaluation is not None
    assert len(random_numbers) == len(sample.evaluation.errors)

    new_sample = sample.from_dict(sample.to_dict())
    errors = [
        e for num, e in zip(random_numbers, new_sample.evaluation.errors) if num > p
    ]
    new_sample.evaluation.errors = errors
    new_sample.evaluation.score = sum(e.score for e in errors)

    return new_sample


def extract_perturbations_from_autoevals(
    autoeval2lp2preprocessed_samples_with_automatic_evaluations: Dict[
        str, Dict[str, List[Sample]]
    ],
    enabled_perturbations: Optional[Set[str]] = None,
) -> Tuple[Dict[str, Dict[str, List[Sample]]], Dict[str, Dict[str, Dict[str, int]]]]:
    """
    Extract perturbations from auto-evaluations.

    Args:
        autoeval2lp2preprocessed_samples_with_automatic_evaluations: Dictionary of auto-evaluations
        enabled_perturbations: Set of perturbation types to generate. If None or empty,
                               no perturbations are generated (only original samples are kept).
                               Valid values: EXT_ONLY, RAND_REMOVE, REMOVE_ALL_1, PROGRESSIVE_LENGTH

    Returns:
        Dictionary with original samples and (optionally) perturbed samples
    """
    # If no perturbations enabled, return the original data as-is
    if not enabled_perturbations:
        return autoeval2lp2preprocessed_samples_with_automatic_evaluations

    # Validate perturbation names
    invalid_perturbations = enabled_perturbations - ALL_PERTURBATIONS
    if invalid_perturbations:
        raise ValueError(
            f"Invalid perturbation types: {invalid_perturbations}. "
            f"Valid types are: {ALL_PERTURBATIONS}"
        )

    perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations = defaultdict(
        lambda: dict()
    )
    perturbed_autoeval2lp2metadata = defaultdict(lambda: defaultdict(dict))

    # Check which perturbations are enabled
    do_ext_only = PERTURBATION_EXT_ONLY in enabled_perturbations
    do_rand_remove = PERTURBATION_RAND_REMOVE in enabled_perturbations
    do_samples_with_1 = (
        PERTURBATION_REMOVE_ERRORS_IN_SAMPLES_WITH_1 in enabled_perturbations
    )
    progressive_length = PERTURBATION_PROGRESSIVE_LENGTH in enabled_perturbations

    for (
        autoeval,
        lp2preprocessed_samples,
    ) in autoeval2lp2preprocessed_samples_with_automatic_evaluations.items():

        rng = np.random.default_rng(45)

        for lp, samples in lp2preprocessed_samples.items():
            # normal annotations (always included)
            perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations[
                autoeval
            ][lp] = samples

            if autoeval not in autoevals_with_no_extended_spans:
                # Perturbation number 1. --> do not use extended span (this has been removed to use the new parallel processing more easily, as it was incompatible)
                if False:
                    no_extended_span_samples = [
                        remove_extended_span_from_sample_errors(sample)
                        for sample in samples
                    ]
                    perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations[
                        autoeval + "_NO_EXT"
                    ][
                        lp
                    ] = no_extended_span_samples

                # Perturbation number 2. --> use extended span in the place of normal spans
                if do_ext_only:
                    extended_span_only_samples = [
                        switch_spans_for_extended_spans(sample) for sample in samples
                    ]
                    perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations[
                        autoeval + "_EXT_ONLY"
                    ][
                        lp
                    ] = extended_span_only_samples

            if do_samples_with_1:
                res = [
                    remove_errors_from_samples_with_1_error(sample)
                    for sample in samples
                ]
                samples_with_removed_errors = [r[0] for r in res]
                num_errors_before_removal = sum([r[1] for r in res])
                n_removed = sum(r[2] for r in res)
                perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations[
                    autoeval + f"_REMOVE_ALL_1"
                ][lp] = samples_with_removed_errors

                perturbed_autoeval2lp2metadata[autoeval + f"_REMOVE_ALL_1"][lp] = {
                    "num_errors_before_removal": num_errors_before_removal,
                    "num_errors_removed": n_removed,
                }

            if do_rand_remove:
                total_errors = sum(len(sample.evaluation.errors) for sample in samples)
                sampled_nums = rng.random(total_errors).tolist()

                for pct in range(10, 100, 10):
                    p = pct / 100
                    rand_remove_samples = []
                    i = 0

                    for sample in samples:
                        n_errs = len(sample.evaluation.errors)
                        sample_numbers = sampled_nums[i : i + n_errs]
                        rand_remove_samples.append(
                            remove_spans_with_probability_p(
                                sample,
                                p,
                                sample_numbers,
                            )
                        )
                        i += n_errs

                    perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations[
                        autoeval + f"_RAND_REMOVE_{pct}"
                    ][
                        lp
                    ] = rand_remove_samples

            if progressive_length:
                for n in range(10, 101, 10):

                    progressive_length_samples = [
                        increase_spans_length_by_n(sample, n, lp) for sample in samples
                    ]

                    perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations[
                        autoeval + f"_PROGRESSIVE_LENGTH_{n}"
                    ][
                        lp
                    ] = progressive_length_samples

    return (
        perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations,
        perturbed_autoeval2lp2metadata,
    )
