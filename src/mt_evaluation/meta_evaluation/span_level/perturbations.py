# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from collections import defaultdict
from typing import Dict, List, Set, Optional
import numpy as np

from mt_evaluation.core import Sample, Error
from mt_evaluation.meta_evaluation.metrics_to_evaluate import (
    autoevals_with_no_extended_spans,
)

# Available perturbation types
PERTURBATION_EXT_ONLY = "EXT_ONLY"
PERTURBATION_RAND_REMOVE_05 = "RAND_REMOVE_05"
PERTURBATION_RAND_REMOVE_09 = "RAND_REMOVE_09"
PERTURBATION_REMOVE_ERRORS_IN_SAMPLES_WITH_1 = "REMOVE_ALL_1"
PERTURBATION_PROGRESSIVE_LENGTH = "PROGRESSIVE_LENGTH"

ALL_PERTURBATIONS = {
    PERTURBATION_EXT_ONLY,
    PERTURBATION_RAND_REMOVE_05,
    PERTURBATION_RAND_REMOVE_09,
    PERTURBATION_REMOVE_ERRORS_IN_SAMPLES_WITH_1,
    PERTURBATION_PROGRESSIVE_LENGTH,
}


def increase_spans_length_by_n_characters(sample: Sample, n: int) -> Sample:
    assert sample.evaluation is not None

    new_sample = sample.from_dict(sample.to_dict())

    for error in new_sample.evaluation.errors:
        start, end = error.start, error.end

        if error.is_source_error:
            new_start = max(start - n, 0)
            new_end = min(end + n, len(sample.src))
            new_span = sample.src[new_start:new_end]
        else:
            new_start = max(start - n, 0)
            new_end = min(end + n, len(sample.tgt))
            new_span = sample.tgt[new_start:new_end]

        error.start = new_start
        error.end = new_end
        error.span = new_span

    return new_sample


def remove_errors_from_samples_with_1_error(sample: Sample) -> Sample:
    assert sample.evaluation is not None or sample.human_evaluation is not None

    new_sample = sample.from_dict(sample.to_dict())

    if new_sample.evaluation is not None:
        if len(new_sample.evaluation.errors) <= 1:
            new_sample.evaluation.errors = []
            new_sample.evaluation.score = 0
    else:
        if len(new_sample.human_evaluation.errors) <= 1:
            new_sample.human_evaluation.errors = []
            new_sample.human_evaluation.score = 0

    return new_sample


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
) -> Dict[str, Dict[str, List[Sample]]]:
    """
    Extract perturbations from auto-evaluations.

    Args:
        autoeval2lp2preprocessed_samples_with_automatic_evaluations: Dictionary of auto-evaluations
        enabled_perturbations: Set of perturbation types to generate. If None or empty,
                               no perturbations are generated (only original samples are kept).
                               Valid values: NO_EXT, EXT_ONLY, RAND_REMOVE_05, RAND_REMOVE_09

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

    # Check which perturbations are enabled
    do_ext_only = PERTURBATION_EXT_ONLY in enabled_perturbations
    do_rand_05 = PERTURBATION_RAND_REMOVE_05 in enabled_perturbations
    do_rand_09 = PERTURBATION_RAND_REMOVE_09 in enabled_perturbations
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
                samples_with_removed_errors = [
                    remove_errors_from_samples_with_1_error(sample)
                    for sample in samples
                ]
                perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations[
                    autoeval + "_REMOVE_ALL_1"
                ][lp] = samples_with_removed_errors

            # Only compute random numbers if needed
            if do_rand_05 or do_rand_09:
                total_errors = sum(len(sample.evaluation.errors) for sample in samples)
                sampled_nums = rng.random(total_errors).tolist()

                i, j = 0, 0
                zero_point_five_samples = [] if do_rand_05 else None
                zero_point_one_samples = [] if do_rand_09 else None

                while j < len(samples):
                    sample = samples[j]
                    sample_j_numbers = sampled_nums[
                        i : i + len(sample.evaluation.errors)
                    ]

                    # Perturbation number 3. --> remove spans with probability 0.5
                    if do_rand_05:
                        zero_point_five_samples.append(
                            remove_spans_with_probability_p(
                                sample,
                                0.5,
                                sample_j_numbers,
                            )
                        )

                    # Perturbation number 4. --> remove spans with probability 0.9
                    if do_rand_09:
                        zero_point_one_samples.append(
                            remove_spans_with_probability_p(
                                sample,
                                0.9,
                                sample_j_numbers,
                            )
                        )

                    j += 1
                    i += len(sample.evaluation.errors)

                if do_rand_05:
                    perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations[
                        autoeval + "_RAND_REMOVE_05"
                    ][
                        lp
                    ] = zero_point_five_samples

                if do_rand_09:
                    perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations[
                        autoeval + "_RAND_REMOVE_09"
                    ][
                        lp
                    ] = zero_point_one_samples

            # NOTE: for now we use characters, because on WMT25 we have target lengths Chinese and Korean, which are denser. You should use words in other languages
            if progressive_length:
                for n in range(1, 100, 10):

                    progressive_length_samples = [
                        increase_spans_length_by_n_characters(sample, n)
                        for sample in samples
                    ]

                    perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations[
                        autoeval + f"_PROGRESSIVE_LENGTH_{n}"
                    ][
                        lp
                    ] = progressive_length_samples

    return perturbed_autoeval2lp2preprocessed_samples_with_automatic_evaluations
