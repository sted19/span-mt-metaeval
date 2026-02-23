# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging

from mt_evaluation.autoevals.autoeval import AutoEval
from mt_evaluation.core import AutomaticEvaluation, Error
from mt_evaluation.core.scoring import severity_to_score
from mt_evaluation.autoevals.utils import extract_json_response

logger = logging.getLogger(__name__)


class Unified(AutoEval):

    def parse_response(self, response: str) -> AutomaticEvaluation:
        if response is None:
            logger.warning("Response is None. Returning empty evaluation...")
            return AutomaticEvaluation(
                score=0.0, errors=[], annotation="", parsing_error=True
            )

        try:
            json_response = extract_json_response(response)
        except Exception as e:
            logger.warning(f"Error parsing json: {e}")
            return AutomaticEvaluation(
                score=0.0, errors=[], annotation=response, parsing_error=True
            )

        if type(json_response) is dict:
            json_response = [json_response]

        assert type(json_response) is list

        errors = []
        for error in json_response:
            if not type(error) is dict:
                logger.error(f"Error is not a dictionary: {error}")
                continue

            span = error.get("span")
            category = error.get("category")
            severity = error.get("severity")
            subcategory = error.get("subcategory", "")
            explanation = error.get("explanation", "")
            span_with_context = error.get("span_with_context", None)

            # If the span, category, or severity fields are not there, skip this error --> severity and category are used to determine the score
            # Span is used for span-level meta-evaluation
            if (span is None) or (category is None) or (severity is None):
                logger.warning(
                    f"Error extracting span, category, or severity from json error: {error}"
                )
                continue

            if "*" in category:
                category = category.replace("*", "").strip()

            if "*" in severity:
                severity = severity.replace("*", "").strip()

            score = severity_to_score(severity, category=category)
            if score is None:
                logger.error(f"Error assigning a score to error {error}. Skipping it")
                continue

            errors.append(
                Error(
                    span=span,
                    category=f"{category}-{subcategory}",
                    severity=severity,
                    explanation=explanation,
                    score=score,
                    is_source_error="source error" in category.lower()
                    or "source error" in subcategory.lower()
                    or "omission" in subcategory.lower(),
                    extended_span=span_with_context,
                )
            )

        final_score = sum(error.score for error in errors)

        return AutomaticEvaluation(
            score=final_score, errors=errors, annotation=response, parsing_error=False
        )
