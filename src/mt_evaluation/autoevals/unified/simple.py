# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging

from mt_evaluation.core import Prompt, AutomaticEvaluation, Error
from mt_evaluation.autoevals.unified.base import Unified
from mt_evaluation.autoevals.utils import extract_json_response


logger = logging.getLogger(__name__)


class Simple(Unified):
    prompt = Prompt(
        system_prompt="You are an expert annotator for machine translation quality assessment. Compare the translation to the source text and identify all error spans in the translation. Return your evaluation in JSON format.",
        user_prompt="""# Task Overview

You will be provided with a source paragraph and its translation. A paragraph may contain one or more sentences. Your task is to identify all translation errors, assigning a severity level to each error.

## Task Guidelines

- To identify an error, you must mark its span of text in the translation. Do not mark spans from the source text under any circumstance.
- Report spans verbatim so they can be located via exact string matching later. Do not modify spans in any way (no added or missing characters, spaces, quotes, or ellipses).
- Be as fine-grained as possible. For example, if two consecutive words are each mistranslated, record two separate errors. However, if multiple errors occur within a single inseparable stretch of text, record only the most severe error (see severity definitions below).

## Severity Levels

Choose one severity level for each error:

- Critical: Errors that severely distort the meaning of the source text or make the translation very difficult to understand or parse.
- Major: Errors that alter the meaning of the source or impact the readability or flow of the translation.
- Minor: Small imperfections with minimal impact on meaning preservation or readability.

## Output Annotation Format

Return your annotations as strict JSON: an array of objects enclosed in triple backticks. Each object represents one error and must have the following fields:

```json
[
  {{
    "span": "<verbatim span of text containing the error>",
    "explanation": "<justification for marking this span as an error>",
    "severity": "<Critical|Major|Minor>"
  }}
]
```

If no errors are found, return:

```json
[]
```

## Input Source and Translation

The source paragraph and translation to evaluate are provided below:

```
{src_lang} source: {src}
{tgt_lang} translation: {tgt}
```""",
        few_shots=[],
    )

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
            severity = error.get("severity")
            explanation = error.get("explanation", "")

            # If the span or severity fields are not there, skip this error --> severity is used to determine the score while Span is used for span-level meta-evaluation
            if (span is None) or (severity is None):
                logger.warning(
                    f"Error extracting span, category, or severity from json error: {error}"
                )
                continue

            severity = severity.lower()

            if "major" in severity:
                score = -5.0
            elif "minor" in severity:
                score = -1.0
            elif "neutral" in severity:
                score = 0.0
            elif "critical" in severity:
                score = -10.0
            else:
                logger.warning(f"Error assigning a score to error {error}. Skipping it")
                continue

            errors.append(
                Error(
                    span=span,
                    category=f"N/A",
                    severity=severity,
                    explanation=explanation,
                    score=score,
                    is_source_error=False,
                )
            )

        final_score = sum(error.score for error in errors)

        return AutomaticEvaluation(
            score=final_score, errors=errors, annotation=response, parsing_error=False
        )
