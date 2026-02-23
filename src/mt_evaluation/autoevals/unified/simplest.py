# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging

from mt_evaluation.core import Prompt, AutomaticEvaluation, Error
from mt_evaluation.autoevals.unified.base import Unified
from mt_evaluation.autoevals.utils import extract_json_response


logger = logging.getLogger(__name__)


class Simplest(Unified):
    prompt = Prompt(
        system_prompt="You are an expert translator specialized in detecting translation errors.",
        user_prompt="""Please analyze this translation for errors by comparing it to its source text:

{src_lang} source: {src}
{tgt_lang} translation: {tgt}

Translation errors occur when the translation fails to convey the meaning of the source text or violates grammatical rules of the target language. 

Identify the EXACT words or phrases containing errors, without including surrounding context. If multiple non-adjacent words have errors, create separate error entries.

Return your analysis as a JSON array enclosed in triple backticks. Each element of the array represents one error and must include the following fields:

```json
[
  {{
    "span": "<verbatim span of text containing the error>",
    "explanation": "<justification for marking this span as an error>",
    "severity": "<one of Critical, Major, or Minor>"
  }}
]
```

If no errors are found, return:

```json
[]
```
""",
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

            if "major" in severity.lower():
                score = -5.0
            elif "minor" in severity.lower():
                score = -1.0
            elif "neutral" in severity.lower():
                score = 0.0
            elif "critical" in severity.lower():
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
