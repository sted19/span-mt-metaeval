# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging

from mt_evaluation.core import Prompt, AutomaticEvaluation
from mt_evaluation.autoevals.unified.base import Unified


logger = logging.getLogger(__name__)


class UnifiedMQMCritical(Unified):
    prompt = Prompt(
        system_prompt="You are an expert annotator for machine translation quality assessment. Your task is to assess translation quality by comparing it to the source text in the original language. You must identify error spans and assign them an error category, subcategory, and severity level. Return your evaluation in JSON format.",
        user_prompt="""## Task Overview
You will be assessing translations at the segment level, where a segment may contain one or more sentences. Each segment is provided with a corresponding source segment.
Please identify all errors within the translated segment, up to a maximum of five. If there are more than five errors, identify only the five most severe. If it is not possible to reliably identify distinct errors because the translation is too badly garbled or is unrelated to the source, then mark a single Non-translation error that spans the entire segment.
To identify an error, highlight the relevant span of text, and select a category/subcategory and severity level from the available options. (The span of text may be in the source segment if the error is a source error or an omission.) When identifying errors, please be as fine-grained as possible. For example, if a sentence contains two words that are each mistranslated, two separate mistranslation errors should be recorded. If a single stretch of text contains multiple errors, you only need to indicate the one that is most severe. If all have the same severity, choose the first matching category listed in the error typology (eg, Accuracy, then Fluency, then Terminology, etc).
Please pay particular attention to the segment context when annotating. If a translation might be questionable on its own but is acceptable within the segment of the document, it should not be considered erroneous; conversely, if a translation might be acceptable in some context, but not within the current segment, it should be marked as wrong.
There are two special error categories: Source error and Non-translation. Source errors should be annotated separately, highlighting the relevant span in the source segment. They do not count against the five-error limit for target errors, which should be handled in the usual way, whether or not they resulted from a source error. There can be at most one Non-translation error per segment, and it should span the entire segment. No other errors should be identified if Non-translation is selected.

## Error Typology
Use the following error categories and subcategories (enclosed between triple backticks) to classify each error:
```
**Accuracy**
    - Addition: Translation includes information not present in the source.
    - Omission: Translation is missing content from the source.
    - Mistranslation: Translation does not accurately represent the source.
    - Untranslated text: Source text has been left untranslated.
    
**Fluency**
    - Punctuation: Incorrect punctuation (for locale or style).
    - Spelling: Incorrect spelling or capitalization.
    - Grammar: Problems with grammar, other than orthography.
    - Register: Wrong grammatical register (e.g., inappropriately informal pronouns).
    - Inconsistency: Internal inconsistency (not related to terminology).
    - Character encoding: Characters are garbled due to incorrect encoding.

**Terminology**
    - Inappropriate for context: Terminology is non-standard or does not fit context.
    - Inconsistent use: Terminology is used inconsistently.

**Style**
    - Awkward: Translation has stylistic problems.

**Locale convention**
    - Address format: Wrong format for addresses.
    - Currency format: Wrong format for currency.
    - Date format: Wrong format for dates.
    - Name format: Wrong format for names.
    - Telephone format: Wrong format for telephone numbers.
    - Time format: Wrong format for time expressions.

**Other**
    - Other: Any other issue.

**Source error**
    - Source error: An error in the source.

**Non-translation**
    - Non-translation: Impossible to reliably characterize distinct errors.
```

## Error Severities
Assign one of these severity levels (enclosed between triple backticks) to each error:
```
- Critical: Severe translation or grammatical errors
- Major: Actual translation or grammatical errors
- Minor: Smaller imperfections
```

## Input Source and Translated Segments
The source segment and translation to evaluate are provided below, enclosed between triple backticks:
```
{src_lang} source: {src}
{tgt_lang} translation: {tgt}
```

## Output Annotation Format
Return your annotations in JSON format as a list of Python dictionaries enclosed between triple backticks. Each dictionary represents a translation error, and has the following form:
```json
{{
    "span": <span of text containing the error>,
    "explanation": <justification for marking this span as error>,
    "category": <error category>,
    "subcategory": <error subcategory>,
    "severity": <error severity>
}}
```
If no errors are found, return an empty list.
""",
        few_shots=[],
    )
