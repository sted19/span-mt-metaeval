# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
from typing import Optional

from mt_evaluation.core import Prompt
from mt_evaluation.autoevals.unified.base import Unified

logger = logging.getLogger(__name__)


class UnifiedMQMBoostedV5(Unified):
    """
    Unified MQM evaluator with boosted prompting (version 5).
    
    This evaluator uses a detailed step-by-step methodology for error identification,
    analysis, and annotation generation.
    """

    def get_final_instruction(self, is_reasoning: bool = False) -> Optional[str]:
        """
        Get the final instruction based on reasoning mode.
        
        Args:
            is_reasoning: Whether the model is using reasoning/thinking mode.
            
        Returns:
            The appropriate final instruction string.
        """
        if is_reasoning:
            return (
                "Execute these steps sequentially. Ensure you complete all steps: "
                "error identification, individual error analysis (including reasoning, "
                "categorization, and verification), and final annotation generation. "
                "Your output outside thinking tags must contain only the final json annotation."
            )
        else:
            return (
                "Execute these steps sequentially. Ensure your output shows your reasoning "
                "for all steps: error identification, individual error analysis (including "
                "reasoning, categorization, and verification), and final annotation generation."
            )

    prompt = Prompt(
        system_prompt="You are an expert annotator for machine translation quality assessment. Your task is to detect translation errors by comparing a translation to its source text. Identify precise error spans and assign each an error category, subcategory, and severity level according to the provided error typology and severity levels. Follow the provided task instructions exactly as specified in the step-by-step methodology. Return your assessment in JSON format.",
        user_prompt="""## Task Overview
        
You will be provided with a source paragraph and its translation. A paragraph may contain one or more sentences. Your task is to identify all translation errors, assigning a category, subcategory, and severity level to each error. 

### Task Guidelines

- To identify an error, you must mark its span of text in the translation. Only in two special cases must the error be located in the source paragraph rather than the translation. These two special cases depend on the error category you assign to the identified error (refer to error categories and subcategories below):
    1. **Omission errors** (category='Accuracy' and subcategory='Omission'): Mark the missing span of text in the source paragraph.  
    2. **Source errors** (category='Source error' and subcategory='Source error'): Mark the problematic span of text in the source paragraph. Source errors are problems in the source paragraph itself, not translation errors (e.g., grammatical errors in the source paragraph). When source errors occur, do not penalize the translation by marking a corresponding translation error unless the translation introduced additional problems. 
  Apart from these two special cases, all errors must be located in the translation.    

- When identifying errors, be as fine-grained as possible. For example, if two consecutive words are each mistranslated, record two separate errors. However, if multiple errors occur in a single stretch of text and cannot be separated, record only the most severe error (refer to the available error severities below).  

- We will later derive the position of the identified error spans in the source or translation paragraphs via string matching. Therefore, report the identified error spans verbatim, without modifying or altering them in any way. 

- If it is not possible to reliably identify distinct errors because the translation is too badly garbled or is unrelated to the source, mark a single 'Unintelligible' error that spans the entire paragraph. There can be at most one 'Unintelligible' error per translation, and it should span the entire paragraph. Do not identify other errors if the 'Unintelligible' category is used.

### Error typology

You must select error categories and subcategories from the following error typology:
```
**Accuracy**
    - Addition: Translation includes information not present in the source.
    - Omission: Translation is missing content from the source.
    - Mistranslation: Translation does not accurately represent the source.
    - Untranslated text: Source text has been left untranslated when it should have been translated (note: use common sense and consider target language conventions, as some text like certain titles or certain proper names are typically left untranslated).  
    
**Fluency**
    - Punctuation: Incorrect punctuation (for locale or style).
    - Spelling: Incorrect spelling or capitalization.
    - Grammar: Problems with grammar, other than orthography.
    - Register: Wrong grammatical register (e.g., inappropriately informal pronouns).
    - Inconsistency: Internal inconsistency (not related to terminology).
    - Character encoding: Characters are garbled due to incorrect encoding.

**Terminology**
    - Inappropriate for context: Terminology is non-standard or does not fit the context.
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

**Unintelligible**
    - Unintelligible: Impossible to reliably characterize distinct errors.
```
Each error category (e.g., Accuracy or Fluency) has one or more subcategories (such as Addition for Accuracy, and Punctuation for Fluency).

### Severity levels

You must select severity levels from the following list:
```
- **Critical**: Errors that severely distort the meaning of the source text or make the translation very difficult to understand or parse.
- **Major**: Errors that alter the meaning of the source or impact the readability or flow of the translation.
- **Minor**: Small imperfections that have minimal impact on meaning preservation or readability.
```

### Output Annotation Format
Return your annotations in JSON format as a list of Python dictionaries enclosed between triple backticks. Each dictionary represents a translation error and has the following form:
```json
{{
    "span": <minimal span of text containing the error>,
    "span_with_context": <extended span of text containing the error>,
    "explanation": <justification for marking this span as error>,
    "category": <error category>,
    "subcategory": <error subcategory>,
    "severity": <error severity>
}}
```
If no errors are found, return an empty list.

## Input Source and Translated Segments
The source paragraph and translation to evaluate are provided below:
```
{src_lang} source: {src}
{tgt_lang} translation: {tgt}
```

## Task instructions

You must execute these steps in order:
1. **ERROR IDENTIFICATION**: Analyze the translation sentence by sentence. For each sentence, quote it, then identify potential translation errors by specifying error spans within the considered sentence. List all the spans of text that are potential translation errors.
2. **ERROR ANALYSIS**: Examine each identified span in isolation:
    1. **REASONING**: Explain why this span of text should be considered an error. If during your reasoning you determine that this is not actually an error, discard it and move to the next span. Otherwise, proceed with the next step.
    2. **CATEGORIZATION**: Assign an appropriate category, a subcategory, and a severity level to the identified error. Pay particular attention to severity assignment. Ensure that the assigned severity label reflects the severity description. Adjust the severity level if your initial assessment doesn't align with the definitions. 
    3. **VERIFICATION**: Review the error you have identified, its category, subcategory, and severity level. Confirm compliance with the annotation guidelines by checking:
        - Was the error span correctly marked in the translation? Or is it an omission or source error and should be marked in the source?
        - Was the error span correctly copied verbatim from the translation or source paragraphs, or have other characters been added? 
3. **FINAL ANNOTATION GENERATION**: Generate the output annotation in JSON format as requested.

{final_instruction}""",
        few_shots=[],
    )
