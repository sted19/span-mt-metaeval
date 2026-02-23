# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import re
import json
from json_repair import repair_json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def extract_json_response(text: str) -> List[Dict]:
    """
    Extract the last piece of text enclosed between triple backticks (```).

    Args:
        text (str): The input text to search in

    Returns:
        str: The content of the last code block (without the backticks)

    Raises:
        ValueError: If no text is found between triple backticks
    """
    # Pattern to match text between triple backticks
    # Using DOTALL flag to match newlines within the code blocks
    pattern = r"```json\n(.*?)\n```"

    # Find all matches
    matches = re.findall(pattern, text, re.DOTALL)

    # If no matches found, simplify the regular expression, then raise an exception
    if not matches:
        pattern = r"```(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

    # If not matches are found even with the simpler regular expression, make sure the model did not output the final answer using latex formatting (Llama does it)
    if not matches:
        pattern = r"\$\\boxed\{(.*?)\}\$"
        matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        raise ValueError("No text found between triple backticks")

    json_str = matches[-1].strip()

    # Try parsing as-is first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.debug(
            f"First tentative of JSON parsing failed with error {e}. Repairing the JSON string."
        )
        try:
            repaired = repair_json(json_str)
            return json.loads(repaired)
        except Exception as e:
            raise Exception(f"JSON parsing failed even after repair with error {e}\n")
