# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
String utilities for the MT Evaluation Framework.

This module provides string manipulation functions used throughout the framework.
"""

from typing import List, Tuple


def find_all_literal(text: str, substring: str) -> List[Tuple[int, int]]:
    """
    Find all literal occurrences of a substring in text.
    
    Args:
        text: The text to search in.
        substring: The substring to search for.
        
    Returns:
        List of (start, end) tuples for each occurrence.
        
    Example:
        >>> find_all_literal("hello world hello", "hello")
        [(0, 5), (12, 17)]
    """
    matches = []
    start = 0
    while True:
        start = text.find(substring, start)
        if start == -1:
            break
        matches.append((start, start + len(substring)))
        start += 1
    return matches


__all__ = [
    "find_all_literal",
]
