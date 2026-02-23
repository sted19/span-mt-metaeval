# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Severity-to-score mapping utilities.

This module provides centralized scoring functions to assign numeric scores
to translation errors based on their severity level. This ensures consistent
scoring across all evaluators.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Standard MQM-style scoring
# These are the canonical severity-to-score mappings for the framework
SEVERITY_SCORES = {
    "neutral": 0.0,
    "minor": -1.0,
    "major": -5.0,
    "critical": -10.0,
}

# Special category scores
SPECIAL_CATEGORY_SCORES = {
    "non-translation": -25.0,
    "unintelligible": -25.0,
    "source error": 0.0,
}


def severity_to_score(
    severity: str,
    category: Optional[str] = None,
    default_score: Optional[float] = None,
) -> Optional[float]:
    """
    Convert a severity string to a numeric score.
    
    This function handles various severity formats (e.g., "major", "Major", 
    "major-accuracy") and applies special scoring for certain categories
    like non-translation or source errors.
    
    Args:
        severity: The severity string (e.g., "major", "minor", "critical")
        category: Optional category string for special handling
        default_score: Score to return if severity is unrecognized. 
                      If None, returns None for unrecognized severities.
    
    Returns:
        Numeric score (negative for errors, 0 for neutral/source errors)
        
    Examples:
        >>> severity_to_score("major")
        -5.0
        >>> severity_to_score("minor")
        -1.0
        >>> severity_to_score("critical", category="non-translation")
        -25.0
    """
    if severity is None:
        return default_score
    
    severity_lower = severity.lower().strip()
    category_lower = category.lower().strip() if category else ""
    
    # Check for special categories first
    for special_cat, score in SPECIAL_CATEGORY_SCORES.items():
        if special_cat in category_lower:
            return score
    
    # Check standard severities
    for sev_name, score in SEVERITY_SCORES.items():
        if sev_name in severity_lower:
            return score
    
    logger.warning(f"Unknown severity: {severity}")
    return default_score


def assign_score_based_on_severity(error_severity: str, category: Optional[str] = None) -> float:
    """
    Assign a score based on error severity.
    
    This is a convenience wrapper around severity_to_score() that raises
    an error for unknown severities instead of returning None.
    
    Args:
        error_severity: The severity string (may contain severity name)
        category: Optional category string for special handling
    
    Returns:
        Numeric score for the severity
    
    Raises:
        ValueError: If severity is unknown
    """
    score = severity_to_score(error_severity, category=category, default_score=None)
    
    if score is None:
        raise ValueError(f"Unknown error severity: {error_severity}")
    
    return score
