# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Constants for the MT Evaluation Framework.

This module contains all constants used throughout the framework, including
error type constants, severity levels, and WMT language pair definitions.
"""

# Error type constants
# These represent special error categories that require specific handling
non_translation = "non-translation"
unintelligible = "unintelligible"
omission = "omission"
source_issue = "source_issue"
source_issue2 = "source issue"
source_error = "source_error"
source_error2 = "source error"
creative_reinterpretation = "creative reinterpretation"
no_error = "no error"
no_error2 = "no-error"

# Severity constants
all_severities = ["neutral", "minor", "major", "critical"]
UNKNOWN_SEVERITY = "unknown"

# WMT language pair constants
# NOTE: we are not considering he-en in wmt23 for several reasons:
#   1. it does not contain multiply evaluated samples (to compute IAA)
#   2. the maximum number of samples I can pass with batch inference is 50k,
#      but using he-en with the others we are at 51k
wmt22_lps = ["en-de", "en-zh"]
wmt23_lps = ["en-de", "zh-en"]
wmt24_lps = ["en-de", "en-es", "ja-zh"]
wmt25_lps_mqm = ["en-ko_KR", "ja-zh_CN"]
wmt25_lps_esa = [
    "cs-de_DE",
    "cs-uk_UA",
    "en-ar_EG",
    "en-bho_IN",
    "en-cs_CZ",
    "en-et_EE",
    "en-is_IS",
    "en-it_IT",
    "en-ja_JP",
    "en-mas_KE",
    "en-ru_RU",
    "en-sr_Cyrl_RS",
    "en-uk_UA",
    "en-zh_CN",
]
wmt25_lps = wmt25_lps_mqm + wmt25_lps_esa

# Test set to language pairs mapping
TEST_SET_TO_LPS = {
    "wmt22": wmt22_lps,
    "wmt23": wmt23_lps,
    "wmt24": wmt24_lps,
    "wmt25": wmt25_lps,
}
