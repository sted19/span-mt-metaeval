# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Language code mappings for the MT Evaluation Framework.

This module provides comprehensive mappings from language codes to language names,
supporting both base language codes (e.g., "en") and regional specifications
(e.g., "en_US", "pt_BR").
"""

# Comprehensive mapping from language codes to language names
LANG_CODE_TO_NAME = {
    # Base language codes
    "en": "English",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zu": "Zulu",
    # Regional specifications (only when they distinguish from standard)
    "ar_EG": "Egyptian Arabic",
    "ar_SA": "Saudi Arabic",
    "es_MX": "Mexican Spanish",
    "fr_CA": "Canadian French",
    "pt_BR": "Brazilian Portuguese",
    "sw_KE": "Kenyan Swahili",
    "sw_TZ": "Tanzanian Swahili",
    "zh_TW": "Taiwanese Chinese",
    # Standard regional forms (mapping to base language)
    "bg_BG": "Bulgarian",
    "bn_IN": "Bengali",
    "ca_ES": "Catalan",
    "cs_CZ": "Czech",
    "da_DK": "Danish",
    "de_DE": "German",
    "el_GR": "Greek",
    "et_EE": "Estonian",
    "fa_IR": "Persian",
    "fi_FI": "Finnish",
    "fil_PH": "Filipino",
    "fr_FR": "French",
    "gu_IN": "Gujarati",
    "he_IL": "Hebrew",
    "hi_IN": "Hindi",
    "hr_HR": "Croatian",
    "hu_HU": "Hungarian",
    "id_ID": "Indonesian",
    "is_IS": "Icelandic",
    "it_IT": "Italian",
    "ja_JP": "Japanese",
    "kn_IN": "Kannada",
    "ko_KR": "Korean",
    "lt_LT": "Lithuanian",
    "lv_LV": "Latvian",
    "ml_IN": "Malayalam",
    "mr_IN": "Marathi",
    "nl_NL": "Dutch",
    "no_NO": "Norwegian",
    "pa_IN": "Punjabi",
    "pl_PL": "Polish",
    "pt_PT": "Portuguese",
    "ro_RO": "Romanian",
    "ru_RU": "Russian",
    "sk_SK": "Slovak",
    "sl_SI": "Slovenian",
    "sr_RS": "Serbian",
    "sv_SE": "Swedish",
    "ta_IN": "Tamil",
    "te_IN": "Telugu",
    "th_TH": "Thai",
    "tr_TR": "Turkish",
    "uk_UA": "Ukrainian",
    "ur_PK": "Urdu",
    "vi_VN": "Vietnamese",
    "zu_ZA": "Zulu",
    "zh_CN": "Chinese",
    "bho_IN": "Bhojpuri",
    "mas_KE": "Maasai",
    "sr_Cyrl_RS": "Serbian Cyrillic",
}

# Backward compatibility alias
lang_code2lang = LANG_CODE_TO_NAME


def get_language_name(lang_code: str) -> str:
    """
    Get the language name for a given language code.

    Args:
        lang_code: The language code (e.g., "en", "de_DE", "zh_CN").

    Returns:
        str: The language name.

    Raises:
        ValueError: If the language code is not found.
    """
    if lang_code in LANG_CODE_TO_NAME:
        return LANG_CODE_TO_NAME[lang_code]
    raise ValueError(f"Unknown language code: {lang_code}")


def get_language_name_safe(lang_code: str, default: str = "Unknown") -> str:
    """
    Get the language name for a given language code, with a default fallback.

    Args:
        lang_code: The language code (e.g., "en", "de_DE", "zh_CN").
        default: The default value to return if the code is not found.

    Returns:
        str: The language name or the default value.
    """
    return LANG_CODE_TO_NAME.get(lang_code, default)
