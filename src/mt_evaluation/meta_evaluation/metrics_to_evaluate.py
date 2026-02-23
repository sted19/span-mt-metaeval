# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from pathlib import Path

autoevals_with_no_extended_spans = [
    "unified-mqm-boosted-v2-claude-3-5-haiku",
    "GemSpanEval.pri",
    "GemSpanEval-QE.sec",
    "AIP.pri",
    "AIP.sec",
    "AutoLQA.pri",
    "AutoLQA41.sec",
    "AutoLQAESA.sec",
    "XCOMET-XL.bas",
    "XCOMET-XXL.bas",
    "mqm.super.1",
    "mqm.super.2",
    "mqm.super.3",
]

metrics_to_evaluate_info_wmt25 = [
    # {
    #     "autoeval": "unified-mqm-boosted-doc-context",
    #     "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "wmt25",
    # },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt25",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt25",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "qwen.qwen3-235b-a22b-2507-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt25",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "openai.gpt-oss-120b-1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt25",
    },
]

metrics_to_evaluate_info_wmt24 = [
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt24",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt24",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "qwen.qwen3-235b-a22b-2507-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt24",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "openai.gpt-oss-120b-1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt24",
    },
]


metrics_to_evaluate_info_wmt23 = [
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt23",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt23",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "qwen.qwen3-235b-a22b-2507-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt23",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "openai.gpt-oss-120b-1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt23",
    },
]

metrics_to_evaluate_info_wmt22 = [
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt22",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt22",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "qwen.qwen3-235b-a22b-2507-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt22",
    },
    {
        "autoeval": "unified-mqm-boosted-v5",
        "model": "openai.gpt-oss-120b-1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "wmt22",
    },
]

metrics_to_merge_info = [
    # {
    #     "merged_name": "merged-gemba-mqm-claude-3.5--2-runs",
    #     "metrics_to_merge": [
    #         {
    #             "autoeval": "gemba-mqm",
    #             "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "repr1",
    #         },
    #         {
    #             "autoeval": "gemba-mqm",
    #             "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #     ],
    # },
    # {
    #     "merged_name": "merged-gemba-mqm-llama3-70b--2-runs",
    #     "metrics_to_merge": [
    #         {
    #             "autoeval": "gemba-mqm",
    #             "model": "us.meta.llama3-3-70b-instruct-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "repr1",
    #         },
    #         {
    #             "autoeval": "gemba-mqm",
    #             "model": "us.meta.llama3-3-70b-instruct-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #     ],
    # },
    # {
    #     "merged_name": "merged-gemba-mqm-claude-3.5-llama3-70b-default",
    #     "metrics_to_merge": [
    #         {
    #             "autoeval": "gemba-mqm",
    #             "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #         {
    #             "autoeval": "gemba-mqm",
    #             "model": "us.meta.llama3-3-70b-instruct-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #     ],
    # },
    # {
    #     "merged_name": "merged-gemba-mqm-claude-3.5-llama3-70b-repr1",
    #     "metrics_to_merge": [
    #         {
    #             "autoeval": "gemba-mqm",
    #             "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #         {
    #             "autoeval": "gemba-mqm",
    #             "model": "us.meta.llama3-3-70b-instruct-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #     ],
    # },
    # {
    #     "merged_name": "merged-accuracy-fluency-union-claude-3.5",
    #     "merging_strategy": "union",
    #     "metrics_to_merge": [
    #         {
    #             "autoeval": "specialized-accuracy",
    #             "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #         {
    #             "autoeval": "specialized-fluency",
    #             "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #     ],
    # },
    # {
    #     "merged_name": "merged-accuracy-fluency-union-with-overlap-claude-3.5",
    #     "merging_strategy": "union-with-overlap",
    #     "metrics_to_merge": [
    #         {
    #             "autoeval": "specialized-accuracy",
    #             "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #         {
    #             "autoeval": "specialized-fluency",
    #             "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #     ],
    # },
    # {
    #     "merged_name": "merged-accuracy-fluency-no-src-claude-3.5",
    #     "metrics_to_merge": [
    #         {
    #             "autoeval": "specialized-accuracy-v2",
    #             "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #         {
    #             "autoeval": "specialized-fluency-no-src",
    #             "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #             "outputs_path": Path("outputs"),
    #             "reference_free": True,
    #             "run_specific_info": "default",
    #         },
    #     ],
    # },
]

metrics_to_evaluate_info_wmt24_old = [
    # {
    #     "autoeval": "gemba-mqm",
    #     "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "specialized-register-simple-strict",
    #     "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "specialized-register-simple",
    #     "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "specialized-register-simple-strict",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "specialized-register-simple",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "specialized-accuracy-v2",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "unified-mqm-critical",
    #     "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "register",
    # },
    # {
    #     "autoeval": "unified-simple",
    #     "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # ============================================
    # The ones above are not on the entire data
    # ============================================
    # {
    #     "autoeval": "specialized-addition-eric",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "specialized-addition",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "specialized-omission-addition",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    {
        "autoeval": "specialized-fluency",
        "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "default",
    },
    # {
    #     "autoeval": "specialized-fluency-no-src",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "unified-simple",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    {
        "autoeval": "specialized-accuracy",
        "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "default",
    },
    # {
    #     "autoeval": "unified-mqm-boosted-v3",
    #     "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "budget-4096",
    # },
    # {
    #     "autoeval": "unified-mqm-critical",
    #     "model": "qwen.qwen3-235b-a22b-2507-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "no-reasoning",
    # },
    # {
    #     "autoeval": "unified-mqm-boosted-v2",
    #     "model": "qwen.qwen3-235b-a22b-2507-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "no-reasoining",
    # },
    # {
    #     "autoeval": "unified-mqm-critical",
    #     "model": "openai.gpt-oss-120b-1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "reasoning-medium",
    # },
    # {
    #     "autoeval": "unified-mqm-boosted-v2",
    #     "model": "openai.gpt-oss-120b-1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "reasoning-medium",
    # },
    # {
    #     "autoeval": "unified-simplest",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    {
        "autoeval": "unified-mqm-boosted-v2",
        "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "outputs_path": Path("outputs"),
        "reference_free": True,
        "run_specific_info": "default",
    },
    # {
    #     "autoeval": "unified-mqm-boosted-v2",
    #     "model": "us.meta.llama3-3-70b-instruct-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "unified-mqm-boosted",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "unified-mqm-boosted",
    #     "model": "us.meta.llama3-3-70b-instruct-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "unified-mqm-critical",
    #     "model": "us.meta.llama3-3-70b-instruct-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "unified-mqm",
    #     "model": "us.meta.llama3-3-70b-instruct-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "unified-mqm-critical",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "unified-mqm",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "gemba-mqm",
    #     "model": "us.meta.llama3-3-70b-instruct-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "repr1",
    # },
    # {
    #     "autoeval": "gemba-mqm",
    #     "model": "us.meta.llama3-3-70b-instruct-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "gemba-mqm",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "repr1",
    # },
    # {
    #     "autoeval": "gemba-mqm",
    #     "model": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "gemba-mqm",
    #     "model": "google/gemma-3-12b-it",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
    # {
    #     "autoeval": "gemba-mqm",
    #     "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    #     "outputs_path": Path("outputs"),
    #     "reference_free": True,
    #     "run_specific_info": "default",
    # },
]
