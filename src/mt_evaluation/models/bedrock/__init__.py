# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

from mt_evaluation.core import Response


@dataclass
class BedrockResponse(Response):
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    stop_reason: str = ""
    cost: float = 0.0
