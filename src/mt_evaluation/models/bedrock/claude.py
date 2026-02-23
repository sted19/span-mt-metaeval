# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List, Dict, Tuple
import logging
import json
import asyncio
import hashlib

from mt_evaluation.core import FewShots
from mt_evaluation.models.bedrock import BedrockResponse
from mt_evaluation.models.bedrock.base import BedrockModel

logger = logging.getLogger(__name__)


class Claude(BedrockModel):

    def batch_inference(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        few_shots: List[FewShots],
        job_name: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> List[BedrockResponse]:

        messages, system_messages = self._prepare_messages(
            system_prompts, user_prompts, few_shots
        )

        def hash_message(msgs: List[Dict], sys_msg: str):
            """Create a hash of the message content for unique identification"""
            content = {"messages": msgs, "system": sys_msg}
            content_str = json.dumps(content, sort_keys=True)
            return hashlib.md5(content_str.encode()).hexdigest()

        # Keep track of original indices based on the input messages
        message_hash_to_index: Dict[str, int] = {}
        json_records = []

        max_tokens = self._estimate_max_tokens(user_prompts, few_shots, max_new_tokens)

        for idx, (message, system_message) in enumerate(zip(messages, system_messages)):
            message_hash = hash_message(message, system_message)
            message_hash_to_index[message_hash] = idx

            model_input = self._prepare_request_body(
                messages=message,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            json_records.append({"recordId": f"CALL{idx}", "modelInput": model_input})

        job_arn = self.create_batch_job(
            json_records,
            job_name,
            overwrite_existing=kwargs.get("overwrite_existing_job", False),
        )

        # Wait for status to be either InProgress or Completed
        status = self.wait_for_batch_job_start(job_arn)

        if status == "InProgress":
            print(
                "Batch Inference in progress. Retrieve results when it completes. Exiting."
            )
            exit()

        assert status == "Completed", f"Batch job status: {status}"

        response = self.download_results_from_s3(job_name, job_arn)

        """
        Make sure the ordering is the same.
        Sorting by recordId is not enough: the same recordId might be assigned to different prompts depending 
        on what data I called the evaluation on. So I cannot just sort by recordId. Instead, I need to use the full
        messages and system prompts to make sure that I match answers to their prompts
        """

        logger.info("Recovering the original order of prompts...")

        raw_responses = [
            json.loads(line)
            for line in response["Body"].read().decode("utf-8").strip().split("\n")
        ]

        # Initialize result array with None values (allows partial results)
        responses: List[BedrockResponse | None] = [None] * len(messages)
        extra_responses = []
        for raw_response in raw_responses:
            message_hash = hash_message(
                raw_response["modelInput"]["messages"],
                raw_response["modelInput"]["system"],
            )

            bedrock_response = self.parse_response_into_a_bedrock_response(
                raw_response["modelOutput"]
            )

            if message_hash in message_hash_to_index:
                original_index = message_hash_to_index[message_hash]
                responses[original_index] = bedrock_response
            else:
                # Handle extra responses that don't match any original prompt
                extra_responses.append(bedrock_response)

        if len(extra_responses) != 0:
            logger.warning(
                "The retrieved results from the S3 bucket contain answers to prompts "
                "not in the list of messages! These answers are discarded!"
            )

        # If any response is missing, count them and ask for approval to potentially collect them via API calls.
        if any(response is None for response in responses):

            missing_indices = [
                idx for idx, response in enumerate(responses) if response is None
            ]
            missing_user_prompts = [user_prompts[idx] for idx in missing_indices]
            missing_system_prompts = [system_prompts[idx] for idx in missing_indices]
            missing_few_shots = [few_shots[idx] for idx in missing_indices]

            missing_responses = self.compute_responses_missing_from_batch_inference(
                missing_user_prompts,
                missing_system_prompts,
                missing_few_shots,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

            for idx, original_index in enumerate(missing_indices):
                responses[original_index] = missing_responses[idx]

        assert all(response is not None for response in responses)

        return responses

    @classmethod
    def _estimate_max_tokens(
        cls, user_prompts: List[str], few_shots: List[FewShots], max_new_tokens: int
    ) -> int:
        # To count total max_tokens using max_new_tokens I'm using the length of the prompt and the length of the provided few shots
        # Max tokens is the sum of all the prompt (including few shots prompts and assistant responses) plus max_new_tokens
        # however, prompt length can change according to the provided prompts (depending on src length). Therefore,
        # I compute the maximum over user prompts
        max_tokens = (
            max(
                (
                    len(user_prompt)
                    + sum(
                        len(shot_user_prompt) + len(shot_assistant_response)
                        for shot_user_prompt, shot_assistant_response in zip(
                            few_shots[0].user_prompts,
                            few_shots[0].assistant_responses,
                        )
                    )
                    if few_shots
                    else len(user_prompts)
                )
                for user_prompt in user_prompts
            )
            + max_new_tokens
        )

        return max_tokens

    @classmethod
    def _prepare_messages(
        cls,
        system_prompts: List[str],
        user_prompts: List[str],
        few_shots: List[FewShots],
    ) -> Tuple[List[List[Dict]], List[str]]:
        if len(user_prompts) != len(few_shots) != len(system_prompts):
            raise ValueError(
                f"Number of system prompts, user prompts, and few-shots must match! ({len(system_prompts)}, {len(user_prompts)}, {len(few_shots)})"
            )

        system_messages, messages = [], []
        for system_prompt, user_prompt, sample_few_shots in zip(
            system_prompts, user_prompts, few_shots
        ):
            sample_message = []

            for shot_user_prompt, shot_response in zip(
                sample_few_shots.user_prompts, sample_few_shots.assistant_responses
            ):
                sample_message += [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": shot_user_prompt}],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": shot_response}],
                    },
                ]

            sample_message += [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}],
                },
            ]

            system_messages.append(system_prompt)
            messages.append(sample_message)

        return messages, system_messages

    @classmethod
    def _prepare_request_body(
        cls,
        messages: List[Dict],
        system_message: str,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        **kwargs,
    ) -> Dict:

        req_body: Dict[str, str | int | List | float] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "system": system_message,
            "max_tokens": max_tokens,
        }

        if temperature is not None:
            req_body["temperature"] = temperature

        if top_p is not None:
            req_body["top_p"] = top_p

        return req_body

    async def single_call(
        self,
        messages: List[Dict],
        system_message: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> Dict:

        invoke_model_inputs = {
            "body": json.dumps(
                self._prepare_request_body(
                    messages,
                    system_message,
                    max_tokens,
                    temperature,
                    top_p,
                    **kwargs,
                )
            ),
            "modelId": self.name,
            "accept": "application/json",
            "contentType": "application/json",
        }

        return await self.invoke_model_and_read_response(invoke_model_inputs)

    async def concurrent_call(
        self,
        messages: List[List[Dict]],
        system_messages: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> List[Dict]:

        tasks = []
        for sample_msg, sample_sys_msg in zip(messages, system_messages):
            tasks.append(
                self.single_call(
                    sample_msg,
                    sample_sys_msg,
                    max_tokens,
                    temperature,
                    top_p,
                    **kwargs,
                )
            )

        responses = await asyncio.gather(*tasks)

        return responses

    async def _async_call(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        few_shots: List[FewShots],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> List[BedrockResponse]:
        """
        Generate the answer to the given prompts

        Args:
            system_prompts (List[str]): List of system_prompts
            user_prompts (List[str]): List of user prompts, to be associated to system_prompts
            few_shots (List[Dict]): List of examples paired with the desired model answers to use for in-context learning
            batch_size (int): Number of prompts to process in one batch
            max_new_tokens (int): Maximum number of tokens to be generated by the model

        Returns:
            List[BedrockMessage]: List of responses, one per prompt
        """
        messages, system_messages = self._prepare_messages(
            system_prompts, user_prompts, few_shots
        )

        max_tokens = self._estimate_max_tokens(user_prompts, few_shots, max_new_tokens)

        if len(messages) == len(system_messages) == 1:
            messages = messages[0]
            system_messages = system_messages[0]

            responses = [
                await self.single_call(
                    messages,
                    system_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    **kwargs,
                )
            ]
        else:
            responses = await self.concurrent_call(
                messages,
                system_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        responses = [
            self.parse_response_into_a_bedrock_response(response)
            for response in responses
        ]

        return responses

    def parse_response_into_a_bedrock_response(self, response: Dict) -> BedrockResponse:
        return BedrockResponse(
            num_input_tokens=response["usage"]["input_tokens"],
            num_output_tokens=response["usage"]["output_tokens"],
            response=response["content"][0]["text"],
            stop_reason=response["stop_reason"],
            cost=self.compute_cost(
                response["usage"]["input_tokens"],
                response["usage"]["output_tokens"],
            ),
        )


class Claude35Haiku(Claude):
    @property
    def price_per_1000_input_tokens(self):
        return 0.0008 * 0.3

    @property
    def price_per_1000_output_tokens(self):
        return 0.004 * 0.3


class Claude35Sonnet(Claude):
    price_per_1000_input_tokens = 0.003
    price_per_1000_output_tokens = 0.015


class ClaudeReasoning(Claude):

    def batch_inference(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        few_shots: List[FewShots],
        job_name: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs,
    ) -> List[BedrockResponse]:

        messages, system_messages = self._prepare_messages(
            system_prompts, user_prompts, few_shots
        )

        reasoning_effort = kwargs.get("reasoning_effort", None)
        reasoning_budget = kwargs.get("reasoning_budget", 0)

        def hash_message(msgs: List[Dict], sys_msg: str):
            """Create a hash of the message content for unique identification"""
            content = {"messages": msgs, "system": sys_msg}
            content_str = json.dumps(content, sort_keys=True)
            return hashlib.md5(content_str.encode()).hexdigest()

        # Keep track of original indices based on the input messages
        message_hash_to_index: Dict[str, int] = {}
        json_records = []

        max_tokens = self._estimate_max_tokens(user_prompts, few_shots, max_new_tokens)

        for idx, (message, system_message) in enumerate(zip(messages, system_messages)):
            message_hash = hash_message(message, system_message)
            message_hash_to_index[message_hash] = idx

            model_input = self._prepare_request_body(
                messages=message,
                system_message=system_message,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                reasoning_effort=reasoning_effort,
                reasoning_budget=reasoning_budget,
            )

            json_records.append({"recordId": f"CALL{idx}", "modelInput": model_input})

        job_arn = self.create_batch_job(
            json_records,
            job_name,
            overwrite_existing=kwargs.get("overwrite_existing_job", False),
        )

        # Wait for status to be either InProgress or Completed
        status = self.wait_for_batch_job_start(job_arn)

        if status == "InProgress":
            print(
                "Batch Inference in progress. Retrieve results when it completes. Exiting."
            )
            exit()

        assert status == "Completed", f"Batch job status: {status}"

        response = self.download_results_from_s3(job_name, job_arn)

        """
        Make sure the ordering is the same.
        Sorting by recordId is not enough: the same recordId might be assigned to different prompts depending 
        on what data I called the evaluation on. So I cannot just sort by recordId. Instead, I need to use the full
        messages and system prompts to make sure that I match answers to their prompts
        """

        logger.info("Recovering the original order of prompts...")

        raw_responses = [
            json.loads(line)
            for line in response["Body"].read().decode("utf-8").strip().split("\n")
        ]

        # Initialize result array with None values (allows partial results)
        responses: List[BedrockResponse | None] = [None] * len(messages)
        extra_responses = []
        for raw_response in raw_responses:
            message_hash = hash_message(
                raw_response["modelInput"]["messages"],
                raw_response["modelInput"]["system"],
            )

            try:
                model_output = raw_response["modelOutput"]
                bedrock_response = self.parse_response_into_a_bedrock_response(
                    model_output
                )
            except KeyError as e:
                logger.error(f"Failed to parse response!")
                bedrock_response = BedrockResponse(
                    num_input_tokens=0,
                    num_output_tokens=0,
                    response="",
                    stop_reason="",
                    cost=0,
                )

            if message_hash in message_hash_to_index:
                original_index = message_hash_to_index[message_hash]
                responses[original_index] = bedrock_response
            else:
                # Handle extra responses that don't match any original prompt
                extra_responses.append(bedrock_response)

        if len(extra_responses) != 0:
            logger.warning(
                "The retrieved results from the S3 bucket contain answers to prompts "
                "not in the list of messages! These answers are discarded!"
            )

        # If any response is missing, count them and ask for approval to potentially collect them via API calls.
        if any(response is None for response in responses):

            missing_indices = [
                idx for idx, response in enumerate(responses) if response is None
            ]
            missing_user_prompts = [user_prompts[idx] for idx in missing_indices]
            missing_system_prompts = [system_prompts[idx] for idx in missing_indices]
            missing_few_shots = [few_shots[idx] for idx in missing_indices]

            missing_responses = self.compute_responses_missing_from_batch_inference(
                missing_user_prompts,
                missing_system_prompts,
                missing_few_shots,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

            for idx, original_index in enumerate(missing_indices):
                responses[original_index] = missing_responses[idx]

        assert all(response is not None for response in responses)

        return responses

    @classmethod
    def _prepare_request_body(
        cls,
        messages: List[Dict],
        system_message: str,
        max_tokens: int,
        temperature: float | None,
        top_p: float | None,
        **kwargs,
    ) -> Dict:

        reasoning_effort = kwargs.get("reasoning_effort", None)
        reasoning_budget = kwargs.get("reasoning_budget", 0)

        req_body: Dict[str, str | int | List | float] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "system": system_message,
            "max_tokens": max_tokens,
            "thinking": {
                "type": "enabled" if reasoning_effort is not None else "disabled",
                "budget_tokens": reasoning_budget,
            },
        }

        if temperature is not None:
            req_body["temperature"] = temperature

        if top_p is not None:
            req_body["top_p"] = top_p

        return req_body

    def parse_response_into_a_bedrock_response(self, response: Dict) -> BedrockResponse:

        if len(response["content"]) < 2:
            text_response = ""
            logger.error(
                "Response does not have the text field. Returning an empty response in its place."
            )
        else:
            text_response = (
                response["content"][0]["thinking"]
                + "</thinking>\n\n"
                + response["content"][1]["text"]
            )

        return BedrockResponse(
            num_input_tokens=response["usage"]["input_tokens"],
            num_output_tokens=response["usage"]["output_tokens"],
            response=text_response,
            stop_reason=response["stop_reason"],
            cost=self.compute_cost(
                response["usage"]["input_tokens"],
                response["usage"]["output_tokens"],
            ),
        )


class Claude37Sonnet(ClaudeReasoning):
    price_per_1000_input_tokens = 0.003
    price_per_1000_output_tokens = 0.015


class Claude45Haiku(ClaudeReasoning):

    @property
    def price_per_1000_input_tokens(self):
        return 0.001 * 0.24

    @property
    def price_per_1000_output_tokens(self):
        return 0.005 * 0.24


class Claude45Sonnet(ClaudeReasoning):

    @property
    def price_per_1000_input_tokens(self):
        return 0.003 * 0.36

    @property
    def price_per_1000_output_tokens(self):
        return 0.015 * 0.36
