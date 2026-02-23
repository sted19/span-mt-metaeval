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


class Qwen(BedrockModel):

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

        messages = self._prepare_messages(system_prompts, user_prompts, few_shots)

        def hash_message(msgs: List[Dict]):
            """Create a hash of the message content for unique identification"""
            content = {"messages": msgs}
            content_str = json.dumps(content, sort_keys=True)
            return hashlib.md5(content_str.encode()).hexdigest()

        # Keep track of original indices based on the input messages
        message_hash_to_index: Dict[str, int] = {}
        json_records = []

        for idx, message in enumerate(messages):
            message_hash = hash_message(message)
            message_hash_to_index[message_hash] = idx

            model_input = self._prepare_request_body(
                model=self.name,
                messages=message,
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                reasoning_effort=kwargs.get("reasoning_effort", None),
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
            )

            try:
                bedrock_response = BedrockResponse(
                    num_input_tokens=raw_response["modelOutput"]["usage"][
                        "prompt_tokens"
                    ],
                    num_output_tokens=raw_response["modelOutput"]["usage"][
                        "completion_tokens"
                    ],
                    response=raw_response["modelOutput"]["choices"][0]["message"][
                        "content"
                    ],
                    stop_reason=raw_response["modelOutput"]["choices"][0][
                        "finish_reason"
                    ],
                    cost=self.compute_cost(
                        raw_response["modelOutput"]["usage"]["prompt_tokens"],
                        raw_response["modelOutput"]["usage"]["completion_tokens"],
                    ),
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
    def _prepare_messages(
        cls,
        system_prompts: List[str],
        user_prompts: List[str],
        few_shots: List[FewShots],
    ) -> List[List[Dict]]:

        if len(user_prompts) != len(few_shots) != len(system_prompts):
            raise ValueError(
                f"Number of system prompts, user prompts, and few-shots must match! ({len(system_prompts)}, {len(user_prompts)}, {len(few_shots)})"
            )

        messages = []
        for system_prompt, user_prompt, sample_few_shots in zip(
            system_prompts, user_prompts, few_shots
        ):
            sample_message = [{"role": "system", "content": system_prompt}]

            for shot_user_prompt, shot_response in zip(
                sample_few_shots.user_prompts, sample_few_shots.assistant_responses
            ):
                sample_message += [
                    {
                        "role": "user",
                        "content": shot_user_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": shot_response,
                    },
                ]

            sample_message += [
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ]

            messages.append(sample_message)

        return messages

    @classmethod
    def _prepare_request_body(
        cls,
        model: str,
        messages: List[Dict],
        max_completion_tokens: int,
        temperature: float,
        top_p: float,
        stream: bool,
        reasoning_effort: str | None,
    ) -> Dict:
        return {
            # Omitting model name as it was giving an error with gpt-oss
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "reasoning_effort": reasoning_effort,
        }

    async def single_call(
        self,
        messages: List[Dict],
        inf_params: Dict,
    ):

        invoke_model_inputs = {
            "body": json.dumps(
                self._prepare_request_body(self.name, messages, **inf_params)
            ),
            "modelId": self.name,
        }

        return await self.invoke_model_and_read_response(invoke_model_inputs)

    async def concurrent_call(
        self,
        messages: List[List[Dict]],
        inf_params: Dict,
    ):

        tasks = []
        for sample_msg in messages:
            tasks.append(self.single_call(sample_msg, inf_params))

        responses = await asyncio.gather(*tasks)

        return responses

    async def _async_call(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        few_shots: List[FewShots],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
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
            List[BedrockResponse]: List of responses, one per prompt
        """

        messages = self._prepare_messages(system_prompts, user_prompts, few_shots)

        inf_params = {
            "max_completion_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,  # can be omitted
            "reasoning_effort": kwargs.get("reasoning_effort", None),
        }

        if len(messages) == 1:
            messages = messages[0]

            responses = [await self.single_call(messages, inf_params)]
        else:
            responses = await self.concurrent_call(messages, inf_params)

        responses = [
            BedrockResponse(
                num_input_tokens=response["usage"]["prompt_tokens"],
                num_output_tokens=response["usage"]["completion_tokens"],
                response=response["choices"][0]["message"]["content"],
                stop_reason=response["choices"][0]["finish_reason"],
                cost=self.compute_cost(
                    response["usage"]["prompt_tokens"],
                    response["usage"]["completion_tokens"],
                ),
            )
            for response in responses
        ]

        return responses


class Qwen3235B(Qwen):
    @property
    def price_per_1000_input_tokens(self):
        return 0.00022 * 0.42

    @property
    def price_per_1000_output_tokens(self):
        return 0.00088 * 0.42


class Qwen332B(Qwen):
    @property
    def price_per_1000_input_tokens(self):
        return 0.00015 * 0.48

    @property
    def price_per_1000_output_tokens(self):
        return 0.0006 * 0.48
