# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from typing import List, Dict, Union, Optional, Tuple
import json
import asyncio
import hashlib
import logging

from transformers import AutoTokenizer, AutoProcessor

from mt_evaluation.core import FewShots
from mt_evaluation.models.bedrock import BedrockResponse
from mt_evaluation.models.bedrock.base import BedrockModel
from mt_evaluation.models.bedrock.utils import bedrock2hf_model_ids

logger = logging.getLogger(__name__)


class Llama(BedrockModel):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = None

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

        formatted_messages = self._prepare_messages(
            system_prompts=system_prompts,
            user_prompts=user_prompts,
            few_shots=few_shots,
        )

        def hash_message(msg: str):
            """Create a hash of the message content for unique identification"""
            return hashlib.md5(msg.encode()).hexdigest()

        # Keep track of original indices based on the input messages
        message_hash_to_index = {}
        json_records = []

        for idx, message in enumerate(formatted_messages):
            message_hash = hash_message(message)
            message_hash_to_index[message_hash] = idx

            model_input = self._prepare_request_body(
                formatted_message=message,
                max_new_tokens=max_new_tokens,
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

        # Download and parse results
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

        responses: List[BedrockResponse | None] = [None] * len(formatted_messages)
        extra_responses = []
        for raw_response in raw_responses:
            message_hash = hash_message(
                raw_response["modelInput"]["prompt"],
            )

            bedrock_response = BedrockResponse(
                num_input_tokens=raw_response["modelOutput"]["prompt_token_count"],
                num_output_tokens=raw_response["modelOutput"]["generation_token_count"],
                response=raw_response["modelOutput"]["generation"],
                stop_reason=raw_response["modelOutput"]["stop_reason"],
                cost=self.compute_cost(
                    raw_response["modelOutput"]["prompt_token_count"],
                    raw_response["modelOutput"]["generation_token_count"],
                ),
            )

            if message_hash in message_hash_to_index:
                original_index = message_hash_to_index[message_hash]
                responses[original_index] = bedrock_response
            else:
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
        formatted_message: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Dict:
        return {
            "prompt": formatted_message,
            "max_gen_len": min(
                max_new_tokens, 8192
            ),  # Llama does not accept larger max_gen_len value than 8192
            "temperature": temperature,
            "top_p": top_p,
        }

    def _prepare_messages(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        few_shots: List[FewShots],
    ) -> List[str]:

        if len(user_prompts) != len(few_shots) != len(system_prompts):
            raise ValueError(
                f"Number of system prompts, user prompts, and few-shots must match! ({len(system_prompts)}, {len(user_prompts)}, {len(few_shots)})"
            )

        messages = []
        for system_prompt, user_prompt, sample_few_shots in zip(
            system_prompts, user_prompts, few_shots
        ):
            sample_message = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ]

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

        formatted_messages = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        return formatted_messages

    async def single_call(
        self,
        formatted_message: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Dict[str, Union[str, int]]:

        invoke_model_inputs = {
            "body": json.dumps(
                self._prepare_request_body(
                    formatted_message, max_new_tokens, temperature, top_p
                )
            ),
            "modelId": self.name,
        }

        return await self.invoke_model_and_read_response(invoke_model_inputs)

    async def concurrent_call(
        self,
        formatted_messages: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> List[Dict[str, Union[str, int]]]:

        tasks = []
        for formatted_message in formatted_messages:
            tasks.append(
                self.single_call(formatted_message, max_new_tokens, temperature, top_p)
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
            List[str]: List of responses, one per prompt
        """

        formatted_messages = self._prepare_messages(
            system_prompts, user_prompts, few_shots
        )

        if len(formatted_messages) == 1:
            formatted_message = formatted_messages[0]

            responses = [
                await self.single_call(
                    formatted_message=formatted_message,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            ]
        else:
            responses = await self.concurrent_call(
                formatted_messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        responses = [
            BedrockResponse(
                num_input_tokens=response["prompt_token_count"],
                num_output_tokens=response["generation_token_count"],
                response=response["generation"],
                stop_reason=response["stop_reason"],
                cost=self.compute_cost(
                    response["prompt_token_count"], response["generation_token_count"]
                ),
            )
            for response in responses
        ]

        return responses


class Llama318B(Llama):

    price_per_1000_input_tokens = 0.00022
    price_per_1000_output_tokens = 0.00022

    def __init__(
        self,
        model_id,
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(bedrock2hf_model_ids[model_id])


class Llama3211B(Llama):
    price_per_1000_input_tokens = 0.00016
    price_per_1000_output_tokens = 0.00016

    def __init__(
        self,
        model_id,
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.tokenizer = AutoProcessor.from_pretrained(bedrock2hf_model_ids[model_id])


class Llama3370B(Llama):

    def __init__(
        self,
        model_id,
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.tokenizer = AutoProcessor.from_pretrained(bedrock2hf_model_ids[model_id])

    @property
    def price_per_1000_input_tokens(self):
        return 0.00072

    @property
    def price_per_1000_output_tokens(self):
        return 0.00072


class LLama4Scout(Llama):
    price_per_1000_input_tokens = 0.00017
    price_per_1000_output_tokens = 0.00066

    def __init__(
        self,
        model_id,
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.tokenizer = AutoProcessor.from_pretrained(bedrock2hf_model_ids[model_id])


class LLama4Maverick(Llama):
    price_per_1000_input_tokens = 0.00024
    price_per_1000_output_tokens = 0.00097

    def __init__(
        self,
        model_id,
        **kwargs,
    ):
        super().__init__(model_id, **kwargs)
        self.tokenizer = AutoProcessor.from_pretrained(bedrock2hf_model_ids[model_id])
