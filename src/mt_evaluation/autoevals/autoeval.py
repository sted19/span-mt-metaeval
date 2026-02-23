# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Abstract base class for automatic evaluation of machine translation.

This module defines the AutoEval abstract base class that serves as the foundation
for all automatic evaluation implementations in the framework.
"""

from typing import List, Optional, Dict
from abc import ABC, abstractmethod
import logging

from mt_evaluation.core import AutomaticEvaluation, Prompt, Sample
from mt_evaluation.models.base import Model
from mt_evaluation.core import FewShots, Response

logger = logging.getLogger(__name__)


class AutoEval(ABC):
    """
    Abstract base class for automatic evaluation of machine translation.

    This class defines the interface that all automatic evaluators must implement.
    It handles the common workflow of formatting prompts, calling the model, and
    processing responses.

    Attributes:
        prompt: The prompt template used for evaluation.
        model: The language model used for evaluation.
        name: The name/identifier of this evaluator.
    """

    prompt: Optional[Prompt] = None

    def __init__(self, model: Model, name: str):
        """
        Initialize the AutoEval instance.

        Args:
            model: The language model to use for evaluation.
            name: The name/identifier for this evaluator.

        Raises:
            ValueError: If prompt is not set by the subclass.
        """
        if self.prompt is None:
            raise ValueError(f"Prompt must be set by {self.__class__.__name__}")

        self.model = model
        self.name = name
        logger.info(f"Initialized {self.__class__.__name__} with model {model.name}")

    @abstractmethod
    def parse_response(self, response: str) -> AutomaticEvaluation:
        """
        Parse the model response into an Evaluation object.

        This method must be implemented by subclasses to handle the specific
        format of responses expected from their evaluation approach.

        Args:
            response: The raw response string from the language model.

        Returns:
            AutomaticEvaluation: Parsed evaluation containing errors, score, and annotation.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement parse_response")

    def get_final_instruction(self, is_reasoning: bool = False) -> Optional[str]:
        """
        Get the final instruction to append to the user prompt.
        
        Override this method in subclasses to provide custom final instructions
        based on whether reasoning mode is enabled.
        
        Args:
            is_reasoning: Whether the model is using reasoning/thinking mode.
            
        Returns:
            Optional[str]: The final instruction string, or None if not needed.
        """
        return None

    def format_prompts(
        self,
        sample: Sample,
        use_few_shots: bool,
        is_reasoning: bool = False,
    ) -> Dict[str, str | FewShots]:
        """
        Format prompts for a sample.
        
        Args:
            sample: The sample to format prompts for.
            use_few_shots: Whether to include few-shot examples.
            is_reasoning: Whether the model is using reasoning/thinking mode.
            
        Returns:
            Dict containing system_prompt, user_prompt, and few_shots.
        """
        system_prompt = self.prompt.format_system_prompt()
        
        # Get optional final instruction from subclass
        final_instruction = self.get_final_instruction(is_reasoning)
        
        # Build format kwargs
        format_kwargs = {}
        if final_instruction is not None:
            format_kwargs["final_instruction"] = final_instruction
        
        user_prompt = self.prompt.format_user_prompt(
            src=sample.src,
            tgt=sample.tgt,
            src_lang=sample.src_lang,
            tgt_lang=sample.tgt_lang,
            **format_kwargs,
        )
        sample_few_shots = (
            self.prompt.format_few_shots()
            if use_few_shots and self.prompt.few_shots
            else FewShots(user_prompts=[], assistant_responses=[])
        )

        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "few_shots": sample_few_shots,
        }

    def create_evaluations_from_responses(
        self,
        samples: List[Sample],
        responses: List[Response],
        formatted_prompts: List[Dict[str, str | FewShots]],
    ) -> List[AutomaticEvaluation]:
        if len(responses) != len(samples):
            raise RuntimeError(
                f"Model returned {len(responses)} responses for {len(samples)} samples"
            )

        # Extract costs from responses
        costs = [getattr(response, "cost", 0.0) for response in responses]

        logger.info("Responses received. Parsing them...")

        # Parse responses into evaluations
        evaluations: List[AutomaticEvaluation] = []
        for i, response in enumerate(responses):
            evaluations.append(self.parse_response(response.response))

        # Add metadata to evaluations
        final_evaluations = []
        for evaluation, formatted_prompt, cost in zip(
            evaluations, formatted_prompts, costs
        ):
            final_evaluation = AutomaticEvaluation(
                annotation=evaluation.annotation,
                errors=evaluation.errors,
                score=evaluation.score,
                user_prompt=formatted_prompt["user_prompt"],
                system_prompt=formatted_prompt["system_prompt"],
                few_shots=formatted_prompt["few_shots"],
                cost=cost,
                parsing_error=evaluation.parsing_error,
            )
            final_evaluations.append(final_evaluation)

        return final_evaluations

    def evaluate(
        self,
        samples: List[Sample],
        use_few_shots: bool = True,
        **kwargs,
    ) -> List[AutomaticEvaluation]:
        """
        Evaluate all the provided samples.

        This method orchestrates the evaluation process by:
        1. Formatting prompts for each sample
        2. Calling the language model
        3. Parsing responses into Evaluation objects
        4. Adding metadata (prompts, costs) to evaluations

        Args:
            samples: List of translation samples to evaluate.
            use_few_shots: Whether to include few-shot examples in prompts.
            **kwargs: Additional keyword arguments passed to the model.
                     Arguments starting with "generation_" are passed to model generation.

        Returns:
            List[AutomaticEvaluation]: List of evaluation results, one per sample.

        Raises:
            ValueError: If samples list is empty.
            RuntimeError: If model evaluation fails.
        """
        if not samples:
            raise ValueError("Cannot evaluate empty list of samples")

        # Extract generation-specific kwargs
        generation_kwargs = {
            k[len("generation_") :]: v
            for k, v in kwargs.items()
            if k.startswith("generation_")
        }

        logger.info(f"Generation kwargs: {generation_kwargs}")

        # Format prompts for all samples
        formatted_prompts = []
        system_prompts, user_prompts, few_shots = [], [], []

        for i, sample in enumerate(samples):
            try:
                formatted_prompt = self.format_prompts(
                    sample,
                    use_few_shots,
                    is_reasoning=generation_kwargs.get("reasoning_effort", None)
                    is not None,
                )
                formatted_prompts.append(formatted_prompt)
                system_prompts.append(formatted_prompt["system_prompt"])
                user_prompts.append(formatted_prompt["user_prompt"])
                few_shots.append(formatted_prompt["few_shots"])

            except Exception as e:
                logger.error(f"Error formatting prompt for sample {i}: {e}")
                raise RuntimeError(f"Failed to format prompt for sample {i}: {e}")

        # Call the model
        try:

            batch_inference_job_name = kwargs.get("batch_inference_job_name", None)
            batch_inference_overwrite_existing_job = kwargs.get(
                "batch_inference_overwrite_existing_job", False
            )
            responses: List[Response] = self.model(
                system_prompts,
                user_prompts,
                few_shots,
                job_name=batch_inference_job_name,
                overwrite_existing_job=batch_inference_overwrite_existing_job,
                **generation_kwargs,
            )
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise RuntimeError(f"Model evaluation failed: {e}")

        evaluations = self.create_evaluations_from_responses(
            samples, responses, formatted_prompts
        )

        return evaluations

    def __str__(self) -> str:
        """Return string representation of the evaluator."""
        return f"{self.__class__.__name__}(name={self.name}, model={self.model.name})"

    def __repr__(self) -> str:
        """Return detailed string representation of the evaluator."""
        return self.__str__()
