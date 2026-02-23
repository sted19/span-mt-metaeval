# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import os
from typing import List, Dict, Optional, Tuple, Any
from botocore.exceptions import ClientError
import aioboto3
from aiobotocore.config import AioConfig
import asyncio
import json
import time
from datetime import datetime

from mt_evaluation.core import FewShots
from mt_evaluation.models.base import Model
from mt_evaluation.models.bedrock import BedrockResponse
from mt_evaluation.models.bedrock.utils import (
    get_bedrock_session_and_config,
    get_bedrock_session_and_config_sync,
)

logger = logging.getLogger(__name__)


class BedrockModel(Model):

    price_per_1000_input_tokens: float = None
    price_per_1000_output_tokens: float = None

    max_attempts = 1

    def __init__(
        self,
        model_id,
        max_concurrent: int = 5,
        bedrock_assume_role: Optional[str] = None,
        use_batch_inference: bool = False,
        s3_bucket_name: Optional[str] = None,
        s3_bucket_dir: Optional[str] = None,
        bedrock_batch_job_role_arn: Optional[str] = None,
        bedrock_jobs_log_filepath: str = "logs/bedrock_jobs.json",
        bedrock_region: Optional[str] = None,
    ):
        super().__init__(model_id)
        self.max_concurrent = max_concurrent
        self.assume_role = bedrock_assume_role
        self.use_batch_inference = use_batch_inference
        self.s3_bucket_name = s3_bucket_name
        self.s3_bucket_dir = s3_bucket_dir
        self.batch_job_role_arn = bedrock_batch_job_role_arn
        self.bedrock_jobs_log_filepath = bedrock_jobs_log_filepath
        self.bedrock_region = (
            bedrock_region if bedrock_region is not None else os.environ["AWS_REGION"]
        )

        self._semaphore = None
        self._semaphore_loop = None
        self._session = None
        self._retry_config = None
        self.s3 = None

        if self.use_batch_inference:
            if self.s3_bucket_name is None:
                raise ValueError(
                    "s3_bucket_name must be provided if use_batch_inference is True"
                )
            if self.s3_bucket_dir is None:
                raise ValueError(
                    "s3_bucket_dir must be provided if use_batch_inference is True"
                )
            if self.batch_job_role_arn is None:
                raise ValueError(
                    "batch_job_role_arn must be provided if use_batch_inference is True"
                )
            bedrock_session, bedrock_config = get_bedrock_session_and_config_sync(
                assume_role=self.assume_role, region=self.bedrock_region
            )
            self.s3 = bedrock_session.client("s3")
            self.bedrock_client = bedrock_session.client(
                "bedrock", config=bedrock_config
            )

    def compute_responses_missing_from_batch_inference(
        self,
        missing_user_prompts: List[str],
        missing_system_prompts: List[str],
        missing_few_shots: List[FewShots],
        **kwargs,
    ) -> List[BedrockResponse]:
        # Count the missing responses and ask the user for approval to collect them via API calls.

        """Get user confirmation before computing the responses missing from a batch inference using API calls"""
        print("\n" + "=" * 60)
        print(
            f"{len(missing_user_prompts)} RESPONSES ARE MISSING FROM YOUR BATCH INFERENCE!"
        )
        print("=" * 60)

        while True:
            response = (
                input(
                    "\nDo you want to proceed with computing the missing responses via standard bedrock API calls? (y/n) "
                )
                .lower()
                .strip()
            )
            if response in ["y", "yes"]:
                return asyncio.run(
                    self._async_call(
                        missing_system_prompts,
                        missing_user_prompts,
                        missing_few_shots,
                        **kwargs,
                    )
                )
            elif response in ["n", "no"]:
                print("New API calls cancelled by user.")
                exit()
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    def get_input_key(self, job_name: str) -> str:
        return f"{self.s3_bucket_dir}/inputs/{job_name}/input.jsonl"

    def get_output_key_and_manifest_key(
        self, job_name: str, job_arn: str
    ) -> Tuple[str, str]:
        alphanumeric_dir = job_arn.split("/")[-1]
        return (
            f"{self.s3_bucket_dir}/outputs/{job_name}/{alphanumeric_dir}/input.jsonl.out",
            f"{self.s3_bucket_dir}/outputs/{job_name}/{alphanumeric_dir}/manifest.jsonl.out",
        )

    @classmethod
    def generate_unique_job_name(cls, base_job_name: str) -> str:
        """Generate unique job name with timestamp"""

        timestamp = datetime.now().strftime("%S%M%H_%d%m%Y")

        # Job name cannot be longer than 63 characters nor contain underscores
        unique_job_name = (
            f"{base_job_name}-{timestamp}".replace("/", "-")
            .replace(".", "-")
            .replace(":", "-")
            .replace("_", "-")[:63]
        )

        return unique_job_name

    def get_input_uri(self, job_name: str) -> str:
        """
        :param job_name: the name of the job
        :return: The location of the input file, based on bucket_name, bucket dir, and job name
        """
        return f"s3://{self.s3_bucket_name}/{self.s3_bucket_dir}/inputs/{job_name}/input.jsonl"

    def get_output_uri(self, job_name: str) -> str:
        """

        :param job_name:
        :return: The location of the output file, based on bucket_name, bucket dir, and job name
        """
        return f"s3://{self.s3_bucket_name}/{self.s3_bucket_dir}/outputs/{job_name}/"

    def upload_records_to_s3(self, records: List[Dict], job_name: str):
        jsonl_content = "\n".join(json.dumps(record) for record in records)
        s3_key = self.get_input_key(job_name)

        self.s3.put_object(
            Bucket=self.s3_bucket_name,
            Key=s3_key,
            Body=jsonl_content,
            ContentType="application/json",
        )
        logger.info(
            f"Uploaded {len(records)} records to {self.s3_bucket_name} at {s3_key}"
        )

    def is_output_already_computed(self, job_name: str, job_arn: str) -> bool:
        """Check if output file already exists in S3"""
        try:
            output_key, _ = self.get_output_key_and_manifest_key(job_name, job_arn)
            self.s3.head_object(Bucket=self.s3_bucket_name, Key=output_key)
            logger.info(
                f"Found existing results at s3://{self.s3_bucket_name}/{output_key}"
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info("No existing results found")
                return False
            else:
                raise e

    def download_results_from_s3(self, job_name: str, job_arn: str) -> Optional[Dict]:
        """Download and parse results from S3"""
        output_key, _ = self.get_output_key_and_manifest_key(job_name, job_arn)
        try:
            response = self.s3.get_object(Bucket=self.s3_bucket_name, Key=output_key)

            logger.info(f"Downloaded response from S3")
            return response
        except ClientError as e:
            logger.error(f"Failed to download results: {e}")
            raise

    def load_job_log(self):
        """Load existing jobs log"""
        if os.path.exists(self.bedrock_jobs_log_filepath):
            try:
                with open(self.bedrock_jobs_log_filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error("Error while loading batch job log.")
                logger.error(e)
                return {}
        return {}

    def save_job_log(self, jobs_log):
        """Save jobs log to file"""
        with open(self.bedrock_jobs_log_filepath, "w") as f:
            json.dump(jobs_log, f, indent=2, default=str)

    @classmethod
    def get_user_confirmation_for_new_batch_inference_job(cls, job_name) -> bool:
        """Get user confirmation before creating a new batch job"""
        print("\n" + "=" * 60)
        print("🚀 NEW BATCH INFERENCE JOB CREATION")
        print("=" * 60)
        print(f"Job Name: {job_name}")
        print("This will create a new Amazon Bedrock batch inference job.")

        while True:
            response = (
                input("\nDo you want to proceed with creating this job? (y/n): ")
                .lower()
                .strip()
            )
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                print("Job creation cancelled by user.")
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    def display_existing_job_info(self, stored_name, job_info):
        """Display information about existing job"""
        print("\n" + "=" * 60)
        print("✅ EXISTING JOB FOUND")
        print("=" * 60)
        print(f"Job Name: {stored_name}")
        print(f"Job ARN: {job_info['job_arn']}")
        print(f"Created: {job_info.get('created_at', 'Unknown')}")

        # Try to get current status
        try:
            response = self.bedrock_client.get_model_invocation_job(
                jobIdentifier=job_info["job_arn"]
            )
            status = response["status"]
            print(f"Status: {status}")

        except Exception as e:
            print(f"Status: Unable to fetch (Error: {e})")

        print("=" * 60)

    def create_batch_job(
        self, json_records: List[Dict], job_name: str, overwrite_existing: bool = False
    ) -> str:
        """
        Create a batch inference job and return job ARN

        Before creating the job, check whether a job with the same base name has been already executed.
        In that case, return its job arn instead of launching another one.
        """

        jobs_log = self.load_job_log()

        for job in jobs_log:

            if job == job_name:
                logger.info(f"Found existing job with the same name: {job_name}")
                if self.is_output_already_computed(job_name, jobs_log[job]["job_arn"]):
                    self.display_existing_job_info(job_name, jobs_log[job])

                    if not overwrite_existing:
                        return jobs_log[job]["job_arn"]
                    else:
                        logger.warning(
                            f"You are going to overwrite the existing job: {job_name}."
                        )
                else:
                    logger.warning(
                        f"Existing job found but output is not present in the S3 bucket: {job_name}. "
                        f"You have to recompute it from scratch."
                    )

        if not self.get_user_confirmation_for_new_batch_inference_job(job_name):
            exit()

        self.upload_records_to_s3(json_records, job_name)

        logger.info(f"Creating batch job: {job_name}")
        input_s3_uri = self.get_input_uri(job_name)
        output_s3_uri = self.get_output_uri(job_name)

        unique_job_name = self.generate_unique_job_name(job_name)

        try:
            response = self.bedrock_client.create_model_invocation_job(
                jobName=unique_job_name,
                roleArn=self.batch_job_role_arn,
                modelId=self.name,
                inputDataConfig={"s3InputDataConfig": {"s3Uri": input_s3_uri}},
                outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_s3_uri}},
            )

            job_arn = response["jobArn"]
            logger.info(f"Created batch job: {unique_job_name} (ARN: {job_arn})")
            logger.info(f"Input: {input_s3_uri}")
            logger.info(f"Output: {output_s3_uri}")

            jobs_log[job_name] = {
                "job_arn": job_arn,
                "job_unique_name": unique_job_name,
                "input_s3_uri": input_s3_uri,
                "output_s3_uri": output_s3_uri,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.save_job_log(jobs_log)

            return job_arn

        except ClientError as e:
            logger.error(f"Failed to create batch job: {e}")
            logger.error(f"Input URI: {input_s3_uri}")
            logger.error(f"Output URI: {output_s3_uri}")
            logger.error(f"Model ID: {self.name}")
            logger.error(f"Role ARN: {self.batch_job_role_arn}")
            raise

    def wait_for_batch_job_start(
        self, job_arn: str, poll_interval: int = 20, max_wait_time: int = 600
    ) -> str:
        """
        Wait for batch job to complete and return final status

        Args:
            job_arn: The ARN of the batch job
            poll_interval: How often to check status (seconds)
            max_wait_time: Maximum time to wait (seconds)

        Returns:
            Final job status if it is InProgress of Completed.

        Raises:
            RuntimeError: if the job is Failed, Stopped, or Stopping
            TimeoutError: if the job never exits the Submitted phase
            ClientError: if the program fails to retrieve the job status
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                response = self.bedrock_client.get_model_invocation_job(
                    jobIdentifier=job_arn
                )
                status = response["status"]

                logger.info(f"Batch job status: {status}")

                # Job has started successfully
                if status in ["InProgress", "Scheduled"]:
                    logger.info("Batch job started successfully! Exiting.")
                    exit()

                elif status == "Completed":
                    logger.info("Batch job completed successfully")
                    return status

                # Job failed to start or was stopped
                elif status in ["Failed", "Stopping", "Stopped"]:
                    error_msg = f"Batch job failed to start properly. Status: {status}"
                    if "failureMessage" in response:
                        error_msg += f". Failure reason: {response['failureMessage']}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Job is in "Submitted" status - keep waiting
                elif status == "Submitted":
                    logger.debug(f"Job submitted, waiting for it to start...")
                    time.sleep(poll_interval)

                # Job is in "Validating" status - keep waiting
                elif status == "Validating":
                    logger.debug(f"Job being validated, waiting for it to start...")
                    time.sleep(poll_interval)

                # Handle any other unexpected statuses
                else:
                    logger.warning(
                        f"Unexpected job status: {status}, continuing to wait..."
                    )
                    time.sleep(poll_interval)

            except ClientError as e:
                logger.error(f"Error checking job status: {e}")
                time.sleep(poll_interval)

        raise TimeoutError(f"Batch job did not start within {max_wait_time} seconds")

    @property
    def semaphore(self):
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error(
                "No running event loop found. Returning no semaphore. Unclear what are the consequences."
            )
            return None

        if self._semaphore is None or self._semaphore_loop is not current_loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
            self._semaphore_loop = current_loop

        return self._semaphore

    async def _get_session_and_config(self) -> Tuple[aioboto3.Session, AioConfig]:
        """Get or create the aioboto3 session and retry_config"""
        if self._session is None or self._retry_config is None:
            self._session, self._retry_config = await get_bedrock_session_and_config(
                assume_role=self.assume_role,
                region=self.bedrock_region,
                max_pool_connections=self.max_concurrent * 2,  # Allow some buffer
            )
        return self._session, self._retry_config

    async def read_response(self, response):
        pass

    async def invoke_model_and_read_response(self, inputs: Dict) -> Dict | None:
        """
        Invoke Bedrock model with automatic credential refresh on auth errors
        """

        async with self.semaphore:
            last_exception = None
            t = 3

            for attempt in range(self.max_attempts):
                try:
                    session, retry_config = await self._get_session_and_config()

                    async with session.client(
                        "bedrock-runtime", config=retry_config
                    ) as client:
                        response = await client.invoke_model(**inputs)
                        response_body_data = await response.get("body").read()
                        response_body_str = response_body_data.decode("utf-8")
                        parsed_response = json.loads(response_body_str)
                        return parsed_response

                except (
                    ClientError,
                    RuntimeError,
                    json.JSONDecodeError,
                    UnicodeError,
                ) as e:
                    last_exception = e
                    logger.debug(f"Error on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(t)
                    t = min(t * 3, 30)
                    print(f"New sleep time is {t}")

            # All attempts failed
            print(f"All {self.max_attempts} attempts failed")
            if last_exception:
                raise last_exception

            return None

    async def single_call(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet")

    def __call__(
        self,
        system_prompts: List[str],
        user_prompts: List[str],
        few_shots: List[FewShots],
        job_name: Optional[str] = None,
        **kwargs,
    ) -> List[BedrockResponse]:
        """
        Synchronous interface that internally uses async for concurrent processing
        """
        if self.use_batch_inference:
            if job_name is None:
                raise ValueError(
                    "job_name must be provided if use_batch_inference is True"
                )
            return self.batch_inference(
                system_prompts,
                user_prompts,
                few_shots,
                job_name,
                **kwargs,
            )
        else:
            return asyncio.run(
                self._async_call(system_prompts, user_prompts, few_shots, **kwargs)
            )

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
        """
        Internal batch inference implementation - to be overridden by subclasses
        """
        raise NotImplementedError("Must be implemented by subclass!")

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
        Internal async implementation - to be overridden by subclasses
        """
        raise NotImplementedError("Must be implemented by subclass!")

    def compute_cost(self, num_input_tokens: int, num_output_tokens: int) -> float:
        if (
            self.price_per_1000_input_tokens is None
            or self.price_per_1000_output_tokens is None
        ):
            print("WARNING: Information about model price not set! Returning 0.")
            return 0.0
        price = (
            self.price_per_1000_input_tokens * num_input_tokens / 1e3
            + self.price_per_1000_output_tokens * num_output_tokens / 1e3
        )

        if self.use_batch_inference:
            price = price * 0.5

        return price
