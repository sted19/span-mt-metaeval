# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
from typing import Optional, Tuple
import logging
import aioboto3
from aiobotocore.config import AioConfig
import boto3
import socket
from botocore.config import Config

bedrock2hf_model_ids = {
    "us.meta.llama3-1-8b-instruct-v1:0": "meta-llama/Llama-3.1-8B-Instruct",
    "us.meta.llama4-scout-17b-instruct-v1:0": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "us.meta.llama4-maverick-17b-instruct-v1:0": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "us.meta.llama3-3-70b-instruct-v1:0": "meta-llama/Llama-3.3-70B-Instruct",
}

logger = logging.getLogger(__name__)


def get_bedrock_session_and_config_sync(
    assume_role: Optional[str] = None,
    region: Optional[str] = None,
) -> Tuple[boto3.Session, Config]:
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assume_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-2").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    """
    target_region = region if region is not None else os.environ.get("AWS_REGION", None)
    if target_region is None:
        raise SystemError(
            "No region provided. Define a region in the parameters or in the env. var. AWS_REGION"
        )

    logger.debug(f"Create new client\n  Using region: {target_region}")

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
        read_timeout=20
        * 60,  # takes about 90s for a full 4096 output in Claude3.*-Sonnet
    )

    session_kwargs = {"region_name": target_region}
    profile_name = os.environ.get("AWS_PROFILE")

    if profile_name:
        logger.debug(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    logger.debug(f"Assume role is {assume_role}")

    session = boto3.Session(**session_kwargs)
    logger.info(
        f"Created new session with region {target_region} and use profile {profile_name}"
    )

    if assume_role:
        logger.debug(f"  Using role: {assume_role}")
        sts_endpoint_url = f"https://sts.{target_region}.amazonaws.com"
        logger.info(f"Using regional STS endpoint: {sts_endpoint_url}")

        sts = session.client("sts", config=retry_config, endpoint_url=sts_endpoint_url)
        response = sts.assume_role(
            RoleArn=str(assume_role), RoleSessionName="bedrock-batch-session"
        )
        logger.info(f"Assumed role: {assume_role}")

        session = boto3.Session(
            aws_access_key_id=response["Credentials"]["AccessKeyId"],
            aws_secret_access_key=response["Credentials"]["SecretAccessKey"],
            aws_session_token=response["Credentials"]["SessionToken"],
            region_name=target_region,
        )

    return session, retry_config


async def get_bedrock_session_and_config(
    assume_role: Optional[str] = None,
    region: Optional[str] = None,
    max_pool_connections: int = 50,
) -> Tuple[aioboto3.Session, AioConfig]:
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assume_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-2").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    max_pool_connections:
        Maximum number of connections that will be created from the same session
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    target_region = region if region is not None else os.environ.get("AWS_REGION", None)
    if target_region is None:
        raise SystemError(
            "No region provided. Define a region in the parameters or in the env. var. AWS_REGION"
        )

    logger.debug(f"Create new client\n  Using region: {target_region}")

    retry_config = AioConfig(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "adaptive",
        },
        max_pool_connections=max_pool_connections,
        read_timeout=20
        * 60,  # takes about 90s for a full 4096 output in Claude3.*-Sonnet
        # To prevent dropping the connection if response.read() is taking too long
        tcp_keepalive=True,
    )

    session_kwargs = {"region_name": target_region}
    profile_name = os.environ.get("AWS_PROFILE")

    if profile_name:
        logger.debug(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    logger.debug(f"Assume role is {assume_role}")

    session = aioboto3.Session(**session_kwargs)
    logger.info(
        f"Created new session with region {target_region} and use profile {profile_name}"
    )

    if assume_role:
        logger.debug(f"  Using role: {assume_role}")
        sts_endpoint_url = f"https://sts.{target_region}.amazonaws.com"
        logger.info(f"Using regional STS endpoint: {sts_endpoint_url}")

        async with session.client(
            "sts", config=retry_config, endpoint_url=sts_endpoint_url
        ) as sts:
            response = await sts.assume_role(
                RoleArn=str(assume_role), RoleSessionName="langchain-llm-1"
            )
            logger.info(f"Assumed role: {assume_role}")

            session = aioboto3.Session(
                aws_access_key_id=response["Credentials"]["AccessKeyId"],
                aws_secret_access_key=response["Credentials"]["SecretAccessKey"],
                aws_session_token=response["Credentials"]["SessionToken"],
            )

    return session, retry_config
