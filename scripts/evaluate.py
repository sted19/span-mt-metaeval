# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
MT Evaluation Pipeline Script.

This script runs automatic evaluation of machine translations using LLM-based
metrics and caches results for efficient re-evaluation.

Usage:
    python scripts/evaluate.py --model-id <model> --test-set <wmt23/wmt24/wmt25> \
        --evaluation-schema <schema> [--lps en-de zh-en]
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Type, Dict, Union
from tqdm import tqdm
import time
import shutil
import sys

# Core data structures and constants
from mt_evaluation.core import (
    Sample,
    AutomaticEvaluation,
    Response,
    wmt22_lps,
    wmt23_lps,
    wmt24_lps,
    wmt25_lps,
)

# Model factory and base class
from mt_evaluation.models import get_model, Model
from mt_evaluation.models.bedrock.base import BedrockModel
from mt_evaluation.models.bedrock.utils import get_bedrock_session_and_config_sync
from mt_evaluation.config import get_model_aliases
from mt_evaluation.utils import standardize_name

# Autoeval factory and base class
from mt_evaluation.autoevals import get_autoeval, AutoEval

# Data utilities
from mt_evaluation.data import (
    MTEvaluationCache,
    get_cache_dir,
    get_raters_evaluations,
    get_super_raters_from_raters,
    flatten_samples_for_evaluation,
)
from mt_evaluation.data.utils import reconstruct_document_context

# General utilities
from mt_evaluation.utils import configure_torch, setup_logging

logger = logging.getLogger(__name__)


def evaluate_and_cache(
    autoeval: AutoEval,
    samples: List[Sample],
    cache: MTEvaluationCache,
    save_frequency: int,
    generation_parameters: Dict,
    zero_shot: bool = False,
    use_batch_inference: bool = False,
    batch_inference_overwrite_existing_job: bool = False,
    batch_inference_job_name: str = None,
    parse_with_autoeval_and_overwrite_cache: bool = False,
) -> List[Sample]:
    """Main evaluation loop with caching"""

    # Separate samples into cached and uncached
    to_evaluate = []
    cached_count = 0

    # NOTE: this serves only to fix parsing errors: Load the cache, re-parse it with the autoeval, and write it back to disk (overwriting the previous one). Then exit()
    #  Differently from running the evaluation from scratch (or retrieving it from S3 using batch inference), the cache saved with this technique will not contain user prompts, system prompts, and few-shots. The reason is that we go over samples as loaded by the script, without the autoeval preprocessing based on the selected prompt
    if parse_with_autoeval_and_overwrite_cache:
        responses, formatted_prompts = [], []
        for sample in samples:
            assert cache.is_evaluated(
                sample
            ), "All samples must be evaluated to conduct this operation"

            responses.append(Response(response=cache.get_evaluation(sample).annotation))

            formatted_prompts.append(
                autoeval.format_prompts(
                    sample,
                    use_few_shots=not zero_shot,
                    is_reasoning=generation_parameters.get(
                        "generation_reasoning_effort", None
                    )
                    is not None,
                )
            )

        new_evaluations = autoeval.create_evaluations_from_responses(
            samples, responses, formatted_prompts
        )

        for sample, evaluation in zip(samples, new_evaluations):
            sample.evaluation = evaluation

        cache.overwrite_cache(samples)
        print("New cache stored on disk. Exiting.")
        exit()

    for sample in samples:
        if cache.is_evaluated(sample):
            sample.evaluation = cache.get_evaluation(sample)
            cached_count += 1
        else:
            to_evaluate.append(sample)

    print(f"Found {cached_count} cached evaluations, {len(to_evaluate)} to evaluate")

    if use_batch_inference:
        save_frequency = len(samples)  # Process all samples at once

    # Process uncached samples in batches of save_frequency
    total_batches = (len(to_evaluate) + save_frequency - 1) // save_frequency

    start_time = time.time()
    # Process uncached samples in batches of save_frequency
    all_costs = []
    for i in tqdm(
        range(0, len(to_evaluate), save_frequency),
        desc="Processing batches",
        total=total_batches,
        unit="batch",
    ):
        batch = to_evaluate[i : i + save_frequency]

        batch_evaluations = autoeval.evaluate(
            batch,
            use_few_shots=not zero_shot,
            batch_inference_overwrite_existing_job=batch_inference_overwrite_existing_job,
            batch_inference_job_name=batch_inference_job_name,
            **generation_parameters,
        )
        evaluation_costs = [evaluation.cost for evaluation in batch_evaluations]
        logger.info(f"Batch {i} cost: {sum(evaluation_costs)}")
        all_costs.extend(evaluation_costs)

        for sample, evaluation in zip(batch, batch_evaluations):
            cache.add_evaluation(sample, evaluation)

    end_time = time.time()
    elapsed_time = end_time - start_time
    cache.finalize()
    print(
        "\nEvaluation Summary\n"
        "===================================================================\n"
        f"Total num samples (including cache): {len(samples)}\n"
        f"Evaluated samples: {len(to_evaluate)}\n"
        f"Elapsed time: {elapsed_time} seconds\n"
        f"Total cost: {sum(all_costs)}\n"
    )

    return samples


def evaluate_without_caching(
    autoeval: AutoEval,
    samples: List[Sample],
    generation_parameters: Dict,
    zero_shot: bool = False,
    batch_inference_overwrite_existing_job: bool = False,
    batch_inference_job_name: str = None,
) -> List[Sample]:
    """Main evaluation loop with caching"""

    start_time = time.time()

    evaluations = autoeval.evaluate(
        samples,
        use_few_shots=not zero_shot,
        batch_inference_overwrite_existing_job=batch_inference_overwrite_existing_job,
        batch_inference_job_name=batch_inference_job_name,
        **generation_parameters,
    )
    costs = [evaluation.cost for evaluation in evaluations]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Num samples: {len(samples)}\nElapsed time: {elapsed_time} seconds")
    print(f"Total cost: {sum(costs)}")

    return samples


def get_max_src_len(samples: List[Sample]):
    return max([len(sample.src) for sample in samples])


def get_max_tgt_len(samples: List[Sample]):
    return max([len(sample.tgt) for sample in samples])


def delete_model_cache(
    model_cache_dir: Path,
    autoeval_name: str,
    autoeval_model_name: str,
    run_specific_info: str,
) -> bool:
    """
    Delete the cache directory for a specific model after user confirmation.

    Args:
        model_cache_dir: Full path to the cache file
        autoeval_name: Name of the autoeval (evaluation schema)
        autoeval_model_name: Name of the model
        run_specific_info: info specific to an evaluation, to differentiate between evaluation with same model and autoeval, but other different configurations

    Returns:
        bool: True if cache was deleted or does not exist, False if user cancelled
    """

    if not model_cache_dir.exists():
        logger.info(f"Cache directory does not exist: {model_cache_dir}")
        return True

    # Show what will be deleted
    cache_files = list(model_cache_dir.rglob("*"))
    cache_size = sum(f.stat().st_size for f in cache_files if f.is_file())
    cache_size_mb = cache_size / (1024 * 1024)

    print(f"\n⚠️  WARNING: This will permanently delete the cache for:")
    print(f"   Model: {autoeval_model_name}")
    print(f"   Evaluation Schema: {autoeval_name}")
    print(f"   Cache Directory: {model_cache_dir}")
    print(f"   Cache Size: {cache_size_mb:.2f} MB ({len(cache_files)} files)")
    print(f"\n🚨 This action cannot be undone!")

    # Request confirmation
    while True:
        response = (
            input("\nDo you want to proceed with deleting this cache? (y/n): ")
            .strip()
            .lower()
        )

        if response in ["yes", "y"]:
            try:
                shutil.rmtree(model_cache_dir)
                print(f"✅ Successfully deleted cache directory: {model_cache_dir}")
                model_cache_dir.mkdir(parents=True, exist_ok=True)
                return True
            except Exception as e:
                print(f"❌ Error deleting cache directory: {e}")
                return False
        elif response in ["no", "n"]:
            print("❌ Cache deletion cancelled by user.")
            return False
        else:
            print("Please enter 'yes' or 'no'.")


def delete_s3_cache(
    s3_client,
    s3_bucket_name: str,
    s3_cache_key: str,
    autoeval_name: str,
    autoeval_model_name: str,
) -> bool:
    """
    Delete the S3 cache for a specific model after user confirmation.

    Args:
        s3_client: Boto3 S3 client
        s3_bucket_name: Name of the S3 bucket
        s3_cache_key: S3 key for the cache file
        autoeval_name: Name of the autoeval (evaluation schema)
        autoeval_model_name: Name of the model

    Returns:
        bool: True if cache was deleted or does not exist, False if user cancelled
    """
    from botocore.exceptions import ClientError

    # Check if the S3 cache file exists and get its size
    try:
        response = s3_client.head_object(Bucket=s3_bucket_name, Key=s3_cache_key)
        cache_size_bytes = response["ContentLength"]
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        cache_exists = True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.info(
                f"S3 cache does not exist: s3://{s3_bucket_name}/{s3_cache_key}"
            )
            return True
        else:
            logger.error(f"Error checking S3 cache: {e}")
            return False
    except Exception as e:
        logger.error(f"Error checking S3 cache: {e}")
        return False

    if not cache_exists:
        logger.info(f"S3 cache does not exist: s3://{s3_bucket_name}/{s3_cache_key}")
        return True

    # Show what will be deleted
    print(f"\n⚠️  WARNING: This will permanently delete the S3 cache for:")
    print(f"   Model: {autoeval_model_name}")
    print(f"   Evaluation Schema: {autoeval_name}")
    print(f"   S3 Location: s3://{s3_bucket_name}/{s3_cache_key}")
    print(f"   Cache Size: {cache_size_mb:.2f} MB")
    print(f"\n🚨 This action cannot be undone!")

    # Request confirmation
    while True:
        response = (
            input("\nDo you want to proceed with deleting this S3 cache? (y/n): ")
            .strip()
            .lower()
        )

        if response in ["yes", "y"]:
            try:
                s3_client.delete_object(Bucket=s3_bucket_name, Key=s3_cache_key)
                print(
                    f"✅ Successfully deleted S3 cache: s3://{s3_bucket_name}/{s3_cache_key}"
                )
                return True
            except Exception as e:
                print(f"❌ Error deleting S3 cache: {e}")
                return False
        elif response in ["no", "n"]:
            print("❌ S3 cache deletion cancelled by user.")
            return False
        else:
            print("Please enter 'yes' or 'no'.")


def filter_data(
    samples: List[Sample],
    included_categories: Union[List[str], str],
    included_severities: List[str],
) -> List[Sample]:

    filtered_samples = []
    for sample in samples:
        if sample is None:
            continue

        if sample.human_evaluation is None:
            continue

        if not sample.human_evaluation.errors:
            continue

        has_matching_error = False
        for error in sample.human_evaluation.errors:

            matching_category = False
            if included_categories == "All":
                matching_category = True
            else:
                assert type(included_categories) == list

                if any(cat in error.category.lower() for cat in included_categories):
                    matching_category = True

            matching_severity = False
            if any(sev in error.severity.lower() for sev in included_severities):
                matching_severity = True

            if matching_category and matching_severity:
                has_matching_error = True

        if has_matching_error:
            filtered_samples.append(sample)

    logger.info(
        f"Samples before filtering: {len(samples)}\nSamples after filtering: {len(filtered_samples)}"
    )

    return filtered_samples


def parse_arguments():
    parser = argparse.ArgumentParser(description="MT Evaluation Pipeline")
    parser.add_argument("--model-id", required=True, help="Model identifier")
    parser.add_argument(
        "--test-set",
        required=True,
        choices=["wmt23", "wmt22", "wmt24", "wmt25"],
        help="Test set name",
    )
    parser.add_argument(
        "--lps",
        default=None,
        nargs="+",
        type=str,
        help="Language pair(s) to evaluate",
    )
    parser.add_argument("--evaluation-schema", required=True, help="Evaluation schema")
    parser.add_argument(
        "--batch-size", default=16, type=int, help="Batch size for inference"
    )
    parser.add_argument(
        "--cache-save-frequency", default=512, type=int, help="Save frequency"
    )
    parser.add_argument(
        "--zero-shot",
        type=bool,
        default=True,
        help="Run evaluation in zero shot",
    )
    parser.add_argument(
        "--logging-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--toy",
        default=0,
        type=int,
        help="Number of samples to evaluate (default: 0 --> all samples)",
    )
    parser.add_argument(
        "--cache-dir",
        default="outputs",
        type=str,
        help="Directory to cache evaluation results (default: outputs)",
    )
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable caching of evaluation results",
    )
    parser.add_argument(
        "--bedrock-max-concurrent",
        default=5,
        type=int,
        help="Maximum number of concurrent async calls",
    )
    parser.add_argument(
        "--bedrock-assume-role",
        default=None,
        type=str,
        help="IAM role ARN to assume for Bedrock API calls (can also set MT_EVAL_ASSUME_ROLE env var)",
    )
    parser.add_argument(
        "--delete-cache",
        action="store_true",
        help="Delete the cache for the specified model before evaluation (requires confirmation)",
    )
    parser.add_argument(
        "--delete-s3-cache",
        action="store_true",
        help="Delete the S3 cache for the specified model before evaluation (requires confirmation)",
    )
    # Some generations (~100, very few) were getting truncated with max_new_tokens=2048. The others instead were terminating normally (I have not encountered occasions where it gets stuck in loops). Therefore, I'd rather increase it by a lot, and prevent that it gets ever truncated. Furthermore, If I use reasoning models I will need a much, much larger number of output tokens.
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Maximum number of tokens for model generation",
    )
    parser.add_argument(
        "--use-batch-inference",
        action="store_true",
        help="Use bedrock batch inference for model generation",
    )
    parser.add_argument(
        "--s3-bucket-name",
        type=str,
        default=None,
        help="S3 bucket name for batch inference and caching (can also set MT_EVAL_S3_BUCKET env var)",
    )
    parser.add_argument(
        "--s3-bucket-dir",
        type=str,
        default="mt-evaluation",
        help="The path to the dir where input (in args.s3_bucket_dir/inputs) and output (in args.s3_bucket_dir/outputs) files will be stored",
    )
    parser.add_argument(
        "--bedrock-batch-job-role-arn",
        default=None,
        type=str,
        help="IAM role ARN that Bedrock assumes for batch inference (can also set MT_EVAL_BATCH_ROLE_ARN env var)",
    )
    parser.add_argument(
        "--bedrock-jobs-log-filepath",
        default="logs/bedrock_jobs.json",
        type=str,
        help="The path to the file where bedrock jobs info will be stored (default: logs/bedrock_jobs.json)",
    )
    parser.add_argument(
        "--batch-inference-overwrite-existing-job",
        action="store_true",
        help="Overwrite existing Batch Inference job if it exists (default: False)",
    )
    parser.add_argument(
        "--s3-bucket-cache-dir",
        type=str,
        default="mt-evaluation/cache/",
        help="The path to the dir where the cache backup files will be stored",
    )
    parser.add_argument(
        "--sync-s3",
        action="store_true",
        help="Only upload the local cache to S3. Evaluation is not run.",
    )
    parser.add_argument(
        "--use-s3-backup",
        action="store_true",
        help="Use S3 backup for cache",
    )
    parser.add_argument(
        "--bedrock-region",
        type=str,
        default="us-east-1",
        help="The region to use for bedrock (default: us-east-1)",
    )
    parser.add_argument(
        "--run-specific-info",
        type=str,
        default="default",
        help="Any additional information to be added to the run name (default: 'default')",
    )
    parser.add_argument(
        "--parse-with-autoeval-and-overwrite-cache",
        action="store_true",
        help="This is used to fix potential parsing errors that got written in the cache. Specifically, when we conduct the evaluation with an autoeval, it parses the model annotation for each sample and constructs an AutomaticEvaluation object which is saved to disk. It might happen that we made a mistake at parsing time, and we want to overwrite the AutomaticEvaluation saved on disk (and on S3).",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="To roughly control the number of reasoning tokens used by reasoning models.",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=4096,
        help="The maximum number of reasoning tokens used by reasoning models.",
    )

    parser.add_argument(
        "--filter-data",
        action="store_true",
        help="Whether to filter the samples to evaluate based on the error categories of the gold errors included in such data. If used, only the samples containing gold errors with the indicated categories/severities will be evaluated.",
    )

    parser.add_argument("--included-categories", type=str, nargs="+", default="All")
    parser.add_argument(
        "--included-severities",
        type=str,
        nargs="+",
        default=["minor", "major", "critical"],
    )
    parser.add_argument(
        "--gold-rating-key",
        type=str,
        default="mqm.super.1",
    )
    parser.add_argument(
        "--human-as-a-metric-rating-keys",
        type=str,
        nargs="*",
        default=["mqm.super.2", "mqm.super.3"],
    )
    parser.add_argument(
        "--num-paragraphs-per-document",
        type=int,
        default=8,
        help="How many paragraphs to include in document context when conducting document-level evaluation",
    )
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):

    configure_torch()
    setup_logging(args.logging_level)

    # Resolve S3 bucket name from CLI argument or environment variable
    if args.s3_bucket_name is None:
        args.s3_bucket_name = os.environ.get("MT_EVAL_S3_BUCKET")

    # Resolve IAM role ARNs from CLI arguments or environment variables
    if args.bedrock_assume_role is None:
        args.bedrock_assume_role = os.environ.get("MT_EVAL_ASSUME_ROLE")

    if args.bedrock_batch_job_role_arn is None:
        args.bedrock_batch_job_role_arn = os.environ.get("MT_EVAL_BATCH_ROLE_ARN")

    test_set = args.test_set

    test_set2lps = {
        "wmt22": wmt22_lps,
        "wmt23": wmt23_lps,
        "wmt24": wmt24_lps,
        "wmt25": wmt25_lps,
    }
    lps = test_set2lps[test_set]
    lps = args.lps if args.lps is not None else lps

    super_raters = (
        [args.gold_rating_key]
        if test_set == "wmt24" or test_set == "wmt25"
        else [args.gold_rating_key] + args.human_as_a_metric_rating_keys
    )

    rater2lp2sys2samples = get_raters_evaluations(test_set, lps)

    rater2lp2sys2samples = reconstruct_document_context(
        rater2lp2sys2samples, args.num_paragraphs_per_document
    )

    super_rater2lp2sys2samples = get_super_raters_from_raters(
        rater2lp2sys2samples, super_raters
    )
    data = flatten_samples_for_evaluation(super_rater2lp2sys2samples)

    if args.filter_data:
        data = filter_data(
            data,
            included_categories=args.included_categories,
            included_severities=args.included_severities,
        )

    if args.toy:
        data = data[: args.toy]

    logger.info(f"Loaded {len(data)} samples")

    model_class: Type[Model] = get_model(args.model_id)

    if issubclass(model_class, BedrockModel):
        model_class: Type[BedrockModel]
        model = model_class(
            args.model_id,
            max_concurrent=args.bedrock_max_concurrent,
            bedrock_assume_role=args.bedrock_assume_role,
            s3_bucket_name=args.s3_bucket_name,
            s3_bucket_dir=args.s3_bucket_dir,
            use_batch_inference=args.use_batch_inference,
            bedrock_batch_job_role_arn=args.bedrock_batch_job_role_arn,
            bedrock_jobs_log_filepath=args.bedrock_jobs_log_filepath,
            bedrock_region=args.bedrock_region,
        )
    else:
        model = model_class(args.model_id)

    logger.info(f"Instantiated model {model}")

    autoeval = get_autoeval(args.evaluation_schema)(
        model,
        args.evaluation_schema,
    )

    logger.info(f"Instantiated autoeval {autoeval}")

    # get cache dir and cache file
    autoeval_name = standardize_name(autoeval.name)
    autoeval_model_name = standardize_name(autoeval.model.name)
    cache_dir = get_cache_dir(
        autoeval_name, autoeval_model_name, args.cache_dir, args.run_specific_info
    )
    cache_file = cache_dir / "cache.jsonl"

    logger.debug(f"Max src length: {get_max_src_len(data)}")
    logger.debug(f"Max tgt length: {get_max_tgt_len(data)}")
    logger.debug(f"Prompt length: {len(autoeval.prompt.user_prompt)}")

    # Handle cache deletion if requested
    if args.delete_cache:

        cache_deleted = delete_model_cache(
            cache_dir,
            autoeval_name,
            autoeval_model_name,
            args.run_specific_info,
        )

        if not cache_deleted:
            logger.error("Exiting due to cache deletion failure or cancellation.")
            sys.exit(1)

    if args.delete_s3_cache:
        # Get S3 client and configuration for cache deletion
        session, _ = get_bedrock_session_and_config_sync(
            args.bedrock_assume_role,
            region=args.bedrock_region,
        )
        s3_client = session.client("s3")

        cache_file = cache_dir / "cache.jsonl"
        s3_cache_key = args.s3_bucket_cache_dir + str(cache_file)

        cache_deleted = delete_s3_cache(
            s3_client,
            args.s3_bucket_name,
            s3_cache_key,
            autoeval_name,
            autoeval_model_name,
        )

        if not cache_deleted:
            logger.error("Exiting due to S3 cache deletion failure or cancellation.")
            sys.exit(1)

    # Note: I'm not passing anymore temperature and top_p (but my code assumes them, so I pass none values). In claude, I adapted the code to handle none values. Do the same when using other models.
    # Taking the recommended reasoning and non-reasoning parameters from the Qwen3 best practices: https://huggingface.co/Qwen/Qwen3-0.6B#best-practices
    if args.reasoning_effort is not None:
        generation_parameters = {
            # "generation_temperature": 0.6,
            "generation_temperature": None,
            "generation_max_new_tokens": args.max_new_tokens,
            # "generation_top_p": 0.95,
            "generation_top_p": None,
            "generation_batch_size": args.batch_size,
            "generation_reasoning_effort": args.reasoning_effort,
            "generation_reasoning_budget": args.reasoning_budget,
        }
    else:
        generation_parameters = {
            # "generation_temperature": 0.7,
            "generation_temperature": None,
            "generation_max_new_tokens": args.max_new_tokens,
            # "generation_top_p": 0.95,
            "generation_top_p": None,
            "generation_batch_size": args.batch_size,
            "generation_reasoning_effort": args.reasoning_effort,
            "generation_reasoning_budget": 0,
        }

    # Get short model name from config for job naming
    model_aliases = get_model_aliases()
    short_model_name = model_aliases.get(model.name, model.name)
    batch_inference_job_name = (
        autoeval.name + "-" + short_model_name + "-" + args.run_specific_info
    )

    if args.disable_cache:
        logger.info("Running evaluation without caching results")
        evaluate_without_caching(
            autoeval,
            data,
            generation_parameters=generation_parameters,
            zero_shot=args.zero_shot,
            batch_inference_overwrite_existing_job=args.batch_inference_overwrite_existing_job,
            batch_inference_job_name=batch_inference_job_name,
        )
        return

    s3_client = None
    if args.use_s3_backup and args.s3_bucket_name and args.s3_bucket_cache_dir:
        session, _ = get_bedrock_session_and_config_sync(
            args.bedrock_assume_role,
            region=args.bedrock_region,
        )
        s3_client = session.client("s3")

    cache = MTEvaluationCache(
        str(cache_file),
        save_frequency=args.cache_save_frequency,
        s3_bucket_name=args.s3_bucket_name,
        s3_cache_key=args.s3_bucket_cache_dir + str(cache_file),
        s3_client=s3_client,
    )

    if args.sync_s3:
        print(
            "Syncing local cache and S3: First, cache on S3 will be merged into local. "
            "Then, The merged cache will be uploaded to S3, overwriting it."
        )

        result = cache.merge_from_s3()
        print(f"Merge cache from S3 result: {result}")

        while True:
            response = (
                input("\nDo you want to proceed with the upload? (y/n): ")
                .lower()
                .strip()
            )
            if response in ["y", "yes"]:
                break
            elif response in ["n", "no"]:
                print("Cache upload cancelled by user.")
                exit()
            else:
                print("Please enter 'y' for yes or 'n' for no.")

        result = cache.force_upload_to_s3()

        print(f"Upload cache to S3 result: {result}")
        exit()

    # Run evaluation
    logger.info("Running evaluation and caching results")
    evaluate_and_cache(
        autoeval,
        data,
        cache,
        save_frequency=args.cache_save_frequency,
        generation_parameters=generation_parameters,
        zero_shot=args.zero_shot,
        use_batch_inference=args.use_batch_inference,
        batch_inference_overwrite_existing_job=args.batch_inference_overwrite_existing_job,
        batch_inference_job_name=batch_inference_job_name,
        parse_with_autoeval_and_overwrite_cache=args.parse_with_autoeval_and_overwrite_cache,
    )


if __name__ == "__main__":
    main(parse_arguments())
