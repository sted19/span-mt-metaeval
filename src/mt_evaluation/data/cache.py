# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Caching system for MT evaluation results.

This module provides a persistent caching system for evaluation results to avoid
re-evaluating the same samples. It supports atomic writes, periodic saving,
and optional S3 backup synchronization.
"""

import json
import logging
import os
import shutil
import tempfile
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

from mt_evaluation.core import AutomaticEvaluation, Sample

logger = logging.getLogger(__name__)


class MTEvaluationCache:
    """
    Cache system for MT evaluation results.

    This class provides persistent caching of evaluation results to avoid
    re-evaluating the same samples. It supports atomic writes and periodic
    saving to prevent data loss.

    The cache prioritizes local cache over S3. If data is in the local cache,
    it will be used. Otherwise, the cache tries to download data from S3.
    Therefore, when new evaluations are conducted, the priority will be given
    to local cache and new evaluations. If the same evaluations were already
    present on S3, this will not be taken into consideration. Worse, S3 will
    be overwritten with the new evaluations.

    Finally, if merging from S3, only the evaluations that are NOT already
    locally will be downloaded. Then, local evaluation + S3 (different)
    evaluations will be uploaded to S3, replacing its content.

    Attributes:
        cache_file: Path to the cache file.
        save_frequency: Number of evaluations to accumulate before saving.
        cache: In-memory cache of evaluations.
        pending_evaluations: List of evaluations waiting to be saved.
        evaluation_count: Total number of evaluations processed.
    """

    def __init__(
        self,
        cache_file: str,
        save_frequency: int = 10,
        s3_bucket_name: Optional[str] = None,
        s3_cache_key: Optional[str] = None,
        s3_client: Optional[boto3.client] = None,
    ):
        """
        Initialize the cache system.

        Args:
            cache_file: Path to the cache file.
            save_frequency: Number of evaluations to accumulate before saving.
            s3_bucket_name: Optional S3 bucket name for backup.
            s3_cache_key: Optional S3 key for the cache file.
            s3_client: Optional boto3 S3 client.
        """
        self.cache_file = cache_file
        self.save_frequency = save_frequency
        self.cache: Dict[str, AutomaticEvaluation] = {}
        self.pending_evaluations: List[Tuple[str, Sample]] = []
        self.evaluation_count = 0

        # S3 configuration
        self.s3_bucket_name = s3_bucket_name
        self.s3_cache_key = s3_cache_key
        self.s3_client = s3_client
        self.s3_enabled = bool(s3_bucket_name and s3_cache_key and s3_client)

        # Load existing cache
        self._load_cache()

    def _load_cache_from_file(self, file_path: str, merge_mode: bool = False) -> int:
        """
        Load cache data from a file.

        Args:
            file_path: Path to the cache file to load from.
            merge_mode: If True, only add entries that don't already exist in cache.
                       If False, add all entries (used for initial loading).

        Returns:
            int: Number of entries loaded/added.
        """
        added_count = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        if data.get("evaluation"):
                            sample = Sample.from_dict(data)
                            cache_key = sample.get_input_hash()

                            # In merge mode, only add if not already in cache
                            # In normal mode, add all entries
                            if not merge_mode or cache_key not in self.cache:
                                self.cache[cache_key] = sample.evaluation
                                added_count += 1
                        else:
                            logger.warning(
                                f"Skipping line {line_num} from {'merge' if merge_mode else 'cache'} file: missing 'evaluation' key"
                            )
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping malformed line {line_num} from {'merge' if merge_mode else 'cache'} file: {e}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error processing line {line_num} from {'merge' if merge_mode else 'cache'} file: {e}"
                        )

        except Exception as e:
            logger.error(f"Error loading cache from file {file_path}: {e}")

        if not merge_mode:
            logger.info(f"Loaded {added_count} cached evaluations")

        return added_count

    def _download_and_merge_s3_cache(self) -> bool:
        """
        Download cache file from S3 and merge it with existing local cache.

        Returns:
            bool: True if download and merge was successful, False otherwise.
        """
        if not self.s3_enabled:
            return False

        try:
            logger.info(
                f"Attempting to download cache from S3: s3://{self.s3_bucket_name}/{self.s3_cache_key}"
            )

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

            # Download to a temporary file first for atomic operation
            temp_fd, temp_file = tempfile.mkstemp(
                dir=os.path.dirname(self.cache_file) or ".", suffix=".tmp"
            )

            try:
                with os.fdopen(temp_fd, "wb") as f:
                    self.s3_client.download_fileobj(
                        self.s3_bucket_name, self.s3_cache_key, f
                    )

                # Merge S3 cache with existing local cache
                s3_cache_count = self._load_cache_from_file(temp_file, merge_mode=True)
                logger.info(f"Merged {s3_cache_count} new evaluations from S3 cache")

                return True

            finally:
                # Cleanup temp file if it still exists
                if os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass

        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.info("No cache file found in S3")
            else:
                logger.warning(f"Failed to download cache from S3: {e}")
            return False
        except Exception as e:
            logger.warning(f"Error downloading cache from S3: {e}")
            return False

    def _upload_cache_to_s3(self) -> bool:
        """
        Upload cache file from local disk to S3.

        Returns:
            bool: True if upload was successful, False otherwise.
        """
        if not self.s3_enabled:
            return False

        try:
            logger.info(
                f"Uploading cache to S3: s3://{self.s3_bucket_name}/{self.s3_cache_key}"
            )

            with open(self.cache_file, "rb") as f:
                self.s3_client.upload_fileobj(
                    f,
                    self.s3_bucket_name,
                    self.s3_cache_key,
                    ExtraArgs={"ContentType": "application/json"},
                )

            logger.info("Successfully uploaded cache to S3")
            return True

        except Exception as e:
            logger.warning(f"Failed to upload cache to S3: {e}")
            return False

    def _load_cache(self) -> None:
        """Load existing evaluations from local disk, then merge from S3 if enabled."""
        # First try to load from local disk
        if os.path.exists(self.cache_file):
            logger.info(f"Loading cache from local file: {self.cache_file}")
            self._load_cache_from_file(self.cache_file, merge_mode=False)
        else:
            logger.info(f"No local cache file found: {self.cache_file}")

        # Then try to merge additional entries from S3
        if self.s3_enabled:
            logger.info("Attempting to merge cache from S3")
            if not self._download_and_merge_s3_cache():
                logger.info("No additional cache found in S3")
        else:
            logger.info("S3 sync is disabled")

    def is_evaluated(self, sample: Sample) -> bool:
        """
        Check if a sample has already been evaluated.

        Args:
            sample: The sample to check.

        Returns:
            bool: True if the sample has been evaluated, False otherwise.
        """
        cache_key = sample.get_input_hash()
        return cache_key in self.cache

    def get_evaluation(self, sample: Sample) -> Optional[AutomaticEvaluation]:
        """
        Get cached evaluation for a sample.

        Args:
            sample: The sample to get evaluation for.

        Returns:
            Optional[AutomaticEvaluation]: The cached evaluation, or None if not found.
        """
        cache_key = sample.get_input_hash()
        return self.cache.get(cache_key)

    def add_evaluation(self, sample: Sample, evaluation: AutomaticEvaluation) -> None:
        """
        Add a new evaluation to the cache.

        The evaluation will be saved periodically based on save_frequency.

        Args:
            sample: The sample that was evaluated.
            evaluation: The evaluation result.
        """
        cache_key = sample.get_input_hash()
        self.cache[cache_key] = evaluation
        sample.evaluation = evaluation

        # Store for batch writing
        self.pending_evaluations.append((cache_key, sample))
        self.evaluation_count += 1

        # Save periodically
        if len(self.pending_evaluations) >= self.save_frequency:
            self._save_pending_evaluations()

    def _save_pending_evaluations(self) -> None:
        """Atomically append pending evaluations to cache file."""
        if not self.pending_evaluations:
            return

        # Use atomic write to prevent corruption
        temp_fd, temp_file = None, None
        try:
            # Create temporary file in same directory
            cache_dir = os.path.dirname(self.cache_file) or "."
            temp_fd, temp_file = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")

            # Copy existing file content if it exists
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as src:
                    with os.fdopen(temp_fd, "w", encoding="utf-8") as dst:
                        dst.write(src.read())
                        temp_fd = None  # File descriptor is now managed by dst
            else:
                os.close(temp_fd)
                temp_fd = None

            # Append new evaluations
            with open(temp_file, "a", encoding="utf-8") as f:
                for _, sample in self.pending_evaluations:
                    json.dump(sample.to_dict(), f, ensure_ascii=True)
                    f.write("\n")

            # Atomic move
            shutil.move(temp_file, self.cache_file)
            temp_file = None

            logger.info(f"Saved {len(self.pending_evaluations)} evaluations to cache")

            self.pending_evaluations.clear()

        except Exception as e:
            logger.error(f"Error saving evaluations: {e}")
            # Cleanup on error
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
            raise

    def finalize(self) -> None:
        """
        Save any remaining pending evaluations and finalize the cache.

        This method should be called when evaluation is complete to ensure
        all evaluations are saved to disk.
        """
        if self.pending_evaluations:
            self._save_pending_evaluations()
        logger.info(f"Total evaluations processed: {self.evaluation_count}")

        # Final sync to S3 if enabled
        if self.s3_enabled and os.path.exists(self.cache_file):
            self._upload_cache_to_s3()

    def merge_from_s3(self) -> bool:
        """
        Safely merge cache entries from S3 without overwriting local cache.
        Only adds entries that don't already exist locally.

        Returns:
            bool: True if operation succeeded, False if it failed.
        """
        if not self.s3_enabled:
            logger.warning("S3 sync is not enabled")
            return False

        return self._download_and_merge_s3_cache()

    def overwrite_cache(self, samples: List[Sample]) -> None:
        """
        Overwrite the cache on disk with the provided samples.

        This is useful when you need to clean up the cache file or when the
        cache file has become corrupted and you want to regenerate it.

        Args:
            samples: List of samples to write to the cache.
        """
        if not samples:
            return

        # Use atomic write to prevent corruption
        temp_fd, temp_file = None, None
        try:
            # Create temporary file in same directory
            cache_dir = os.path.dirname(self.cache_file) or "."
            temp_fd, temp_file = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")

            # Append new evaluations
            with open(temp_file, "w", encoding="utf-8") as f:
                for sample in samples:
                    json.dump(sample.to_dict(), f, ensure_ascii=False)
                    f.write("\n")

            # Atomic move
            shutil.move(temp_file, self.cache_file)
            temp_file = None

            logger.info(f"Saved {len(samples)} evaluations to cache")

        except Exception as e:
            logger.error(f"Error saving evaluations: {e}")
            # Cleanup on error
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except OSError:
                    pass
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
            raise

    def force_upload_to_s3(self) -> bool:
        """
        Force an immediate upload of the current cache file to S3.

        Returns:
            bool: True if upload was successful, False otherwise.
        """
        # Save any pending evaluations first
        if self.pending_evaluations:
            self._save_pending_evaluations()

        return self._upload_cache_to_s3()

    def __len__(self) -> int:
        """Return the number of cached evaluations."""
        return len(self.cache)
