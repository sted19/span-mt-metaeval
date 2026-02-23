# Span-Level Machine Translation Meta-Evaluation

A toolkit for evaluating the quality of automatic machine translation metrics using WMT Metrics Shared Task benchmarks. This framework enables researchers to run LLM-based evaluation of translations and measure how well these automatic evaluators correlate with human MQM annotations.

## Overview

The MT Evaluation Framework supports two main research workflows:

1. **Automatic Evaluation**: Use LLMs (via AWS Bedrock) to evaluate translation quality by identifying errors (spans, categories, severities) following MQM guidelines
2. **Meta-Evaluation**: Measure how well your automatic evaluator's error detection aligns with human MQM annotations from WMT benchmarks

### Supported Benchmarks

- **WMT22**: en-de, zh-en language pairs
- **WMT23**: en-de, zh-en language pairs  
- **WMT24**: en-de, en-es, ja-zh language pairs
- **WMT25**: en-ko, ja-zh language pairs (MQM data)


## ⚠️ Security & Usage Disclaimer

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

This framework is intended **solely for personal research and experimentation purposes** 
using **non-sensitive, public data** in isolated, trusted environments.

**This package does NOT implement the security controls required for:**
- Processing confidential, proprietary, or personally identifiable data
- Production or customer-facing environments
- Multi-tenant or shared environments requiring enterprise-grade security controls

Users are responsible for managing their own AWS credentials, infrastructure security, 
and data handling practices. By using this package, you acknowledge that:

- Cache files (local and S3) are stored without encryption or integrity verification.
- Logs may contain information about your AWS resources and evaluation content.
- No cost-control mechanisms are enforced, monitor your AWS usage accordingly.
- Input validation is minimal; do not use untrusted or adversarial inputs.

Any use beyond personal experimentation with public data would require significant 
security enhancements. If you have questions about appropriate use, consult your security team.

## Installation
Create a conda environment (Optional, but recommended) using Python >= 3.12 (< 3.14):
```bash
conda create -n mt-evaluation "python=3.12"
```

Then, install the [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval) package and download its data:
```bash
git clone https://github.com/google-research/mt-metrics-eval.git
cd mt-metrics-eval
pip install .

alias mtme='python3 -m mt_metrics_eval.mtme'
mtme --download  # Puts ~2G of data into $HOME/.mt-metrics-eval.
```

Finally, clone the repository and install the mt_evaluation package: 
```bash
# Clone the repository
git clone <repository-url>
cd MTEvaluationFramework

# Install in development mode
pip install -e .
```

### Requirements

- Python ≥ 3.12
- AWS credentials configured for Bedrock access (see [AWS Configuration](#aws-configuration))
- The `mt-metrics-eval` package (installed automatically) for loading WMT benchmark data

## Quick Start

### 1. Run LLM Evaluation on WMT Data

Evaluate translations from WMT23 using Claude 3.5 Haiku:

```bash
python scripts/evaluate.py \
    --model-id us.anthropic.claude-3-5-haiku-20241022-v1:0 \
    --evaluation-schema unified-mqm-boosted-v5 \
    --test-set wmt23 \
    --lps en-de zh-en
```

This will:
- Load WMT23 translation samples with human MQM annotations
- Send each source/translation pair to the LLM for error annotation
- Cache results in `outputs/` to avoid re-evaluation
- Print a summary of evaluation costs and timing

### 2. Span-Level Meta-Evaluation

Measure precision/recall/F1 of error span detection against human annotations:

```bash
python scripts/meta_evaluate.py \
    --test-sets wmt23 \
    --lps en-de zh-en
```

Output includes:
- Precision: What fraction of predicted error spans match human annotations
- Recall: What fraction of human error spans were detected
- F1: Harmonic mean of precision and recall
- Statistics broken down by language pair and globally

### 3. Score-Level Meta-Evaluation

Compute correlation between automatic scores and human MQM scores (legacy, might not work without adjustments):

```bash
python scripts/meta_evaluate_scores.py \
    --lps en-de en-es ja-zh
```

Output includes:
- PCE (Pairwise Confidence Error)
- KWT (Kendall's Tau with Ties)
- Pearson correlation

## Workflows

### Workflow 1: Evaluating Translations

The `evaluate.py` script runs LLM-based evaluation on WMT benchmark data.

```bash
python scripts/evaluate.py \
    --model-id <model-id> \
    --evaluation-schema <schema> \
    --test-set <wmt22|wmt23|wmt24|wmt25> \
    --lps <language-pairs>
```

**Key options:**

| Option | Description                                                                  |
|--------|------------------------------------------------------------------------------|
| `--model-id` | Model identifier (see [Available Models](#available-models))                 |
| `--evaluation-schema` | Evaluation prompt schema (see [Available Evaluators](#available-evaluators)) |
| `--test-set` | WMT test set year                                                            |
| `--lps` | Space-separated language pairs (e.g., `en-de zh-en`)                         |
| `--cache-dir` | Directory for caching results (default: `outputs`)                           |
| `--zero-shot` | Run without few-shot examples (default: `true`)                              |
| `--toy N` | Only evaluate first N samples (for testing)                                  |
| `--bedrock-max-concurrent` | Maximum number of concurrent async API calls (default: `5`)                  |

**Batch inference options:**

| Option | Description |
|--------|-------------|
| `--use-batch-inference` | Use AWS Bedrock batch inference (cost-effective for large runs) |
| `--s3-bucket-name` | S3 bucket for batch inference I/O |
| `--use-s3-backup` | Sync cache to S3 for persistence |

**Example with batch inference:**

```bash
python scripts/evaluate.py \
    --model-id global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --evaluation-schema unified-mqm-boosted-v5 \
    --test-set wmt25 \
    --lps en-ko ja-zh \
    --use-batch-inference \
    --s3-bucket-name my-bucket \
    --run-specific-info wmt25-submission
```

### Workflow 2: Span-Level Meta-Evaluation

The `meta_evaluate.py` script computes precision/recall/F1 for error span detection.

```bash
python scripts/meta_evaluate.py \
    --test-sets <test-sets> \
    --lps <language-pairs>
```

**Key options:**

| Option | Description |
|--------|-------------|
| `--test-sets` | One or more WMT test sets (e.g., `wmt23 wmt24`) |
| `--lps` | Language pairs to evaluate |
| `--human-severities` | Filter human errors by severity (default: `minor major critical`) |
| `--auto-severities` | Filter automatic errors by severity |
| `--human-categories` | Filter human errors by category (default: `All`) |
| `--severity-penalty` | Penalty (0-1) for severity mismatches in matching |
| `--remove-overlapping-errors` | Remove overlapping error spans |
| `--do-not-verify-completeness` | Allow partial evaluation results |

**Example with severity filtering:**

```bash
python scripts/meta_evaluate.py \
    --test-sets wmt24 \
    --lps en-de en-es ja-zh \
    --human-severities major critical \
    --auto-severities major critical
```

### Workflow 3: Score-Level Meta-Evaluation

The `meta_evaluate_scores.py` script computes correlation with human MQM scores (legacy, it might need adjustments).

```bash
python scripts/meta_evaluate_scores.py \
    --lps en-de en-es ja-zh
```

**Key options:**

| Option | Description |
|--------|-------------|
| `--lps` | Language pairs to evaluate |
| `--exclude-human` | Exclude human/reference translations from evaluation |

## Available Evaluators

| Schema | Description |
|--------|-------------|
| `unified-mqm-boosted-v5` | **Recommended.** Detailed step-by-step MQM evaluation with chain-of-thought |
| `unified-mqm-boosted-doc-context` | Document-level context-aware evaluation |
| `unified-mqm-critical` | Standard unified MQM evaluation |
| `unified-simple` | Simplified unified evaluation |
| `unified-simplest` | Minimal unified evaluation |
| `gemba-mqm` | GEMBA-MQM baseline implementation |

## Available Models

### AWS Bedrock Models

| Model ID | Description |
|----------|-------------|
| `us.anthropic.claude-3-5-haiku-20241022-v1:0` | Claude 3.5 Haiku (fast, cost-effective) |
| `us.anthropic.claude-3-7-sonnet-20250219-v1:0` | Claude 3.7 Sonnet (with extended thinking) |
| `global.anthropic.claude-haiku-4-5-20251001-v1:0` | Claude 4.5 Haiku |
| `global.anthropic.claude-sonnet-4-5-20250929-v1:0` | Claude 4.5 Sonnet |
| `us.meta.llama3-1-8b-instruct-v1:0` | Llama 3.1 8B |
| `us.meta.llama3-3-70b-instruct-v1:0` | Llama 3.3 70B |
| `us.meta.llama4-scout-17b-instruct-v1:0` | Llama 4 Scout 17B |
| `us.meta.llama4-maverick-17b-instruct-v1:0` | Llama 4 Maverick 17B |
| `amazon.nova-pro-v1:0` | Amazon Nova Pro |
| `qwen.qwen3-235b-a22b-2507-v1:0` | Qwen 3 235B |
| `qwen.qwen3-32b-v1:0` | Qwen 3 32B |
| `openai.gpt-oss-20b-1:0` | GPT OSS 20B |
| `openai.gpt-oss-120b-1:0` | GPT OSS 120B |

### HuggingFace Models (Local)

Legacy; local models might not work and need updates to the codebase:

| Model ID | Description |
|----------|-------------|
| `google/gemma-3-1b-it` | Gemma 3 1B |
| `google/gemma-3-4b-it` | Gemma 3 4B |
| `google/gemma-3-12b-it` | Gemma 3 12B |
| `google/gemma-3-27b-it` | Gemma 3 27B |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Llama 3 8B |
| `Qwen/Qwen3-8B` | Qwen 3 8B |

## AWS Configuration

### Required Permissions

To use AWS Bedrock models, you need:
1. AWS credentials configured (`~/.aws/credentials` or environment variables)
2. Access to Bedrock models in your AWS account
3. (Optional) S3 bucket access for batch inference and cache backup

### Environment Variables

Set these environment variables to avoid passing them on every command:

```bash
# S3 bucket for batch inference and cache
export MT_EVAL_S3_BUCKET="your-bucket-name"

# IAM role to assume for Bedrock access (optional)
export MT_EVAL_ASSUME_ROLE="arn:aws:iam::123456789:role/BedrockRole"

# IAM role for Bedrock batch inference jobs (optional)
export MT_EVAL_BATCH_ROLE_ARN="arn:aws:iam::123456789:role/BedrockBatchRole"
```

### Batch Inference

For large-scale evaluations, batch inference is more cost-effective:

```bash
python scripts/evaluate.py \
    --model-id global.anthropic.claude-sonnet-4-5-20250929-v1:0 \
    --evaluation-schema unified-mqm-boosted-v5 \
    --test-set wmt25 \
    --use-batch-inference \
    --s3-bucket-name my-bucket \
    --bedrock-batch-job-role-arn arn:aws:iam::123456789:role/BedrockBatchRole
```

The script will:
1. Upload input data to S3
2. Create a Bedrock batch inference job
3. Wait for completion (or exit if still in progress)
4. Download and cache results

## Caching

Evaluation results are cached to avoid redundant API calls:

- **Local cache**: Stored in `outputs/<evaluator>/<model>/<run-info>/cache.jsonl`
- **S3 backup**: Use `--use-s3-backup` to sync cache to S3

### Managing the Cache

```bash
# Delete local cache for a specific run
python scripts/evaluate.py \
    --model-id <model> \
    --evaluation-schema <schema> \
    --test-set <test-set> \
    --delete-cache

# Sync local cache to S3
python scripts/evaluate.py \
    --model-id <model> \
    --evaluation-schema <schema> \
    --test-set <test-set> \
    --sync-s3 \
    --use-s3-backup \
    --s3-bucket-name my-bucket
```

## Additional Scripts

### Analyze Annotations

Analyze cached evaluation results for parsing errors and malformed annotations:

```bash
python scripts/analyze_annotations.py \
    --outputs-dir outputs \
    --autoeval-name unified-mqm-boosted-v5 \
    --model-name global.anthropic.claude-sonnet-4-5-20250929-v1:0
```

Outputs include:
- Samples with parsing errors (JSON couldn't be parsed)
- Malformed errors (empty spans, unknown severities)
- Per-language pair statistics
- Logs saved to `analysis/` directory

### Visualize Annotations

Interactive GUI browser for comparing model annotations with human MQM annotations:

```bash
python scripts/visualize_annotations.py \
    outputs/unified-mqm-boosted-v5/<model>/<run>/cache.jsonl \
    --test-set wmt24 \
    --lps en-de en-es ja-zh
```

Features:
- Side-by-side comparison of model vs human errors
- Filter by severity and category
- Navigate through samples interactively

## Project Structure

```
MTEvaluationFramework/
├── src/mt_evaluation/
│   ├── core/                    # Core data structures
│   │   ├── datastructures.py    # Sample, Error, Evaluation classes
│   │   ├── constants.py         # WMT language pairs, severity constants
│   │   ├── model_io.py          # Response, FewShots classes
│   │   ├── scoring.py           # MQM score computation
│   │   └── types.py             # Type definitions
│   │
│   ├── autoevals/               # Automatic evaluators
│   │   ├── autoeval.py          # Base AutoEval class
│   │   ├── factory.py           # get_autoeval() factory
│   │   ├── gemba_mqm.py         # GEMBA-MQM implementation
│   │   ├── utils.py             # JSON parsing utilities
│   │   ├── unified/             # Unified MQM evaluators
│   │   │   ├── base.py          # Base Unified class
│   │   │   ├── unified_mqm_boosted_v5.py  # Recommended evaluator
│   │   │   └── ...
│   │   └── specialized/         # Category-specific evaluators
│   │       ├── base.py          # Base Specialized class
│   │       ├── accuracy.py      # Accuracy errors
│   │       ├── fluency.py       # Fluency errors
│   │       └── ...
│   │
│   ├── models/                  # Model backends
│   │   ├── base.py              # Abstract Model class
│   │   ├── factory.py           # get_model() factory
│   │   ├── bedrock/             # AWS Bedrock implementations
│   │   │   ├── base.py          # BedrockModel base class
│   │   │   ├── claude.py        # Claude models
│   │   │   ├── llama.py         # Llama models
│   │   │   ├── nova.py          # Amazon Nova
│   │   │   └── ...
│   │   └── huggingface/         # Local HuggingFace models
│   │       ├── gemma3.py
│   │       ├── llama3.py
│   │       └── qwen3.py
│   │
│   ├── data/                    # Data handling
│   │   ├── cache.py             # MTEvaluationCache class
│   │   ├── wmt_loaders.py       # WMT data loading functions
│   │   ├── language_codes.py    # Language code mappings
│   │   └── utils.py             # Data utilities
│   │
│   ├── meta_evaluation/         # Meta-evaluation
│   │   ├── utils.py             # Score-level correlation
│   │   ├── metrics_to_evaluate.py  # Metric configurations
│   │   └── span_level/          # Span-level meta-evaluation
│   │       ├── matching.py      # Bipartite error matching
│   │       ├── metrics.py       # P/R/F1 computation
│   │       ├── preprocessing.py # Error filtering/normalization
│   │       └── utils.py         # Aggregation utilities
│   │
│   ├── config/                  # Configuration files
│   │   └── model_aliases.yaml   # Model name aliases
│   │
│   └── utils/                   # General utilities
│       ├── naming.py            # Name standardization
│       └── string.py            # String utilities
│
├── scripts/                     # CLI scripts
│   ├── evaluate.py              # Run LLM evaluation
│   ├── meta_evaluate.py         # Span-level meta-evaluation
│   ├── meta_evaluate_scores.py  # Score-level meta-evaluation
│   ├── analyze_annotations.py   # Analyze annotation statistics
│   └── visualize_annotations.py # Visualize error annotations
│
├── data/                        # WMT data files
│   ├── wmt22/                   # WMT22 MQM annotations
│   ├── wmt23/                   # WMT23 MQM annotations
│   └── wmt25/                   # WMT25 MQM/ESA annotations
│
├── tests/                       # Test suite
│   ├── test_smoke.py            # Smoke tests
│   └── test_overlapping_spans.py
│
└── outputs/                     # Cached evaluation results
```

## License
This work is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).