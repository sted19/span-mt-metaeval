# MT Evaluation Framework - Package Structure

This document describes the organization of the `mt_evaluation` package.

## Directory Structure

```
mt_evaluation/
├── __init__.py              # Package entry point, re-exports core classes
├── README.md                # This file
├── utils.py                 # General utilities (torch config, logging)
│
├── core/                    # Core data structures
│   ├── __init__.py          # Exports all core classes
│   ├── datastructures.py    # Sample, Error, Evaluation classes
│   ├── model_io.py          # FewShots, Response classes
│   └── constants.py         # Error types, severities, WMT LPs
│
├── autoevals/               # Automatic evaluation implementations
│   ├── __init__.py          # Re-exports core for backward compatibility
│   ├── autoeval.py          # Abstract base class for evaluators
│   ├── factory.py           # Registry and factory for evaluators
│   ├── gemba_mqm.py         # GEMBA-MQM evaluator
│   ├── utils.py             # JSON parsing utilities
│   ├── specialized/         # Task-specific evaluators
│   │   ├── base.py          # Specialized base class
│   │   ├── accuracy.py      # Accuracy evaluator
│   │   ├── fluency.py       # Fluency evaluator
│   │   └── ...
│   └── unified/             # Unified MQM evaluators
│       ├── base.py          # Unified base class
│       ├── unified_mqm.py   # Standard UnifiedMQM
│       ├── unified_mqm_boosted_v5.py  # Latest version
│       └── ...
│
├── models/                  # Language model implementations
│   ├── __init__.py          # Re-exports FewShots, Response
│   ├── base.py              # Abstract Model base class
│   ├── factory.py           # Model registry and factory
│   ├── bedrock/             # AWS Bedrock models
│   │   ├── base.py          # Bedrock base class
│   │   ├── claude.py        # Claude models
│   │   ├── llama.py         # Llama models
│   │   └── ...
│   └── huggingface/         # HuggingFace models
│       ├── gemma3.py
│       ├── llama3.py
│       └── qwen3.py
│
├── data/                    # Data handling utilities
│   ├── __init__.py          # Re-exports common utilities
│   ├── cache.py             # MTEvaluationCache class
│   ├── language_codes.py    # Language code mappings
│   └── utils.py             # WMT data loaders, misc utilities
│
├── meta_evaluation/         # Meta-evaluation tools
│   ├── __init__.py          # Result dataclasses, WMT LP constants
│   ├── utils.py             # Correlation computation, stats
│   ├── preprocessing_utils.py
│   └── span_level/          # Span-level evaluation
│       ├── utils.py         # P/R/F1 computation
│       └── perturbations.py # Perturbation experiments
│
└── config/                  # Configuration management
    ├── __init__.py          # YAML loading utilities
    └── metrics.yaml         # Metrics configuration
```

## Key Components

### Core Module (`core/`)
Contains the fundamental data structures used throughout the framework:
- **Sample**: Represents a translation pair with source, target, and language info
- **Error**: Represents a translation error with span, category, severity
- **Evaluation**: Base class for evaluation results
- **HumanEvaluation**: Human annotation results
- **AutomaticEvaluation**: LLM-based evaluation results
- **Prompt**: Template container for system/user prompts
- **FewShots**: Container for few-shot examples
- **Response**: Container for model responses

### Autoevals Module (`autoevals/`)
Contains automatic evaluation implementations:
- **AutoEval**: Abstract base class defining the evaluator interface
- **GembaMQM**: Implementation of the GEMBA-MQM evaluation approach
- **UnifiedMQM variants**: Various MQM-style evaluators
- **Specialized evaluators**: Task-specific evaluators (accuracy, fluency, etc.)

### Models Module (`models/`)
Contains language model wrappers:
- **Model**: Abstract base class for model implementations
- **BedrockModel**: AWS Bedrock model wrapper with async support
- **HuggingFace models**: Local model wrappers

### Data Module (`data/`)
Contains data handling utilities:
- **MTEvaluationCache**: Persistent caching with S3 backup support
- **Language code mappings**: ISO code to language name mappings
- **WMT data loaders**: Functions to load WMT evaluation data

### Meta-evaluation Module (`meta_evaluation/`)
Contains tools for evaluating metric quality:
- **Correlation computation**: Pearson, Spearman correlations with human scores
- **Span-level metrics**: Precision, recall, F1 for error span detection
- **Statistics**: Aggregation and reporting utilities

### Config Module (`config/`)
Contains configuration management:
- **YAML configuration**: Metrics definitions per test set
- **Configuration loaders**: Functions to load and parse config files

## Import Conventions

For new code, prefer importing from the canonical locations:

```python
# Core data structures
from mt_evaluation.core import Sample, Error, AutomaticEvaluation

# Evaluators
from mt_evaluation.autoevals.factory import get_autoeval

# Models
from mt_evaluation.models.factory import get_model

# Data utilities
from mt_evaluation.data import MTEvaluationCache
from mt_evaluation.data.language_codes import get_language_name

# Configuration
from mt_evaluation.config import get_metrics_to_evaluate
```
