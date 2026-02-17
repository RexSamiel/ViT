# ViT Fault Injection & Activation Analysis Framework

Framework for analyzing bit-flip fault injection vulnerabilities and activation distributions in Vision Transformers (ViT, DeiT, Swin, BEiT).

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ViT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### Fault Injection

```bash
# 1. Generate fault-free baseline (required once per model)
python -m src.main --model vit_base fi --condition faultfree --save_logits true

# 2. Run fault injection experiment
python -m src.main --model vit_base fi --condition faulty --repeat 100
```

### Activation Analysis

```bash
# Profile activation distributions across all layers
python -m src.main --model vit_base aa --sampling 1.0

# Use lower sampling for faster analysis on large datasets
python -m src.main --model vit_base --max_batches 500 aa --sampling 0.1
```

## Modes

The framework operates in two modes via subcommands:

### `fi` - Fault Injection

Inject single-bit faults into model parameters and measure accuracy degradation and Silent Data Corruption (SDC) metrics.

```bash
python -m src.main --model <model> fi [options]
```

| Option | Description |
|--------|-------------|
| `--condition` | `faultfree` (baseline) or `faulty` (inject faults) |
| `--repeat` | Number of fault injection runs (default: 1) |
| `--save_logits` | Save fault-free logits for SDC analysis (default: false) |
| `--component` | Target: `mlp`, `attention`, `norm`, `patch_embed`, `classifier`, `all` |
| `--sub_component` | Sub-component: `qkv`/`proj` for attention, `fc1`/`fc2` for mlp |
| `--block_idx` | Target specific transformer block index |
| `--bit_range` | Bit range (e.g., `0-7` for sign/exponent, `8-31` for mantissa) |
| `--info` | Show per-run info in multi-run mode (default: false) |

### `aa` - Activation Analysis

Profile activation value distributions across all model layers. Useful for understanding model behavior, identifying outlier layers, and informing fault injection strategies.

```bash
python -m src.main --model <model> aa [options]
```

| Option | Description |
|--------|-------------|
| `--sampling` | Percentage of activations to sample per layer (default: 1.0%, min: 0.01%) |

**Output includes:**
- Per-layer activation ranges (min/max values)
- Histogram distributions by component (input, output, block, mha, mlp)
- Global statistics and sampling ratios
- Results saved to `results/new_runs/activations_<model>_<samples>samples_<date>.json`

## Shared Options

These options apply to both modes:

| Option | Description |
|--------|-------------|
| `--model` | Model key to evaluate (required) |
| `--batch_size` | Batch size (int or 'None' for full batches) |
| `--max_batches` | Max batches (int or 'None' for all) |
| `--verbose` | Print verbose output (default: true) |
| `--seed` | Random seed for reproducibility |

## Supported Models

**Vision Transformer (ViT):**
- `vit_tiny`, `vit_small`, `vit_base`, `vit_large`, `vit_huge`

**DeiT (Data-efficient Image Transformers):**
- `deit_tiny`, `deit_small`, `deit_small_distilled`, `deit_base`, `deit_base_distilled`

**Swin Transformer:**
- `swin_tiny`, `swin_small`, `swin_base`, `swin_large`

**BEiT:**
- `beit_base`, `beit_large`

## Project Structure

```
ViT/
├── src/
│   ├── main.py                     # Entry point with mode routing
│   ├── config/
│   │   └── settings.py             # Configuration dataclass
│   └── core/
│       ├── model.py                # ModelRunner (loading, inference, batching)
│       ├── activation/
│       │   ├── manager.py          # ActivationAnalyzer + run/save functions
│       │   ├── hooks.py            # HookManager for forward hooks
│       │   └── histogram.py        # Histogram computation utilities
│       ├── fault_injection/
│       │   ├── manager.py          # FaultInjection engine + run/save functions
│       │   ├── injection.py        # Injector (bit-flip injection/restoration)
│       │   ├── accuracy.py         # AccuracyTracker
│       │   └── sdc.py              # SDCTracker
│       └── library/
│           ├── ui.py               # SUPPORTED_MODELS registry, formatting
│           ├── layers.py           # Layer identification utilities
│           ├── logits.py           # Fault-free logits cache
│           ├── imagenet_loader.py  # ImageNet data loading
│           └── utils.py            # Shared utilities
├── logits/                         # Fault-free logits cache (gitignored)
├── results/                        # Experiment results (gitignored)
└── scripts/                        # HPC/batch scripts
```

## Examples

```bash
# Fault injection with specific component targeting
python -m src.main --model vit_base fi --condition faulty --repeat 50 --component mlp --sub_component fc1

# Target specific block and bit range
python -m src.main --model deit_small fi --condition faulty --repeat 100 --block_idx 6 --bit_range 0-7

# Activation analysis with 10% sampling on 500 batches
python -m src.main --model swin_base --max_batches 500 aa --sampling 10.0

# Quick activation profiling (100 samples, 0.1% sampling)
python -m src.main --model vit_tiny --max_batches 1 aa --sampling 0.1
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- timm (PyTorch Image Models)
- ImageNet validation dataset
