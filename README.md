# ViT Fault Injection Framework

Framework for analyzing bit-flip fault injection vulnerabilities in Vision Transformers (ViT, DeiT, Swin, BEiT).

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ViT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

```bash
# 1. Generate fault-free logits (required once per model)
python -m src.main --model vit_base --mode faultfree --save_logits true

# 2. Run fault injection experiment
python -m src.main --model vit_base --mode faulty --repeat 100
```

## Supported Models

- `vit_base` - ViT-Base/16
- `vit_small` - ViT-Small/16
- `deit_tiny` - DeiT-Tiny
- `swin_small` - Swin-Small
- `beit_base` - BEiT-Base

## Command Line Options

| Option | Description |
|--------|-------------|
| `--model` | Model to evaluate (required) |
| `--mode` | `faultfree` or `faulty` |
| `--repeat` | Number of fault injection runs |
| `--component` | Target component: `mlp`, `attention`, `norm`, `patch_embed`, `classifier`, `all` |
| `--sub_component` | Sub-component: `qkv`/`proj` for attention, `fc1`/`fc2` for mlp |
| `--bit_range` | Bit range for fault injection (e.g., `0,31`) |
| `--batch_size` | Override batch size |
| `--max_batches` | Limit number of batches |

## Project Structure

```
ViT/
├── src/
│   ├── main.py              # Entry point + RunManager
│   ├── core/
│   │   ├── model.py         # ModelEvaluator (loading, inference)
│   │   └── data_manager.py  # Data validation, NaN handling
│   ├── metrics/
│   │   ├── base.py          # BaseMetric interface
│   │   ├── accuracy.py      # AccuracyMetrics class
│   │   └── sdc.py           # SDCMetrics class
│   ├── fault_injector/
│   │   └── fault_injection.py
│   ├── config/
│   │   └── settings.py
│   ├── data/
│   │   └── imagenet_loader.py
│   └── utils/
│       ├── helper.py        # Supported models
│       └── logits.py        # Fault-free logits cache
├── logits/                   # Fault-free logits (gitignored)
├── results/                  # Experiment results (gitignored)
└── scripts/                  # HPC scripts
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- ImageNet validation dataset
