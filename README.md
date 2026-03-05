# ViT Fault Injection and Detection Framework

A framework for fault injection experiments and detection on Vision Transformers.

## Project Structure

```
ViT/
├── src/
│   └── vit_fault/          # Main package
│       ├── core/           # Model loading, layers, bit operations
│       ├── detection/      # Fault detection (NeuroChecker)
│       ├── injection/      # Bit-flip fault injection
│       ├── eval/           # Accuracy and SDC metrics
│       ├── analysis/       # Activation/weight analysis
│       └── cli.py          # Command-line interface
├── tests/                  # Pytest tests
├── results/                # Experiment results and plotting
│   ├── data/               # JSON result files
│   └── graphing/           # Plotting tools
├── data/                   # Runtime data
│   ├── weights/            # Precomputed checker weights
│   └── logits/             # Fault-free logits for SDC
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### Python API

```python
from vit_fault import Model, Detector, Injector, evaluate

# Load model
model = Model("vit_tiny")

# Save baseline (required once for SDC metrics)
model.save_baseline()

# Setup detection and injection
detector = Detector(model, layers="fc1", threshold=0.1)
injector = Injector(model, layers="fc1")

# Run experiment
injector.inject(count=1)
results = evaluate(model, detector)
results.print()

# See detection details
detector.print_results()

# Restore original weights
injector.restore()
```

### Command Line

```bash
# Save fault-free logits (required for SDC)
python -m vit_fault.cli -m vit_tiny --baseline

# Run fault injection with detection
python -m vit_fault.cli -m vit_tiny --detect fc1 --inject fc1 --faults 1

# Multiple runs with JSON output
python -m vit_fault.cli -m vit_tiny --detect fc1 --inject fc1 \
    --faults 1 --repeat 10 -o results/my_experiment.json
```

### Analysis

```python
from vit_fault import Model
from vit_fault.analysis import ActivationAnalyzer, WeightAnalyzer

model = Model("vit_tiny")

# Activation analysis
analyzer = ActivationAnalyzer(model)
analyzer.run(num_batches=10)
analyzer.save("results/activations_vit_tiny.json")

# Weight analysis
weight_analyzer = WeightAnalyzer(model)
weight_analyzer.run()
weight_analyzer.save("results/weights_vit_tiny.json")
```

### Plotting

```bash
# Plot fault injection results
python results/graphing/plot.py results/data/*.json

# Plot activation/weight analysis
python results/graphing/plot_activations.py results/activations_vit_tiny.json
```

## Supported Models

- ViT: `vit_tiny`, `vit_small`, `vit_base`, `vit_large`
- DeiT: `deit_tiny`, `deit_small`, `deit_base`
- Swin: `swin_tiny`, `swin_small`, `swin_base`
- BEiT: `beit_base`

## Detection Layers

- `all` - All linear layers
- `fc1` - MLP first projection
- `fc2` - MLP second projection
- `qkv` - Attention QKV projection
- `proj` - Attention output projection

## Running Tests

```bash
pytest tests/ -v
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- timm (PyTorch Image Models)
- ImageNet validation dataset
