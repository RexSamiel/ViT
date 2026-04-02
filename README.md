# ViT Fault Injection and Detection Framework

A modular framework for fault injection experiments, analysis and detection on Vision Transformers.

## Features

- **Fault Injection**: Bit-flip fault injection into model weights with configurable bit ranges
- **Multiple Detection Methods**: Neuro checker, ABFT checksums, softmax-mean checksums
- **Plugin Architecture**: Add new detection methods without modifying core code
- **Parameter Analysis**: Analyze weight distributions and activation ranges across model layers
- **Evaluation Metrics**: Top-1/Top-5 accuracy, SDC (Silent Data Corruption) rates
- **CLI and API**: Use from command line or as a Python library

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd ViT

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from core.model import Model
from detection import create_detector
from injection import Injector
from eval import evaluate

# Load model
model = Model("vit_tiny")

# Setup detection and injection
detector = create_detector(model, method="neuro", layers="fc1")
injector = Injector(model, layers="fc1")

# Inject fault and evaluate
injector.inject(count=1)
results = evaluate(model, detector)

# Print results
results.print()
detector.print_results()

# Restore original weights
injector.restore()
detector.remove()
```

### CLI

```bash
# Basic evaluation
python -m cli -m vit_tiny

# Fault injection with detection
python -m cli -m vit_tiny hr --detect fc1 --inject fc1 --method neuro

# Multiple runs with output
python -m cli -m vit_tiny hr --detect fc1 --inject fc1 --repeat 10 -o results.json
```

## API Reference

### Detection

The detection system uses a plugin architecture. Methods self-register and are automatically available.

```python
from detection import create_detector, list_methods, Detector

# See available methods
print(list_methods())  # ['neuro', 'checksum', 'softmean']

# Create detector (recommended)
detector = create_detector(
    model,
    method="neuro",      # Detection method
    layers="fc1",        # Layer filter
    threshold=0.1,       # Detection threshold
)

# Or use backwards-compatible Detector class
detector = Detector(model, method="checksum", layers="fc1", threshold=1e-3)

# After running inference...
faults = detector.faults_found      # List of faulty layer names
results = detector.check()          # Dict: layer_name -> bool
detector.print_results()            # Print detailed results
detector.remove()                   # Restore original layers
```

#### Detection Methods

| Method | Target | Description |
|--------|--------|-------------|
| `neuro` | Linear layers | Extra neuron checker - compares checker output vs mean of regular outputs |
| `checksum` | Linear layers | Classical ABFT row/column checksums |
| `softmean` | Attention modules | Softmax-mean checksums for Q/K matrices |

### Injection

```python
from injection import Injector

injector = Injector(
    model,
    layers="fc1",              # Layer filter
    bit_range=(0, 31),         # Optional: restrict bit positions
)

# Inject faults
injector.inject(count=1)       # Inject N random bit-flips
injector.print_info()          # Show injection details

# Get fault info
info = injector.get_info()     # List of fault dictionaries
print(injector.count)          # Number of active faults

# Restore
injector.restore()             # Restore all original values
```

### Model

```python
from core.model import Model, ModelConfig

# With default config
model = Model("vit_tiny")

# With custom config
config = ModelConfig(
    batch_size=100,
    max_batches=10,
    data_root="/path/to/imagenet",
    use_train=False,           # Use validation set
)
model = Model("vit_base", config=config)

# Access internals
net = model.net                # The actual nn.Module
batches = model.get_batches()  # Cached data batches
```

### Evaluation

```python
from eval import evaluate, Results

# Basic evaluation
results = evaluate(model)

# With detection
results = evaluate(model, detector)

# Access metrics
print(f"Top-1: {results.top1}%")
print(f"Top-5: {results.top5}%")
print(f"SDC Rate: {results.sdc_rate}%")
print(f"Faults Detected: {results.faults_detected}")

results.print()  # Formatted output
```

### Analysis

Analyze model activations and weight distributions:

```python
from analysis import ActivationAnalyzer, WeightAnalyzer

# Activation analysis - captures min/max ranges and distributions during inference
analyzer = ActivationAnalyzer(model, include_histogram=True)
results = analyzer.run(num_batches=10)
analyzer.save("results/activations_vit_tiny.json")
analyzer.remove()  # Remove hooks

# Weight analysis - analyzes parameter distributions
analyzer = WeightAnalyzer(model)
results = analyzer.run()
analyzer.save("results/weights_vit_tiny.json")
```

#### ActivationAnalyzer

Hooks into model layers during inference to capture activation statistics.

```python
from analysis import ActivationAnalyzer

analyzer = ActivationAnalyzer(
    model,
    include_histogram=True,   # Build distribution histograms
    histogram_bins=1000,      # Number of histogram bins
)

# Run on multiple batches
results = analyzer.run(num_batches=10)

# Results structure:
# {
#   "layers": {layer_idx: {name, min, max, component, op_type, shape}},
#   "block_aggregated": {block_idx: {mha: {min, max}, mlp: {min, max}}},
#   "distributions": {component: {bin_centers, counts, data_range}},
#   "statistics": {total_layers, num_blocks}
# }

# Access layer data
for idx, layer in results["layers"].items():
    print(f"{layer['name']}: [{layer['min']:.4f}, {layer['max']:.4f}]")

# Save to file
analyzer.save("activations.json")

# Clean up hooks
analyzer.remove()
```

#### WeightAnalyzer

Analyzes model weight parameters without requiring inference.

```python
from analysis import WeightAnalyzer

analyzer = WeightAnalyzer(model, histogram_bins=1000)
results = analyzer.run()

# Results structure:
# {
#   "parameters": {param_idx: {name, component, shape, min, max, mean, std}},
#   "distributions": {component: {bin_centers, counts, data_range}},
#   "component_stats": {component: {count, min, max}},
#   "statistics": {total_parameters, total_values, num_blocks}
# }

# Access parameter statistics
for idx, param in results["parameters"].items():
    print(f"{param['name']}: mean={param['mean']:.4f}, std={param['std']:.4f}")

# Component breakdown
for comp, stats in results["component_stats"].items():
    print(f"{comp}: {stats['count']} params, range=[{stats['min']:.4f}, {stats['max']:.4f}]")

analyzer.save("weights.json")
```

#### Component Classification

Both analyzers classify layers/parameters into components:

| Component | Description |
|-----------|-------------|
| `mha` / `attention` | Multi-head attention layers |
| `mlp` | MLP/FFN layers |
| `norm` | Layer normalization |
| `patch_embed` | Patch embedding layer |
| `embedding` | CLS token, position embeddings |
| `classifier` / `output` | Classification head |

## CLI Reference

### Common Arguments

```bash
python -m cli -m <model> [options] <mode>

Options:
  -m, --model       Model name (required)
  -b, --batch_size  Batch size (default: 100)
  --max_batches     Limit number of batches
  --data            Dataset split: train/val (default: val)
  --seed            Random seed for reproducibility
```

### HR Mode (Hardware Resilience)

```bash
python -m cli -m vit_tiny hr [options]

Options:
  --detect LAYERS   Enable detection on layers
  --method METHOD   Detection method: neuro, checksum, softmean
  --threshold FLOAT Detection threshold
  --inject LAYERS   Enable injection on layers
  --faults N        Number of faults to inject (default: 1)
  --bit_range M,N   Restrict bit flip range
  --repeat N        Number of experiment runs
  -o, --output      Output file for results (JSON)
```

Examples:
```bash
# Neuro detection on fc1 layers
python -m cli -m vit_tiny hr --detect fc1 --method neuro --inject fc1

# Checksum detection with custom threshold
python -m cli -m vit_tiny hr --detect all --method checksum --threshold 1e-4

# Softmean on attention modules
python -m cli -m vit_tiny hr --detect attn --method softmean --inject qkv

# Multiple runs with JSON output
python -m cli -m vit_base hr --detect fc1 --inject fc1 --repeat 100 -o results.json
```

### PA Mode (Parameter Analysis)

Analyze model activations and weight distributions:

```bash
python -m cli -m vit_tiny pa [options]

Options:
  --type TYPE       Analysis type: activations, weights, both (default: activations)
  -o, --output      Output path/directory for JSON results
```

Examples:
```bash
# Analyze activation ranges during inference
python -m cli -m vit_tiny --max_batches 10 pa --type activations -o results/

# Analyze weight distributions (no inference needed)
python -m cli -m vit_tiny pa --type weights -o results/

# Run both analyses
python -m cli -m vit_tiny --max_batches 10 pa --type both -o results/
```

Output files:
- `{model}_activations.json` - Layer activation ranges and histograms
- `{model}_weights.json` - Parameter statistics and distributions

### Save Mode

Pre-compute detection data for faster startup:

```bash
# Save neuro checker data
python -m cli -m vit_tiny save --method neuro --layers all

# Save checksum data
python -m cli -m vit_tiny save --method checksum --layers fc1

# Save fault-free logits (for SDC metrics)
python -m cli -m vit_tiny save --logits
```

## Project Structure

```
src/
├── __init__.py              # Top-level API exports
├── cli.py                   # Command-line interface
│
├── core/                    # Core utilities
│   ├── model.py            # Model loading and data management
│   ├── data.py             # ImageNet dataset
│   ├── layers.py           # Layer traversal utilities
│   └── bits.py             # Bit manipulation for fault injection
│
├── detection/              # Plugin-based detection system
│   ├── __init__.py        # Public API
│   ├── base.py            # BaseDetector abstract class
│   ├── registry.py        # Method registration system
│   ├── manager.py         # Factory function
│   └── methods/           # Detection method plugins
│       ├── neuro.py       # Extra neuron checker
│       ├── checksum.py    # ABFT checksums
│       └── softmean.py    # Softmax-mean checksums
│
├── injection/             # Fault injection
│   └── injector.py       # Bit-flip injector
│
├── eval/                  # Evaluation metrics
│   ├── metrics.py        # Main evaluate() function
│   ├── accuracy.py       # Top-K accuracy
│   └── sdc.py            # SDC metrics
│
└── analysis/             # Model analysis
    ├── activations.py    # Activation analysis
    └── weights.py        # Weight analysis
```

## Supported Models

| Family | Models |
|--------|--------|
| ViT | `vit_tiny`, `vit_small`, `vit_base`, `vit_large` |
| DeiT | `deit_tiny`, `deit_small`, `deit_base` |
| Swin | `swin_tiny`, `swin_small`, `swin_base` |
| BEiT | `beit_base` |

## Layer Filters

| Filter | Description |
|--------|-------------|
| `all` | All linear layers |
| `fc1` | MLP first projection |
| `fc2` | MLP second projection |
| `qkv` | Attention QKV projection |
| `proj` | Attention output projection |
| `attn` | Attention modules (for softmean) |

## Data Setup

The framework expects ImageNet data in the following structure:

```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
├── val/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── imagenet_class_index.json
```

Configure the path in `ModelConfig`:

```python
config = ModelConfig(data_root="/path/to/imagenet")
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_detection.py -v
```
