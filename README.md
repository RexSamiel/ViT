# ViT Fault Injection and Detection Framework

A research framework for fault injection experiments, parameter analysis, and ABFT-based fault detection on Vision Transformers (ViT, DeiT, Swin).

## Installation

```bash
git clone <repo-url>
cd ViT

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu124
```

Set the ImageNet path via environment variable (default: `/run/media/samiel/K_USB_256/imagenet/`):

```bash
export IMAGENET_PATH=/path/to/imagenet
```

## Quick Start

```bash
# Run 100 fault injection experiments on vit_tiny with CheckOne detection and zero correction
python -m cli -m vit_tiny -r 100 -w 10 --max_batches 1 --batch_size 100 \
  fi --faults 1 --bit_range 0,31 --fault_seed 1 \
  hr --method checkone --detect all --correction zero

# Save calibration data before running detection
python -m cli -m vit_tiny save --inputs --threshold --weights --logits
```

## CLI Reference

The CLI uses chained subcommands. Global options come first, then one or more subcommands:

```bash
python -m cli -m <model> [global options] <subcommand> [subcommand options] ...
```

### Global Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | required | Model name (see supported models) |
| `-b, --batch_size` | 100 | Batch size |
| `--max_batches` | 1 | Number of batches per run |
| `-r, --repeat` | 1 | Number of experiment repetitions |
| `-w, --warmup` | 0 | Silent GPU warmup passes before timing |
| `--data` | val | Dataset split: `train` or `val` |
| `--seed` | random | Global random seed |
| `--info` | false | Verbose per-run output |
| `--time` | false | Show total script execution time |
| `-o, --output` | none | Output JSON file path |

### `fi` — Fault Injection

Injects bit-flip faults into model weights.

```bash
python -m cli -m vit_tiny fi [options]
```

| Flag | Description |
|------|-------------|
| `--layers` | Target layers: `all`, `fc1`, `fc2`, `qkv`, `proj` (default: `all`) |
| `--faults N` | Number of bit-flip faults to inject |
| `--ber FLOAT` | Bit error rate (alternative to `--faults`) |
| `--bit_range LO,HI` | Restrict flips to bit range, e.g. `23,31` or `0,31^30` to exclude bit 30 |
| `--fault_seed N` | RNG seed for fault injection — same seed across runs injects identical faults |
| `--time` | Show per-run inference timing |

Bit range modes used in experiments:

| Mode | Flag | Bits affected |
|------|------|---------------|
| Unrestricted | `0,31` | All 32 bits |
| Without bit 30 | `0,31^30` | All except bit 30 |
| Without mantissa | `23,31` | Exponent and sign bits only |

### `hr` — Hardware Resilience (Detection)

Wraps model layers with ABFT-based fault detection and optional correction.

```bash
python -m cli -m vit_tiny fi --faults 1 hr [options]
```

| Flag | Description |
|------|-------------|
| `--method` | Detection method: `checkone`, `checksum`, `baseline` (default: `checkone`) |
| `--detect` | Layers to wrap: `all`, `fc1`, `fc2`, `qkv`, `proj` |
| `--correction` | Correction mode: `zero`, `rerun`, `correct` |
| `--time` | Show per-layer detection timing |

Detection methods:

| Method | Mechanism | GPU-portable |
|--------|-----------|--------------|
| `checkone` | Row-sum weight check — compares `ones @ W.T` against saved sums. Input-independent. | Yes |
| `checksum` | ApproxABFT row and column checksums — row check detects faults, col check localises which output feature | No (threshold depends on GPU BLAS noise) |
| `baseline` | No detection — measures overhead baseline | — |

Correction modes:

| Mode | Behaviour |
|------|-----------|
| `zero` | Zero the entire faulty output column |
| `rerun` | Recompute the faulty output feature from saved clean weights |
| `correct` | Algebraic correction using weight diff (simulation only) |

### `save` — Save Calibration Data

Pre-computes and saves detection baselines. Must be run before `hr`.

```bash
python -m cli -m vit_tiny save [options]
```

| Flag | Description |
|------|-------------|
| `--logits` | Save fault-free logits for SDC metrics |
| `--inputs` | Calibrate input fault detection range (CheckOne) |
| `--threshold` | Calibrate detection thresholds via 3-sigma rule (both methods) |
| `--weights` | Save full weight matrices for col-check and rerun correction |
| `--margin FLOAT` | Sigma multiplier for threshold calibration (default: 3.0) |
| `--layers` | Layers to calibrate (default: `all`) |

Typical full calibration:

```bash
python -m cli -m vit_tiny --max_batches 100 save --inputs --threshold --weights --logits
```

Saved files (under `data/{model}/`):

| File | Contents |
|------|----------|
| `calibration/checkone.pt` | Weight sums, input ranges, atol/rtol per layer |
| `calibration/checksum.pt` | Row and col check atol/rtol per layer |
| `weights/checkone.pt` | Full weight matrices for rerun correction |
| `weights/checksum.pt` | Full weight matrices for col check and rerun correction |
| `logits/{n}_samples.pt` | Fault-free logits for SDC comparison |

### `pa` — Parameter Analysis

Analyzes weight distributions and activation ranges.

```bash
python -m cli -m vit_tiny pa [options]
```

| Flag | Description |
|------|-------------|
| `--type` | `activations`, `weights`, or `both` (default: `activations`) |
| `-o, --output` | Output path for JSON results |

## Running Experiments

A batch experiment script is provided at `scripts/run_detection.sh`. Configure the flags at the top:

```bash
RUN_BASELINE=true
RUN_DETECTION=true
RUN_ZERO=true
RUN_CORRECTION=false

MODELS=(vit_tiny deit_tiny swin_tiny)
METHODS=(checkone checksum)
FAULT_SEED=1
```

Then run from the repo root:

```bash
bash scripts/run_detection.sh
```

Results are appended to `results/detection_results/detection_measurements/runs.json` and merged into the database JSONs (`baseline.json`, `detection.json`, `zero.json`, `correction.json`).

### Viewing Results

```bash
cd results/detection_results
streamlit run plot.py
```

## Project Structure

```
scripts/
└── run_detection.sh          # Batch experiment runner

src/
├── cli.py                    # CLI argument parsing and entry point
├── main.py                   # Experiment orchestration
│
├── core/
│   ├── config.py             # Paths, supported models, ModelConfig
│   ├── model.py              # Model loading, data, fault-free logits
│   ├── data.py               # ImageNet dataset
│   ├── layers.py             # Layer traversal and wrapping utilities
│   └── bits.py               # Bit manipulation for fault injection
│
├── detection/
│   ├── checkone.py           # CheckOne — row-sum ABFT detector
│   ├── checksum.py           # CheckSum — approxABFT row/col detector
│   └── baseline.py           # Baseline — no detection, overhead only
│
├── injection/
│   └── injector.py           # Bit-flip fault injector
│
├── eval/
│   ├── accuracy.py           # Top-1/Top-5 accuracy with Welford aggregation
│   └── sdc.py                # SDC metrics (logit SDC rate, critical SDC, MSDC)
│
└── analysis/
    ├── activations.py        # Activation range and histogram analysis
    └── weights.py            # Weight distribution analysis

results/
├── detection_results/
│   ├── baseline.json         # Aggregated baseline results database
│   ├── detection.json        # Detection-only results database
│   ├── zero.json             # Zero-correction results database
│   ├── correction.json       # Arithmetic correction results database
│   ├── merge.py              # Merges run output into database JSONs
│   └── plot.py               # Streamlit results explorer
```

## Supported Models

| Family | Keys |
|--------|------|
| ViT | `vit_tiny`, `vit_small`, `vit_base`, `vit_large` |
| DeiT | `deit_tiny`, `deit_small`, `deit_base` |
| Swin | `swin_tiny`, `swin_small`, `swin_base` |
| BEiT | `beit_base` |

## Metrics

| Metric | Description |
|--------|-------------|
| Top-1 / Top-5 accuracy | Classification accuracy across runs (mean ± std) |
| Detection accuracy | % of runs where a weight fault was detected |
| False positives | Detections fired on layers with no injected fault |
| Logit SDC rate | % of output logits with >0.1% relative change vs fault-free |
| Critical SDC (Top-1/5) | % of runs where the predicted class changed |
| MSDC | Median absolute logit change vs fault-free |
| Batch speed | Mean ± std inference time per batch (ms) |
