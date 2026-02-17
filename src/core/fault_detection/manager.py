"""Fault detection - high-level API.

Public surface (consumed by main.py and fault_injection/manager.py):

    add_detection(model, detection, num_blocks)   -> dict[str, DetectorNeurons]
    remove_detection(neurons)
    update_tracker(neurons, tracker)
    capture_baselines(runner, detection, model_key, verbose) -> dict[str, dict[str, float]]
    detect_faults(runner, detection, model_key, threshold, verbose)
         -> list[dict]  (detection result rows)
    print_results(results, threshold)
"""

from __future__ import annotations

import torch

from src.core.fault_detection.neuron import (
    DetectorNeurons,
    get_qkv_layer,
    get_proj_layer,
    get_fc1_layer,
    get_fc2_layer,
)
from src.core.fault_detection.baseline import DetectionBaseline
from src.core.fault_detection.tracker import DetectionTracker
from src.core.library.layers import get_num_blocks
from src.core.library.utils import resolve_amp


# ---------------------------------------------------------------------------
# Layer-type resolution helpers
# ---------------------------------------------------------------------------

_LAYER_GETTERS: dict[str, callable] = {
    "qkv": get_qkv_layer,
    "proj": get_proj_layer,
    "fc1": get_fc1_layer,
    "fc2": get_fc2_layer,
}


def _layer_types_for(detection: str) -> list[str]:
    """Expand the ``detection`` config value into a list of layer-type strings.

    Args:
        detection: One of "none", "qkv", "proj", "fc1", "fc2", "all".

    Returns:
        Ordered list of individual layer-type keys, e.g. ``["qkv", "proj"]``
        for ``"all"``.  Empty list when ``detection`` is ``"none"``.
    """
    if detection == "none":
        return []
    if detection == "all":
        return ["qkv", "proj", "fc1", "fc2"]
    return [detection]


def _layer_name(block_idx: int, layer_type: str) -> str:
    """Canonical layer identifier string, e.g. ``"Block0.qkv"``."""
    return f"Block{block_idx}.{layer_type}"


def add_detection(
    model: torch.nn.Module,
    detection: str,
    num_blocks: int,
) -> dict[str, DetectorNeurons]:
    """Attach detector neurons to every targeted layer in the model.

    Registers a forward hook on each targeted layer so that detection values
    (sum, avg, min of inputs) are captured during subsequent inference passes.

    Args:
        model: The transformer model.
        detection: Detection configuration ("none", "qkv", "proj", "fc1",
                   "fc2", or "all").
        num_blocks: Total number of transformer blocks in the model.

    Returns:
        Dict mapping layer-name string to its :class:`DetectorNeurons` instance.
        Empty dict when ``detection`` is ``"none"``.
    """
    layer_types = _layer_types_for(detection)
    neurons: dict[str, DetectorNeurons] = {}

    for block_idx in range(num_blocks):
        for lt in layer_types:
            getter = _LAYER_GETTERS[lt]
            layer = getter(model, block_idx)
            neuron = DetectorNeurons()
            neuron.add_to_layer(layer)
            neurons[_layer_name(block_idx, lt)] = neuron

    return neurons


def remove_detection(neurons: dict[str, DetectorNeurons]) -> None:
    """Remove all detection neuron hooks.

    Args:
        neurons: Dict returned by :func:`add_detection`.
    """
    for neuron in neurons.values():
        neuron.remove()
    neurons.clear()


def update_tracker(
    neurons: dict[str, DetectorNeurons],
    tracker: DetectionTracker,
) -> None:
    """Read current detection values from all neurons into the tracker.

    Call this once per inference batch, *after* the forward pass completes.

    Each :class:`DetectorNeurons` yields a dict
    ``{"sum": tensor, "avg": tensor, "min": tensor}`` which is forwarded to
    :meth:`DetectionTracker.update`.

    Args:
        neurons: Active neurons dict from :func:`add_detection`.
        tracker: :class:`DetectionTracker` accumulating values for this run.
    """
    for layer_name, neuron in neurons.items():
        vals = neuron.get_detection_values()
        if vals is not None:
            tracker.update(layer_name, vals)


def capture_baselines(
    runner,
    detection: str,
    model_key: str,
    verbose: bool = True,
) -> dict[str, dict[str, float]]:
    """Run a full evaluation pass and save per-layer baseline means to JSON.

    The model is run on all batches with detection hooks attached.  The mean
    (or min, for the min metric) per layer is saved to disk via
    :class:`DetectionBaseline`.

    Args:
        runner: :class:`~src.core.model.ModelRunner` instance.
        detection: Detection config string (must not be ``"none"``).
        model_key: Short model identifier for the baseline filename.
        verbose: Whether to print progress.

    Returns:
        Nested dict mapping layer-name to ``{"sum": float, "avg": float, "min": float}``.
    """
    num_blocks = get_num_blocks(runner.model)
    tracker = DetectionTracker()
    neurons = add_detection(runner.model, detection, num_blocks)

    if verbose:
        layer_types = _layer_types_for(detection)
        print(
            f"\nCapturing detection baselines "
            f"({num_blocks} blocks x {layer_types} = {len(neurons)} layers)..."
        )

    batches = runner.get_batches()
    use_amp = resolve_amp(runner.config)

    with torch.inference_mode():
        for images, _ in batches:
            runner.inference(images, use_amp)
            update_tracker(neurons, tracker)

    remove_detection(neurons)

    means = tracker.get_means()

    # Persist to disk as JSON.
    baseline = DetectionBaseline(model_key=model_key, detection=detection)
    baseline.save(means)

    if verbose:
        print(f"Baseline captured for {len(means)} layers.")

    return means


def detect_faults(
    runner,
    detection: str,
    model_key: str,
    threshold: float = 0.1,
    current_tracker: DetectionTracker | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Load baseline and compare current run against it.

    If ``current_tracker`` is provided (i.e. detection was run *inline* during
    a faulty fi pass) it is used directly; otherwise a fresh evaluation pass
    is performed.

    Args:
        runner: :class:`~src.core.model.ModelRunner` instance.
        detection: Detection config string.
        model_key: Short model identifier used to find the saved baseline.
        threshold: Relative difference above which a layer is flagged.
        current_tracker: Pre-populated tracker from an inline run.  When
                         ``None`` a fresh pass is executed.
        verbose: Whether to print progress.

    Returns:
        List of result dicts (see :meth:`DetectionTracker.compare`), one row
        per (layer, metric) combination.
    """
    baseline = DetectionBaseline(model_key=model_key, detection=detection)

    if not baseline.available:
        if verbose:
            print(
                "Cannot run fault detection: no baseline found. "
                f"Run with --condition faultfree --detection {detection} first."
            )
        return []

    # baseline.get_all() returns dict[str, dict[str, float]] (no tensors).
    baseline_means: dict[str, dict[str, float]] = baseline.get_all()

    # Use a provided tracker or run a fresh evaluation pass.
    if current_tracker is not None:
        tracker = current_tracker
    else:
        num_blocks = get_num_blocks(runner.model)
        tracker = DetectionTracker(threshold=threshold)
        neurons = add_detection(runner.model, detection, num_blocks)

        if verbose:
            print(f"\nRunning detection pass ({len(neurons)} layers)...")

        batches = runner.get_batches()
        use_amp = resolve_amp(runner.config)

        with torch.inference_mode():
            for images, _ in batches:
                runner.inference(images, use_amp)
                update_tracker(neurons, tracker)

        remove_detection(neurons)

    # Inject the threshold into the tracker when using a pre-built one.
    tracker.threshold = threshold

    results = tracker.compare(baseline_means)
    return results


def print_results(results: list[dict], threshold: float) -> None:
    """Print the detection results table.

    Delegates to :meth:`DetectionTracker.print_results`.

    Args:
        results: List of result dicts from :func:`detect_faults`.
        threshold: Detection threshold (displayed in the header).
    """
    DetectionTracker.print_results(results, threshold)
