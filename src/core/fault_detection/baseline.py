"""Baseline persistence for fault detection.

Mirrors the pattern of src/core/library/logits.py (FaultFreeLogits).
Saves and loads per-layer detection values captured during a fault-free run,
keyed by model and the set of monitored layer types so baselines from
different configurations never collide.

Storage format is plain JSON so the file can be inspected and remains
portable across PyTorch versions and devices.  Each layer stores three
scalar floats: ``sum``, ``avg``, and ``min``.

File layout::

    detection_baselines/
        det_baseline_<model_key>_<detection_tag>.json

JSON structure::

    {
        "model_key": "vit_tiny",
        "detection": "all",
        "layers": {
            "Block0.qkv": {"sum": 1.234, "avg": 0.005, "min": -2.5},
            "Block0.proj": {"sum": 0.567, "avg": 0.002, "min": -1.8},
            ...
        }
    }
"""

from __future__ import annotations

import json
from pathlib import Path


# Anchor to the project root (two levels up from this file: fault_detection -> core -> src,
# then one more up to project root).
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = (
    _THIS_FILE.parent.parent.parent.parent
)  # src/core/fault_detection -> project root
BASELINES_DIR = _PROJECT_ROOT / "detection_baselines"

# Metric keys stored per layer.
# Input-based (from detector weight rows) + Output-based (from original output)
METRIC_KEYS: tuple[str, ...] = ("sum_input", "avg_input", "sum", "avg", "min")


def _make_config_tag(detection: str) -> str:
    """Produce a short filesystem-safe string for the detection config.

    Args:
        detection: One of "none", "qkv", "proj", "fc1", "fc2", "all".

    Returns:
        A lowercase string suitable for use in a filename, e.g. "all" or "qkv".
    """
    return detection.lower()


class DetectionBaseline:
    """Save and load per-layer detection values captured from a fault-free run.

    Each layer stores three scalar floats: ``sum``, ``avg``, and ``min``.
    Values are stored in a JSON file so they are human-readable and device-agnostic.

    Usage - faultfree run (save)::

        baseline = DetectionBaseline(model_key="vit_tiny", detection="qkv")
        baseline.save({
            "Block0.qkv": {"sum": 1.23, "avg": 0.005, "min": -2.5},
        })

    Usage - faulty run (load)::

        baseline = DetectionBaseline(model_key="vit_tiny", detection="qkv")
        if baseline.available:
            layers = baseline.get_all()
            # layers["Block0.qkv"] == {"sum": 1.23, "avg": 0.005, "min": -2.5}

    Args:
        model_key: Short model identifier (e.g. "vit_tiny", "deit_small").
        detection: Detection configuration string (e.g. "qkv", "all").
    """

    def __init__(self, model_key: str, detection: str) -> None:
        self.model_key = model_key
        self.detection = detection
        self._tag = _make_config_tag(detection)
        self._filename: Path = (
            BASELINES_DIR / f"det_baseline_{model_key}_{self._tag}.json"
        )
        # data maps layer_name -> {"sum": float, "avg": float, "min": float}
        self.data: dict[str, dict[str, float]] | None = None
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Attempt to load baseline from disk.  Silent on missing file."""
        if self._filename.exists():
            with open(self._filename, "r") as fh:
                doc = json.load(fh)
            self.data = doc.get("layers", {})
            print(
                f"Detection baseline loaded from {self._filename} "
                f"({len(self.data)} layers)"
            )
        else:
            print(
                f"No detection baseline found at {self._filename}. "
                f"Run with --condition faultfree --detection {self.detection} first."
            )

    def save(self, layer_values: dict[str, dict[str, float]]) -> None:
        """Persist baseline values to disk as JSON.

        Args:
            layer_values: Mapping from layer-name string to a dict of metric
                          floats, e.g. ``{"sum": 1.23, "avg": 0.005, "min": -2.5}``.
                          Plain Python floats are expected (not tensors).
        """
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)

        # Coerce to plain Python floats in case tensors or numpy scalars are passed.
        clean: dict[str, dict[str, float]] = {}
        for layer_name, metrics in layer_values.items():
            clean[layer_name] = {k: float(v) for k, v in metrics.items()}

        doc = {
            "model_key": self.model_key,
            "detection": self.detection,
            "layers": clean,
        }

        with open(self._filename, "w") as fh:
            json.dump(doc, fh, indent=2)

        self.data = clean
        print(f"Detection baseline saved to {self._filename} ({len(clean)} layers)")

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def get(self, layer_name: str) -> dict[str, float]:
        """Return the baseline metrics dict for a single layer.

        Args:
            layer_name: Key such as ``"Block3.fc1"``.

        Returns:
            Dict with keys ``"sum"``, ``"avg"``, ``"min"`` as Python floats.

        Raises:
            RuntimeError: If no baseline has been loaded.
            KeyError: If ``layer_name`` is not present in the baseline.
        """
        if self.data is None:
            raise RuntimeError(
                "Detection baseline is not loaded. "
                f"Run with --condition faultfree --detection {self.detection} first."
            )
        if layer_name not in self.data:
            available = list(self.data.keys())
            raise KeyError(
                f"Layer '{layer_name}' not found in baseline. Available: {available}"
            )
        return dict(self.data[layer_name])

    def get_all(self) -> dict[str, dict[str, float]]:
        """Return all baseline values as a nested dict.

        Returns:
            Dict mapping layer-name to ``{"sum": float, "avg": float, "min": float}``.

        Raises:
            RuntimeError: If no baseline has been loaded.
        """
        if self.data is None:
            raise RuntimeError(
                "Detection baseline is not loaded. "
                f"Run with --condition faultfree --detection {self.detection} first."
            )
        return {k: dict(v) for k, v in self.data.items()}

    @property
    def available(self) -> bool:
        """True when a baseline has been successfully loaded or saved."""
        return self.data is not None

    @property
    def layer_names(self) -> list[str]:
        """Sorted list of layer names present in the baseline."""
        if self.data is None:
            return []
        return sorted(self.data.keys())
