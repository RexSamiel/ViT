"""Fault detection for ViT models.

Usage:
    from detection import CheckOne

    detector = CheckOne(model, layers="fc1")
    # ... inject faults, run forward ...
    detector.print_results()
    detector.remove()

Adding new methods:
    1. Create detection/method2.py
    2. Define _Wrapper(nn.Module) with forward() and detect()
    3. Define Method2 class with wrap/save/load/print methods
    4. Export from this __init__.py
"""

from detection.checksum import Checksum
from detection.checkone import DetectedFault, CheckOne

__all__ = ["CheckOne", "Checksum", "DetectedFault"]
