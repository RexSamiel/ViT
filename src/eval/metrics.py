"""Evaluation trackers — re-exported for convenience.

AccuracyTracker and SDCTracker each own their own reset / update / aggregate / print / get_summary.
The batch loop lives in main.py; these classes are pure accumulators.
"""

from eval.accuracy import AccuracyTracker
from eval.sdc import SDCTracker

__all__ = ["AccuracyTracker", "SDCTracker"]
