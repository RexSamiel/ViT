"""ViT Fault Injection and Detection Framework.

Simple API for fault injection experiments on Vision Transformers.

Example - Fault injection with detection:
    from core.model import Model
    from detection import CheckOne
    from injection import Injector
    from eval import evaluate

    model = Model("vit_tiny")
    detector = CheckOne(model, layers="fc1")
    injector = Injector(model, layers="fc1")

    injector.inject(count=1)
    results = evaluate(model, detector)
    results.print()

    detector.print_results()
    injector.restore()

Adding new detection methods:
    1. Create detection/method2.py
    2. Define _Wrapper class with forward() and detect()
    3. Define Method2 class with wrap/save/load/print methods
    4. Export from detection/__init__.py
"""

from core.model import Model
from core.config import ModelConfig, SUPPORTED_MODELS
from detection import CheckOne, DetectedFault
from injection import Injector, InjectedFault
from eval import evaluate, Results

__all__ = [
    "Model",
    "ModelConfig",
    "SUPPORTED_MODELS",
    "CheckOne",
    "DetectedFault",
    "Injector",
    "InjectedFault",
    "evaluate",
    "Results",
]
__version__ = "0.3.0"
