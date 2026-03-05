"""ViT Fault Injection and Detection Framework.

Simple API for fault injection experiments on Vision Transformers.

Example - Fault injection with detection:
    from vit_fault import Model, Detector, Injector, evaluate

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

    detector.print_results()
    injector.restore()

Example - Activation analysis:
    from vit_fault import Model
    from vit_fault.analysis import ActivationAnalyzer

    model = Model("vit_tiny")
    analyzer = ActivationAnalyzer(model)
    analyzer.run(num_batches=10)
    analyzer.save("activations_vit_tiny.json")
"""

from vit_fault.core.model import Model
from vit_fault.detection.detector import Detector
from vit_fault.injection.injector import Injector
from vit_fault.eval.metrics import evaluate, Results

__all__ = ["Model", "Detector", "Injector", "evaluate", "Results"]
__version__ = "0.1.0"
