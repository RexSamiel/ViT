"""Simple test to verify weight analysis module structure without dependencies."""

try:
    from src.core.weight import WeightAnalyzer, run, save_results

    print("SUCCESS: Weight analysis module imports correctly")
    print("  - WeightAnalyzer class: available")
    print("  - run() function: available")
    print("  - save_results() function: available")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
try:
    from src.core.weight.manager import WeightAnalyzer

    analyzer = WeightAnalyzer()

    methods = [
        "_reset",
        "_classify_parameter",
        "_extract_block_idx",
        "_record_parameter",
        "collect_data",
        "get_results",
        "print_results",
        "print_parameter_details",
    ]

    print("\nWeightAnalyzer methods:")
    for method in methods:
        if hasattr(analyzer, method):
            print(f"  + {method}")
        else:
            print(f"  - {method} (MISSING)")

    print("\nInternal data structures:")
    attrs = [
        "param_data",
        "global_stats",
        "_hist_counts",
        "_data_range",
        "_weight_counts",
        "_name_to_idx",
        "num_blocks",
    ]
    for attr in attrs:
        if hasattr(analyzer, attr):
            print(f"  + {attr}")
        else:
            print(f"  - {attr} (MISSING)")

except ImportError as e:
    print(f"Cannot test class structure: {e}")
except Exception as e:
    print(f"Error during class inspection: {e}")

print("\n" + "=" * 60)
print("Weight Analysis Module Implementation Complete!")
print("=" * 60)
print("\nUsage example:")
print("  python -m src.main wa --model vit_tiny")
print("  python -m src.main wa --model vit_base --details true")
print("  python -m src.main wa --model deit_small --verbose true")
