"""Test unified DataCollector for activation and weight analysis."""

import sys


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    try:
        from src.core.parameter_analysis import ParameterAnalyzer, run, save_results

        print("  ✓ ParameterAnalyzer imports correctly")

        from src.core.parameter_analysis.collector import DataCollector

        print("  ✓ DataCollector imports correctly")

        from src.core.library.histogram import (
            get_histogram_config,
            sample_and_histogram,
            compute_histogram,
        )

        print("  ✓ Histogram functions import correctly")

        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_histogram_config():
    """Test histogram configuration for different analysis types."""
    print("\nTesting histogram configuration...")
    try:
        from src.core.library.histogram import get_histogram_config

        aa_range, aa_res, aa_bins = get_histogram_config("aa")
        print(
            f"  Activation config: range={aa_range}, resolution={aa_res}, bins={aa_bins}"
        )
        assert aa_range == 10000, f"Expected aa_range=10000, got {aa_range}"
        assert aa_res == 1.0, f"Expected aa_res=1.0, got {aa_res}"
        assert aa_bins == 20001, f"Expected aa_bins=20001, got {aa_bins}"
        print("  ✓ Activation config correct")

        wa_range, wa_res, wa_bins = get_histogram_config("wa")
        print(f"  Weight config: range={wa_range}, resolution={wa_res}, bins={wa_bins}")
        assert wa_range == 100, f"Expected wa_range=100, got {wa_range}"
        assert wa_res == 0.01, f"Expected wa_res=0.01, got {wa_res}"
        assert wa_bins == 20001, f"Expected wa_bins=20001, got {wa_bins}"
        print("  ✓ Weight config correct (finer resolution)")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_collector_initialization():
    """Test DataCollector initialization for both types."""
    print("\nTesting DataCollector initialization...")
    try:
        from src.core.parameter_analysis.collector import DataCollector

        # Test activation collector
        aa_collector = DataCollector("aa", sampling_percent=1.0)
        print(f"  Activation collector: type={aa_collector.analysis_type}")
        print(
            f"    - bin_range={aa_collector.bin_range}, resolution={aa_collector.bin_resolution}"
        )
        print(f"    - components={aa_collector.get_components()}")
        assert aa_collector.analysis_type == "aa"
        assert aa_collector.bin_resolution == 1.0
        assert hasattr(aa_collector, "_hook_manager")
        print("  ✓ Activation collector initialized correctly")

        # Test weight collector
        wa_collector = DataCollector("wa")
        print(f"  Weight collector: type={wa_collector.analysis_type}")
        print(
            f"    - bin_range={wa_collector.bin_range}, resolution={wa_collector.bin_resolution}"
        )
        print(f"    - components={wa_collector.get_components()}")
        assert wa_collector.analysis_type == "wa"
        assert wa_collector.bin_resolution == 0.01
        assert not hasattr(wa_collector, "_hook_manager") or not hasattr(
            wa_collector, "total_samples"
        )
        print("  ✓ Weight collector initialized correctly (finer bins)")

        # Test invalid type
        try:
            bad_collector = DataCollector("invalid")
            print("  ✗ Should have raised ValueError for invalid type")
            return False
        except ValueError:
            print("  ✓ Correctly rejects invalid analysis type")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_parameter_analyzer():
    """Test ParameterAnalyzer uses correct histogram config."""
    print("\nTesting ParameterAnalyzer...")
    try:
        from src.core.parameter_analysis import ParameterAnalyzer

        # Test activation analyzer
        aa_analyzer = ParameterAnalyzer("aa", sampling_percent=1.0)
        print(f"  Activation analyzer: type={aa_analyzer.analysis_type}")
        print(
            f"    - bin_range={aa_analyzer.bin_range}, resolution={aa_analyzer.bin_resolution}"
        )
        print(f"    - num_bins={aa_analyzer.num_bins}")
        assert aa_analyzer.bin_resolution == 1.0
        assert aa_analyzer.num_bins == 20001
        print("  ✓ Activation analyzer configured correctly")

        # Test weight analyzer
        wa_analyzer = ParameterAnalyzer("wa")
        print(f"  Weight analyzer: type={wa_analyzer.analysis_type}")
        print(
            f"    - bin_range={wa_analyzer.bin_range}, resolution={wa_analyzer.bin_resolution}"
        )
        print(f"    - num_bins={wa_analyzer.num_bins}")
        assert wa_analyzer.bin_resolution == 0.01
        assert wa_analyzer.num_bins == 20001
        print("  ✓ Weight analyzer configured correctly (0.01 resolution)")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_collector_methods():
    """Test that DataCollector has all required methods."""
    print("\nTesting DataCollector methods...")
    try:
        from src.core.parameter_analysis.collector import DataCollector

        collector = DataCollector("aa")

        required_methods = [
            "collect",
            "get_components",
            "cleanup",
            "classify_parameter",
            "extract_block_idx_from_param",
            "record_activation_callback",
            "record_weight",
            "_collect_activations",
            "_collect_weights",
        ]

        missing = []
        for method in required_methods:
            if not hasattr(collector, method):
                missing.append(method)
            else:
                print(f"  ✓ Has method: {method}")

        if missing:
            print(f"  ✗ Missing methods: {missing}")
            return False

        print("  ✓ All required methods present")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Unified DataCollector Implementation")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Histogram Config", test_histogram_config),
        ("Collector Initialization", test_collector_initialization),
        ("Parameter Analyzer", test_parameter_analyzer),
        ("Collector Methods", test_collector_methods),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' failed with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed!")
        print("\nKey improvements:")
        print("  1. Unified DataCollector handles both aa and wa")
        print("  2. Weight analysis uses 0.01 bin resolution (100x finer)")
        print("  3. Activation analysis uses 1.0 bin resolution (integer bins)")
        print("  4. Histogram properly adapts based on analysis_type")
    else:
        print("Some tests failed - see details above")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
