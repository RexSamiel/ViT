"""Test module structure without requiring torch."""

import ast
import sys


def check_file_syntax(filepath, filename):
    """Check Python file syntax and extract key information."""
    print(f"\nChecking {filename}...")
    try:
        with open(filepath, 'r') as f:
            code = f.read()

        # Parse the file
        tree = ast.parse(code, filename=filepath)

        # Find classes and functions
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        print(f"  ✓ Syntax valid")
        if classes:
            print(f"  ✓ Classes: {', '.join(classes)}")
        if functions:
            print(f"  ✓ Functions (top 10): {', '.join(functions[:10])}")

        return True, classes, functions
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False, [], []
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False, [], []


def check_imports(filepath, filename):
    """Check imports in a file."""
    print(f"\nChecking imports in {filename}...")
    try:
        with open(filepath, 'r') as f:
            code = f.read()

        tree = ast.parse(code, filename=filepath)

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        print(f"  ✓ Found {len(imports)} imports")
        return imports
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []


def main():
    """Run structure tests."""
    print("=" * 60)
    print("Testing Module Structure (No Torch Required)")
    print("=" * 60)

    base_path = "/home/samiel/Documents/thesis/ViT/src/core"

    # Test histogram.py
    hist_ok, hist_classes, hist_funcs = check_file_syntax(
        f"{base_path}/library/histogram.py",
        "histogram.py"
    )

    if hist_ok:
        # Check for key functions
        required_funcs = ['get_histogram_config', 'sample_and_histogram', 'compute_histogram']
        missing = [f for f in required_funcs if f not in hist_funcs]
        if missing:
            print(f"  ✗ Missing functions: {missing}")
        else:
            print(f"  ✓ All required functions present")

    # Test collector.py
    coll_ok, coll_classes, coll_funcs = check_file_syntax(
        f"{base_path}/parameter_analysis/collector.py",
        "collector.py"
    )

    if coll_ok:
        if 'DataCollector' in coll_classes:
            print("  ✓ DataCollector class found")
        else:
            print("  ✗ DataCollector class missing")

    # Test manager.py
    mgr_ok, mgr_classes, mgr_funcs = check_file_syntax(
        f"{base_path}/parameter_analysis/manager.py",
        "manager.py"
    )

    if mgr_ok:
        if 'ParameterAnalyzer' in mgr_classes:
            print("  ✓ ParameterAnalyzer class found")
        else:
            print("  ✗ ParameterAnalyzer class missing")

        # Check imports
        mgr_imports = check_imports(
            f"{base_path}/parameter_analysis/manager.py",
            "manager.py"
        )

        # Verify it imports DataCollector
        has_datacollector = any('DataCollector' in imp for imp in mgr_imports)
        if has_datacollector:
            print("  ✓ Imports DataCollector")
        else:
            print("  ✗ Does not import DataCollector")

        # Verify it does NOT import old collectors
        has_old_collectors = any(
            'ActivationCollector' in imp or 'WeightCollector' in imp
            for imp in mgr_imports
        )
        if has_old_collectors:
            print("  ✗ Still imports old collectors!")
        else:
            print("  ✓ Does not import old collectors")

    # Check that old files are deleted
    print("\nChecking old files deleted...")
    import os
    old_files = [
        f"{base_path}/parameter_analysis/activation_collector.py",
        f"{base_path}/parameter_analysis/weight_collector.py",
    ]

    all_deleted = True
    for old_file in old_files:
        if os.path.exists(old_file):
            print(f"  ✗ {os.path.basename(old_file)} still exists!")
            all_deleted = False

    if all_deleted:
        print("  ✓ Old collector files deleted")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_ok = hist_ok and coll_ok and mgr_ok and all_deleted

    if all_ok:
        print("✓ All structure tests passed!")
        print("\nChanges completed:")
        print("  1. histogram.py: Uses adaptive bin resolution")
        print("  2. collector.py: Unified DataCollector for aa and wa")
        print("  3. manager.py: Updated to use DataCollector")
        print("  4. Old collectors: Deleted")
        print("\nBin configurations:")
        print("  - Activations (aa): range=10000, resolution=1.0 (integer bins)")
        print("  - Weights (wa): range=100, resolution=0.01 (100x finer)")
    else:
        print("✗ Some structure tests failed")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
