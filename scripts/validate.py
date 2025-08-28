#!/usr/bin/env python3
"""
Code quality validation script
"""
import os
import subprocess
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_flake8():
    """Run flake8 linting"""
    print("Running flake8...")
    result = subprocess.run(["flake8", "."], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ flake8: All checks passed")
        return True
    else:
        print("❌ flake8: Issues found")
        print(result.stdout)
        return False


def check_imports():
    """Check if imports work correctly"""
    print("Checking imports...")

    import_tests = [
        ("src.cl_strategies", "BaseCLStrategy"),
        ("src.data", "DatasetLoader"),
        ("src.evaluation", "Evaluator"),
        ("src.models", "BaseModel"),
        ("src.training", "Trainer"),
        ("src.utils", ["load_config", "setup_logging"]),
    ]

    failed_imports = []

    for module_name, items in import_tests:
        try:
            module = __import__(module_name, fromlist=[items] if isinstance(items, str) else items)
            if isinstance(items, str):
                getattr(module, items)
            else:
                for item in items:
                    getattr(module, item)
        except (ImportError, AttributeError) as e:
            failed_imports.append(f"{module_name}: {e}")

    if failed_imports:
        print("❌ Imports: Failed imports found:")
        for failure in failed_imports:
            print(f"  - {failure}")
        return False
    else:
        print("✅ Imports: All modules can be imported successfully")
        return True


def main():
    """Main validation function"""
    print("=== Code Quality Validation ===\n")

    checks = [
        run_flake8(),
        check_imports()
    ]

    if all(checks):
        print("\n🎉 All validation checks passed!")
        return 0
    else:
        print("\n⚠️  Some validation checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
