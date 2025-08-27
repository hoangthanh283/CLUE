#!/usr/bin/env python3
"""
Code quality validation script
"""

import os
import subprocess
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
    print("\nChecking imports...")
    try:
        # Test imports from src modules (using noqa to ignore F401)
        from src.cl_strategies import BaseCLStrategy  # noqa: F401
        from src.data import DatasetLoader  # noqa: F401
        from src.evaluation import Evaluator  # noqa: F401
        from src.models import BaseModel  # noqa: F401
        from src.training import Trainer  # noqa: F401
        from src.utils import load_config, setup_logging  # noqa: F401
        print("✅ Imports: All modules can be imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Imports: {e}")
        return False


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
