# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run all test files in the current directory sequentially and report results."""
    tests_dir = Path(__file__).parent
    test_files = list(tests_dir.glob("test_*.py"))

    if not test_files:
        print(f"No test files found in {tests_dir}")
        return 0

    print(f"Found {len(test_files)} test files. Running them sequentially...")

    failed_tests = []

    for test_file in test_files:
        print(f"\n{'=' * 80}")
        print(f"Running {test_file.name}...")
        print(f"{'=' * 80}\n")

        # Run pytest on the specific file
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-s", str(test_file)], env=os.environ.copy()
        )

        if result.returncode != 0:
            failed_tests.append(test_file.name)

    print(f"\n{'=' * 80}")
    print("Test Run Summary")
    print(f"{'=' * 80}")

    if failed_tests:
        print(f"Failed tests ({len(failed_tests)}/{len(test_files)}):")
        for failed in failed_tests:
            print(f"  - {failed}")
        return 1
    else:
        print(f"All {len(test_files)} tests passed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
