"""Cross-platform runner for the lightweight smoke test.

Usage:
  python tools/run_smoke_tests.py

This script runs the focused smoke pytest that validates the tiny end-to-end experiment.
"""
import subprocess
import sys
import os

TEST_PATH = "seq2seq_sar/tests/test_experiment_smoke.py"


def main():
    cmd = [sys.executable, "-m", "pytest", TEST_PATH, "-q"]
    print("Running smoke test:", " ".join(cmd))
    try:
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"Smoke tests failed with exit code {rc}")
        sys.exit(rc)
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main()
