"""
run_all.py — Run all tests and examples for micrograd-explained
================================================================
Executes every test suite and example script in sequence.
If any fails, stops and reports which one.

Usage:
    python run_all.py
"""

import os
import subprocess
import sys

# Get the directory where this script lives
ROOT = os.path.dirname(os.path.abspath(__file__))

scripts_to_run = [
    # Core tests (fast)
    "test_engine.py",
    "test_nn.py",
    "test_optim.py",
    "test_loss.py",
    # Comprehensive validation
    "test_validation.py",
]

print("=" * 70)
print("  micrograd-explained: Full Test Suite")
print("=" * 70)
print()

all_passed = True
for script in scripts_to_run:
    script_path = os.path.join(ROOT, script)
    print(f"▶ Running {script}...")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
    )
    if result.returncode != 0:
        print(f"  ✗ FAILED: {script}")
        if result.stdout:
            # Show last 20 lines of output
            lines = result.stdout.strip().split("\n")
            for line in lines[-20:]:
                print(f"    {line}")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-10:]:
                print(f"    [stderr] {line}")
        all_passed = False
        break
    else:
        # Show summary line from output
        lines = result.stdout.strip().split("\n")
        summary = [
            l for l in lines if "passed" in l.lower() or "all tests" in l.lower()
        ]
        if summary:
            print(f"  ✓ {summary[-1].strip()}")
        else:
            print(f"  ✓ PASSED")
    print()

print("=" * 70)
if all_passed:
    print("  ✅ ALL TESTS PASSED SUCCESSFULLY")
else:
    print("  ❌ SOME TESTS FAILED — see output above")
print("=" * 70)
