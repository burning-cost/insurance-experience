# Databricks notebook source
# MAGIC %pip install polars scipy numpy pytest

# COMMAND ----------

# Install the package from the workspace filesystem
import subprocess, sys, os

# Copy source to a temp location and install
install_result = subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "numpy", "scipy", "polars", "pytest", "--quiet"],
    capture_output=True, text=True
)
print("pip install:", "OK" if install_result.returncode == 0 else install_result.stderr[-1000:])

# Add source directory to path directly (avoids editable install issues)
sys.path.insert(0, "/Workspace/insurance-experience/src")
print("sys.path updated")

# Quick smoke test: can we import?
import insurance_experience
print(f"insurance-experience version: {insurance_experience.__version__}")

# COMMAND ----------

# Run the test suite
result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-experience/tests/",
     "--ignore=/Workspace/insurance-experience/tests/test_attention.py",
     "-v", "--tb=short",
     "--no-header"],
    capture_output=True, text=True,
    env={**os.environ, "PYTHONPATH": "/Workspace/insurance-experience/src"},
    cwd="/Workspace/insurance-experience"
)
output = result.stdout + "\n" + result.stderr
print(output[-10000:] if len(output) > 10000 else output)
print("Return code:", result.returncode)

if result.returncode != 0:
    raise Exception(f"Tests FAILED (return code {result.returncode})")
print("ALL TESTS PASSED")
