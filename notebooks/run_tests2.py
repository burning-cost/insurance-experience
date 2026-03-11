# Databricks notebook source
# MAGIC %pip install pytest polars scipy numpy

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", "/Workspace/insurance-experience", "--quiet"],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("INSTALL FAILED:", result.stderr[-2000:])
else:
    print("Install OK")

# COMMAND ----------

result = subprocess.run(
    [sys.executable, "-m", "pytest",
     "/Workspace/insurance-experience/tests/",
     "--ignore=/Workspace/insurance-experience/tests/test_attention.py",
     "-v", "--tb=short"],
    capture_output=True, text=True,
    cwd="/Workspace/insurance-experience"
)
output = result.stdout + result.stderr
print(output[-8000:] if len(output) > 8000 else output)
print("Return code:", result.returncode)

# COMMAND ----------

# Raise exception if tests failed so the notebook cell fails visibly
if result.returncode != 0:
    raise Exception(f"Tests failed with return code {result.returncode}")
print("All tests passed!")
