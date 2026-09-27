#!/usr/bin/env python3
"""Launcher script for Amazon ML Challenge 2026 full pipeline."""

import importlib.util
import os
import sys

# Ensure site packages and local directory are in python path
for p in [
    os.path.abspath("."),
    os.path.abspath("src"),
    "/home/pyro/snap/antigravity-cli/common/local/lib/python3.14/dist-packages",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

runner_path = os.path.abspath("src/run_pipeline.py/run_pipeline.py")
if not os.path.isfile(runner_path):
    runner_path = os.path.abspath("src/run_pipeline.py")

spec = importlib.util.spec_from_file_location("runner_module", runner_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["runner_module"] = mod
spec.loader.exec_module(mod)

if __name__ == "__main__":
    mod.main()
