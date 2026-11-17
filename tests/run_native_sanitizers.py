"""Run native tests against a separately instrumented extension."""

import argparse
import importlib.util
from pathlib import Path
import sys

import pytest
import researchseek


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("researchseek._native", args.extension)
    if spec is None or spec.loader is None:
        parser.error("Expected a compiled _native extension path")
    module = importlib.util.module_from_spec(spec)
    sys.modules["researchseek._native"] = module
    spec.loader.exec_module(module)
    researchseek._native = module
    tests = Path(__file__).resolve().parent
    return pytest.main([str(tests / "test_native_search.py"), str(tests / "test_retrieval.py"), "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
