"""Smoke test: the package imports and its subpackages exist.

Replaced by real tests as each component lands. Its job for now is to give CI
something to run so the first pipeline goes green.
"""
import importlib


def test_package_imports():
    assert importlib.import_module("macs") is not None


def test_subpackages_exist():
    for sub in ("llm", "trace", "sandbox", "tasks", "metrics", "prompts", "agents"):
        assert importlib.import_module(f"macs.{sub}") is not None
