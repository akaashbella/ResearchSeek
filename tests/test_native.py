import importlib.machinery

from researchseek import _native


def test_native_import():
    assert any(
        _native.__file__.endswith(suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
    )


def test_native_diagnostic():
    assert _native.diagnostic() == 201703
