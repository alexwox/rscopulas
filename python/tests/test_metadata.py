"""Packaging metadata: version, typing marker, and stub completeness."""

from __future__ import annotations

import ast
import importlib.metadata
import pathlib

import rscopulas
from rscopulas import _rscopulas

PACKAGE_DIR = pathlib.Path(rscopulas.__file__).resolve().parent
STUB = PACKAGE_DIR / "_rscopulas.pyi"


def test_version_is_exposed() -> None:
    assert isinstance(rscopulas.__version__, str) and rscopulas.__version__
    assert isinstance(_rscopulas.__version__, str) and _rscopulas.__version__
    try:
        installed = importlib.metadata.version("rscopulas")
    except importlib.metadata.PackageNotFoundError:
        installed = None
    if installed is not None:
        assert rscopulas.__version__ == installed


def test_typing_marker_and_stub_ship_with_the_package() -> None:
    assert (PACKAGE_DIR / "py.typed").is_file()
    assert STUB.is_file()


def _stub_declarations() -> tuple[dict[str, set[str]], set[str], set[str]]:
    tree = ast.parse(STUB.read_text(encoding="utf-8"))
    classes: dict[str, set[str]] = {}
    functions: set[str] = set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes[node.name] = {
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            }
        elif isinstance(node, ast.FunctionDef):
            functions.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return classes, functions, names


def test_stub_declares_every_extension_export() -> None:
    classes, functions, names = _stub_declarations()
    exported = {name for name in dir(_rscopulas) if not name.startswith("__")}
    exported.add("__version__")
    missing = exported - (set(classes) | functions | names)
    assert not missing, f"names missing from _rscopulas.pyi: {sorted(missing)}"


def test_stub_methods_exist_on_extension_classes() -> None:
    classes, _, _ = _stub_declarations()
    assert classes, "stub declares no classes"
    for class_name, methods in classes.items():
        real = getattr(_rscopulas, class_name)
        missing = {method for method in methods if not hasattr(real, method)}
        assert not missing, f"{class_name} stub declares {sorted(missing)} that the extension lacks"


def test_public_api_all_is_resolvable() -> None:
    for name in rscopulas.__all__:
        assert hasattr(rscopulas, name), name
    assert "to_pseudo_obs" in rscopulas.__all__
    assert "InvalidInputError" in rscopulas.__all__
