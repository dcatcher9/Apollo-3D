#!/usr/bin/env python3
"""Path-independent identities for native Python package implementations.

Version strings are not sufficient for byte-addressed preprocessing caches: a
wheel may be rebuilt with different compiler flags or bundled codecs without a
version change.  This module hashes the native extensions and bundled shared
libraries that implement image and numerical operations while deliberately
excluding their installation paths from the semantic identity.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.machinery
from pathlib import Path


_SHARED_LIBRARY_SUFFIXES = (".dll", ".dylib", ".pyd", ".so")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_native_binary(path: Path) -> bool:
    name = path.name.lower()
    return (
        any(name.endswith(suffix.lower())
            for suffix in importlib.machinery.EXTENSION_SUFFIXES) or
        any(name.endswith(suffix) or f"{suffix}." in name
            for suffix in _SHARED_LIBRARY_SUFFIXES)
    )


def _module_roots(module):
    roots = []
    module_paths = getattr(module, "__path__", None)
    if module_paths is not None:
        roots.extend(Path(value).resolve(strict=True) for value in module_paths)
    module_file = getattr(module, "__file__", None)
    if module_file:
        path = Path(module_file).resolve(strict=True)
        roots.append(path if path.is_dir() else path.parent)
    unique = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique


def _native_rows_uncached(role: str, root_names: tuple[str, ...]):
    candidates = {}
    for root_index, root_name in enumerate(root_names):
        root = Path(root_name).resolve(strict=True)
        search_roots = [(f"package{root_index}", root)]
        # NumPy/OpenCV wheels commonly keep dependent DLLs beside the package
        # in ``<distribution>.libs``.  Include any such sibling deterministically.
        tokens = {role.lower().replace("-", "_").replace(" ", "_"),
                  root.name.lower()}
        for sibling in sorted(root.parent.glob("*.libs")):
            sibling_name = sibling.name.lower()
            if (sibling.is_dir() and
                    any(token and token in sibling_name for token in tokens)):
                search_roots.append((f"package{root_index}.libs.{sibling.name}",
                                     sibling.resolve(strict=True)))
        for root_label, search_root in search_roots:
            for path in sorted(search_root.rglob("*")):
                if (path.is_file() and not path.is_symlink() and
                        _is_native_binary(path)):
                    relative = path.relative_to(search_root).as_posix()
                    candidates[f"{role}/{root_label}/{relative}"] = path
    if not candidates:
        raise RuntimeError(f"{role} has no discoverable native implementation")
    return tuple({
        "role": name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    } for name, path in sorted(candidates.items()))


@lru_cache(maxsize=None)
def _native_rows(role: str, root_names: tuple[str, ...]):
    return _native_rows_uncached(role, root_names)


def module_native_identity(role: str, module, *, fresh=False):
    """Return path-independent byte identities for one native-backed module."""
    if not isinstance(role, str) or not role:
        raise RuntimeError("native runtime role is invalid")
    roots = tuple(str(path) for path in _module_roots(module))
    if not roots:
        raise RuntimeError(f"cannot locate native implementation for {role}")
    rows = (_native_rows_uncached(role, roots) if fresh else
            _native_rows(role, roots))
    return [dict(row) for row in rows]


def verify_module_native_identity(role: str, module, expected):
    """Rehash native files without the process-local memoization layer."""
    observed = module_native_identity(role, module, fresh=True)
    if observed != expected:
        raise RuntimeError(f"{role} native implementation changed during generation")
    return observed


def python_file_identity(role: str, path):
    """Return a path-independent identity for byte-producing Python code."""
    if not isinstance(role, str) or not role:
        raise RuntimeError("Python runtime role is invalid")
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"cannot locate Python implementation for {role}")
    return {
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
