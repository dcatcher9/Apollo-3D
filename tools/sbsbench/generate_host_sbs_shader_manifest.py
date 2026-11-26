#!/usr/bin/env python3
"""Generate the cross-language Host SBS shader closure registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT / "tools" / "sbsbench" / "contracts" /
    "host-sbs-shader-closures-v1.json"
)
CPP_TARGET = ROOT / "src" / "generated" / "host_sbs_shader_manifest.h"
PYTHON_TARGET = ROOT / "tools" / "sbsbench" / "host_sbs_shader_manifest.py"
DOC_TARGET = ROOT / "docs" / "host-sbs.md"
DVC_MANIFEST = (
    ROOT / "tools" / "sbsbench" / "contracts" / "depth-coordinate-v2-v1.json"
)
SHADER_ROOT = (
    ROOT / "src_assets" / "windows" / "assets" / "shaders" / "directx"
)

DOC_BEGIN = "<!-- BEGIN GENERATED HOST SBS SHADER CLOSURES -->"
DOC_END = "<!-- END GENERATED HOST SBS SHADER CLOSURES -->"
SOURCE_CLOSURE_DOMAIN = b"apollo-host-sbs-source-closure-v2\n"
EXPECTED_TOP_LEVEL_KEYS = {
    "schema", "source_closure_schema", "shader_compile_flags",
    "source_macro_count", "specs", "groups",
}
EXPECTED_SPEC_KEYS = {
    "name", "source_file", "source_entrypoint", "source_target",
}
EXPECTED_GROUP_KEYS = {"name", "description", "specs"}
OPTIONAL_GROUP_KEYS = {"root"}
EXPECTED_GROUP_NAMES = (
    "preprocess",
    "parallax_v2_producer",
    "parallax_v2_replay_resolve",
    "parallax_v2_coordinate_diagnostic",
    "near_identical_detector",
    "gpu_trace",
    "parallax_v2_live_renderer",
    "parallax_v2_p010_y",
    "sbs_flat_fallback",
    "parallax_v2_live_diagnostic",
    "direct_glyph_plan",
    "direct_glyph_transport",
    "direct_glyph_palette",
    "direct_glyph_prepare",
    "shared_ui_warp_completed",
    "shared_ui_warp_completed_diagnostic",
    "text_game_ui_warp",
    "direct_glyph_empty_gate",
    "direct_glyph_source_audit",
)
# Only these closures bind stored provenance: the model calibration (preprocess), the DVC
# contract and its dumps (producer), and the evaluator's renderer contract and baselines. Their
# pins stay checked in, so a source change to them is a deliberate re-pin. Every other group's
# identity is derived from the shader sources at build time.
PINNED_GROUP_NAMES = (
    "preprocess",
    "parallax_v2_producer",
    "parallax_v2_live_renderer",
)
BUILD_IDENTITY_HEADER = "host_sbs_shader_build_identity.h"
ANY_INCLUDE = re.compile(rb"^\s*#\s*include\b")
QUOTED_INCLUDE = re.compile(rb'^\s*#\s*include\s*"([^"]+)"')
SHA256 = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _identifier(value: str) -> str:
    if IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid generated identifier: {value!r}")
    return value


def _upper_identifier(value: str) -> str:
    return _identifier(value).upper()


def _python_group_identifier(value: str) -> str:
    return f"{_upper_identifier(value)}_GROUP"


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _validate_emitted_identifier_names(
        spec_names: set[str], group_names: list[str]) -> None:
    python_spec_identifiers = [_upper_identifier(name) for name in spec_names]
    python_group_identifiers = [_python_group_identifier(name) for name in group_names]
    python_spec_names = set(python_spec_identifiers)
    python_group_names = set(python_group_identifiers)
    python_reserved_names = {
        "MANIFEST_SCHEMA", "SOURCE_CLOSURE_SCHEMA", "SHADER_COMPILE_FLAGS",
        "SOURCE_MACRO_COUNT", "SHADER_SPECS", "CLOSURE_GROUPS",
    }
    python_collisions = (
        _duplicates(python_spec_identifiers) |
        _duplicates(python_group_identifiers) |
        (python_spec_names & python_group_names) |
        ((python_spec_names | python_group_names) & python_reserved_names)
    )
    if python_collisions:
        raise ValueError(
            "generated Python identifiers collide: " +
            ", ".join(sorted(python_collisions))
        )

    cpp_spec_names = set(spec_names)
    cpp_group_spec_names = {f"{name}_specs" for name in group_names}
    cpp_group_pin_names = {
        f"{name}_source_closure_sha256" for name in group_names
    } | {f"{name}_root" for name in group_names}
    cpp_reserved_names = {
        "source_closure_schema", "shader_compile_flags", "source_macro_count",
        "shader_spec",
    }
    cpp_collisions = (
        (cpp_spec_names & cpp_group_spec_names) |
        (cpp_spec_names & cpp_group_pin_names) |
        (cpp_group_spec_names & cpp_group_pin_names) |
        ((cpp_spec_names | cpp_group_spec_names | cpp_group_pin_names) &
         cpp_reserved_names)
    )
    if cpp_collisions:
        raise ValueError(
            "generated C++ identifiers collide: " +
            ", ".join(sorted(cpp_collisions))
        )


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Host SBS shader manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("Host SBS shader manifest must be a JSON object")
    validate_manifest(value)
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    _require_exact_keys(manifest, EXPECTED_TOP_LEVEL_KEYS, "manifest")
    if manifest["schema"] != 1:
        raise ValueError("unsupported Host SBS shader manifest schema")
    if manifest["source_closure_schema"] != 2:
        raise ValueError("Host SBS shader source-closure schema must remain 2")
    if manifest["shader_compile_flags"] != 0x00008800:
        raise ValueError("Host SBS shader compile flags must remain strictness+O3")
    if manifest["source_macro_count"] != 0:
        raise ValueError("Host SBS authenticated shader closures do not admit macros")

    specs = manifest["specs"]
    if not isinstance(specs, list) or not specs:
        raise ValueError("manifest.specs must be a non-empty array")
    spec_names: set[str] = set()
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"manifest.specs[{index}] must be an object")
        _require_exact_keys(spec, EXPECTED_SPEC_KEYS, f"manifest.specs[{index}]")
        name = spec["name"]
        if not isinstance(name, str):
            raise ValueError(f"manifest.specs[{index}].name must be a string")
        _identifier(name)
        if name in spec_names:
            raise ValueError(f"duplicate shader spec name: {name}")
        spec_names.add(name)
        for key in ("source_file", "source_entrypoint", "source_target"):
            if not isinstance(spec[key], str) or not spec[key]:
                raise ValueError(f"manifest.specs[{index}].{key} must be non-empty")
        source_file = Path(spec["source_file"])
        if source_file.is_absolute() or ".." in source_file.parts:
            raise ValueError(f"shader source path escapes the shader root: {source_file}")

    groups = manifest["groups"]
    if not isinstance(groups, list):
        raise ValueError("manifest.groups must be an array")
    group_names: list[str] = []
    referenced_specs: set[str] = set()
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            raise ValueError(f"manifest.groups[{index}] must be an object")
        pinned = group.get("name") in PINNED_GROUP_NAMES
        expected_keys = (
            EXPECTED_GROUP_KEYS | (OPTIONAL_GROUP_KEYS & set(group)) |
            ({"source_closure_sha256"} if pinned else set())
        )
        _require_exact_keys(group, expected_keys, f"manifest.groups[{index}]")
        name = group["name"]
        if not isinstance(name, str):
            raise ValueError(f"manifest.groups[{index}].name must be a string")
        _identifier(name)
        group_names.append(name)
        if not isinstance(group["description"], str) or not group["description"]:
            raise ValueError(f"manifest.groups[{index}].description must be non-empty")
        members = group["specs"]
        if not isinstance(members, list) or not members:
            raise ValueError(f"manifest.groups[{index}].specs must be non-empty")
        if len(members) != len(set(members)):
            raise ValueError(f"manifest.groups[{index}].specs contains duplicates")
        unknown = set(members) - spec_names
        if unknown:
            raise ValueError(f"manifest.groups[{index}] references unknown specs: {sorted(unknown)}")
        referenced_specs.update(members)
        if "root" in group:
            root = group["root"]
            if not isinstance(root, str) or not root or Path(root).is_absolute() or \
                    ".." in Path(root).parts:
                raise ValueError(
                    f"manifest.groups[{index}].root must be a shader-root subdirectory"
                )
        if pinned and (not isinstance(group["source_closure_sha256"], str) or
                       SHA256.fullmatch(group["source_closure_sha256"]) is None):
            raise ValueError(
                f"manifest.groups[{index}].source_closure_sha256 must be lowercase SHA-256"
            )
    if tuple(group_names) != EXPECTED_GROUP_NAMES:
        raise ValueError(
            "manifest closure groups must remain complete and in canonical order: "
            f"expected {EXPECTED_GROUP_NAMES}, found {tuple(group_names)}"
        )
    if referenced_specs != spec_names:
        raise ValueError(f"unreferenced shader specs: {sorted(spec_names - referenced_specs)}")
    _validate_emitted_identifier_names(spec_names, group_names)


def spec_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {spec["name"]: spec for spec in manifest["specs"]}


def closure_group(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [group for group in manifest["groups"] if group["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"closure group {name!r} must appear exactly once")
    return matches[0]


def is_pinned(group: dict[str, Any]) -> bool:
    return group["name"] in PINNED_GROUP_NAMES


def group_root(
        manifest: dict[str, Any], name: str, shader_root: Path = SHADER_ROOT) -> Path:
    group = closure_group(manifest, name)
    return shader_root / group["root"] if "root" in group else shader_root


def group_specs(
        manifest: dict[str, Any], name: str) -> tuple[tuple[str, str, str], ...]:
    specs = spec_index(manifest)
    return tuple(
        (
            specs[member]["source_file"],
            specs[member]["source_entrypoint"],
            specs[member]["source_target"],
        )
        for member in closure_group(manifest, name)["specs"]
    )


def _append_length_prefixed(output: bytearray, value: bytes) -> None:
    output.extend(len(value).to_bytes(8, "little"))
    output.extend(value)


def shader_source_closure_sha256(
        root: Path = SHADER_ROOT,
        specs: tuple[tuple[str, str, str], ...] | None = None,
        *,
        shader_compile_flags: int = 0x00008800) -> str:
    """Hash ordered roots plus the exact reachable quoted-include source closure."""

    return shader_source_closure(root, specs, shader_compile_flags=shader_compile_flags)[0]


def shader_source_closure(
        root: Path = SHADER_ROOT,
        specs: tuple[tuple[str, str, str], ...] | None = None,
        *,
        shader_compile_flags: int = 0x00008800) -> tuple[str, tuple[Path, ...]]:
    """Return the source-closure digest and every source file it read."""

    if specs is None:
        specs = group_specs(load_manifest(), "preprocess")
    canonical_root = root.resolve(strict=True)
    sources: dict[str, bytes] = {}
    include_edges: dict[tuple[str, str], str] = {}
    visited: set[Path] = set()

    def collect(requested: Path) -> None:
        path = requested.resolve(strict=True)
        try:
            relative = path.relative_to(canonical_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"shader dependency escapes source root: {path}") from exc
        if path in visited:
            return
        visited.add(path)
        source = path.read_bytes()
        sources[relative] = source
        normalized = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        for line in normalized.split(b"\n"):
            match = QUOTED_INCLUDE.search(line)
            if match is None:
                if ANY_INCLUDE.search(line) is not None:
                    raise ValueError(
                        f"shader uses an unauthenticated non-quoted include: {relative}"
                    )
                continue
            include_text = match.group(1).decode("utf-8")
            candidate = path.parent / Path(include_text)
            if not candidate.exists():
                candidate = canonical_root / Path(include_text)
            resolved = candidate.resolve(strict=True)
            try:
                child = resolved.relative_to(canonical_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"shader dependency escapes source root: {resolved}") from exc
            edge = (relative, include_text)
            previous = include_edges.setdefault(edge, child)
            if previous != child:
                raise ValueError(f"shader include edge is ambiguous: {edge!r}")
            collect(resolved)

    for filename, entrypoint, target in specs:
        if not filename or not entrypoint or not target:
            raise ValueError("shader source closure specs must be non-empty")
        collect(canonical_root / filename)

    canonical = bytearray(SOURCE_CLOSURE_DOMAIN)
    canonical.extend(b"C")
    canonical.extend(shader_compile_flags.to_bytes(8, "little"))
    _append_length_prefixed(canonical, b"macros:none")
    for spec in specs:
        canonical.extend(b"S")
        for value in spec:
            _append_length_prefixed(canonical, value.encode("utf-8"))
    for (parent, include), child in sorted(include_edges.items()):
        canonical.extend(b"I")
        _append_length_prefixed(canonical, parent.encode("utf-8"))
        _append_length_prefixed(canonical, include.encode("utf-8"))
        _append_length_prefixed(canonical, child.encode("utf-8"))
    for path in sorted(sources):
        canonical.extend(b"F")
        _append_length_prefixed(canonical, path.encode("utf-8"))
        normalized = sources[path].replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _append_length_prefixed(canonical, normalized)
    return hashlib.sha256(canonical).hexdigest(), tuple(sorted(visited))


def computed_group_pins(
        manifest: dict[str, Any], shader_root: Path = SHADER_ROOT,
        names: Iterable[str] | None = None) -> dict[str, str]:
    """Closures of the named groups, or of every group, computed from the shader sources."""

    selected = None if names is None else set(names)
    return {
        group["name"]: shader_source_closure_sha256(
            group_root(manifest, group["name"], shader_root),
            group_specs(manifest, group["name"]),
            shader_compile_flags=manifest["shader_compile_flags"],
        )
        for group in manifest["groups"]
        if selected is None or group["name"] in selected
    }


def validate_group_pins(
        manifest: dict[str, Any], shader_root: Path = SHADER_ROOT) -> dict[str, str]:
    computed = computed_group_pins(manifest, shader_root, PINNED_GROUP_NAMES)
    for name, expected in computed.items():
        pin = closure_group(manifest, name)["source_closure_sha256"]
        if pin != expected:
            raise ValueError(
                f"{name} source closure pin is stale: expected {expected}, found {pin}"
            )
    return computed


def render_cpp(manifest: dict[str, Any]) -> str:
    lines = [
        "// Generated by tools/sbsbench/generate_host_sbs_shader_manifest.py.",
        "// Do not edit by hand; edit contracts/host-sbs-shader-closures-v1.json.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "#include <string_view>",
        "",
        "namespace models::host_sbs_shader_cache {",
        f"  inline constexpr std::uint32_t source_closure_schema = {manifest['source_closure_schema']}u;",
        f"  inline constexpr std::uint32_t shader_compile_flags = 0x{manifest['shader_compile_flags']:08X}u;",
        f"  inline constexpr std::uint32_t source_macro_count = {manifest['source_macro_count']}u;",
        "",
        "  struct shader_spec {",
        "    std::string_view filename;",
        "    std::string_view entrypoint = \"main\";",
        "    std::string_view target = \"cs_5_0\";",
        "    constexpr bool operator==(const shader_spec &) const = default;",
        "  };",
        "",
    ]
    for spec in manifest["specs"]:
        lines.append(
            f"  inline constexpr shader_spec {spec['name']} "
            f"{{{json.dumps(spec['source_file'])}, "
            f"{json.dumps(spec['source_entrypoint'])}, "
            f"{json.dumps(spec['source_target'])}}};"
        )
    lines.append("")
    for group in manifest["groups"]:
        lines.append(f"  inline constexpr std::array {group['name']}_specs {{")
        lines.extend(f"    {member}," for member in group["specs"])
        lines.append("  };")
        if "root" in group:
            lines.append(
                f"  inline constexpr std::string_view {group['name']}_root = "
                f"{json.dumps(group['root'])};"
            )
        if is_pinned(group):
            lines.extend([
                f"  inline constexpr std::string_view {group['name']}_source_closure_sha256 =",
                f"    {json.dumps(group['source_closure_sha256'])};",
            ])
        lines.append("")
    lines.extend([
        "}  // namespace models::host_sbs_shader_cache",
        "",
        "// The build derives every other group's closure from the shader sources.",
        f'#include "{BUILD_IDENTITY_HEADER}"',
        "",
    ])
    return "\n".join(lines)


def render_build_identity(manifest: dict[str, Any], closures: dict[str, str]) -> str:
    lines = [
        "// Generated at build time by tools/sbsbench/generate_host_sbs_shader_manifest.py",
        "// from the shader sources of the closure groups that have no checked-in pin.",
        "#pragma once",
        "",
        "#include <string_view>",
        "",
        "namespace models::host_sbs_shader_cache {",
    ]
    for group in manifest["groups"]:
        if not is_pinned(group):
            lines.extend([
                f"  inline constexpr std::string_view {group['name']}_source_closure_sha256 =",
                f"    {json.dumps(closures[group['name']])};",
            ])
    lines.extend([
        "}  // namespace models::host_sbs_shader_cache",
        "",
    ])
    return "\n".join(lines)


def _depfile_path(path: Path) -> str:
    return path.as_posix().replace("$", "$$").replace("#", "\\#").replace(" ", "\\ ")


def render_depfile(target: Path, sources: Iterable[Path]) -> str:
    return "".join(
        [f"{_depfile_path(target)}:"] +
        [f" \\\n  {_depfile_path(source)}" for source in sources] +
        ["\n"]
    )


def write_build_identity(
        header: Path, depfile: Path, shader_root: Path = SHADER_ROOT) -> None:
    """Write the unpinned group closures and the shader sources they read."""

    manifest = load_manifest()
    closures: dict[str, str] = {}
    sources: set[Path] = set()
    for group in manifest["groups"]:
        if is_pinned(group):
            continue
        closures[group["name"]], paths = shader_source_closure(
            group_root(manifest, group["name"], shader_root),
            group_specs(manifest, group["name"]),
            shader_compile_flags=manifest["shader_compile_flags"],
        )
        sources.update(paths)
    _write_utf8_if_changed(header, render_build_identity(manifest, closures))
    _write_utf8_if_changed(depfile, render_depfile(header, sorted(sources)))


def render_python(manifest: dict[str, Any]) -> str:
    lines = [
        '"""Generated Host SBS shader closure registry; do not edit by hand."""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "from functools import lru_cache",
        "from typing import Dict, Optional, Tuple",
        "",
        f"MANIFEST_SCHEMA = {manifest['schema']}",
        f"SOURCE_CLOSURE_SCHEMA = {manifest['source_closure_schema']}",
        f"SHADER_COMPILE_FLAGS = 0x{manifest['shader_compile_flags']:08X}",
        f"SOURCE_MACRO_COUNT = {manifest['source_macro_count']}",
        "",
        "",
        "@dataclass(frozen=True)",
        "class ShaderSpec:",
        "    source_file: str",
        "    source_entrypoint: str",
        "    source_target: str",
        "",
        "",
        "@dataclass(frozen=True)",
        "class ClosureGroup:",
        "    name: str",
        "    description: str",
        "    specs: Tuple[ShaderSpec, ...]",
        "    root: str",
        "    pinned_source_closure_sha256: Optional[str]",
        "",
        "    @property",
        "    def source_closure_sha256(self) -> str:",
        '        """The checked-in pin, or the closure of the current shader sources."""',
        "        if self.pinned_source_closure_sha256 is not None:",
        "            return self.pinned_source_closure_sha256",
        "        return _source_closure(self)",
        "",
        "",
        "@lru_cache(maxsize=None)",
        "def _source_closure(group: ClosureGroup) -> str:",
        "    try:",
        "        from . import generate_host_sbs_shader_manifest as generator",
        "    except ImportError:",
        "        import generate_host_sbs_shader_manifest as generator",
        "    return generator.shader_source_closure_sha256(",
        "        generator.SHADER_ROOT / group.root,",
        "        tuple((spec.source_file, spec.source_entrypoint, spec.source_target)",
        "              for spec in group.specs),",
        "        shader_compile_flags=SHADER_COMPILE_FLAGS,",
        "    )",
        "",
        "",
    ]
    for spec in manifest["specs"]:
        lines.extend([
            f"{_upper_identifier(spec['name'])} = ShaderSpec(",
            f"    source_file={json.dumps(spec['source_file'])},",
            f"    source_entrypoint={json.dumps(spec['source_entrypoint'])},",
            f"    source_target={json.dumps(spec['source_target'])},",
            ")",
        ])
    lines.extend(["", "SHADER_SPECS: Dict[str, ShaderSpec] = {"])
    lines.extend(
        f"    {json.dumps(spec['name'])}: {_upper_identifier(spec['name'])},"
        for spec in manifest["specs"]
    )
    lines.extend(["}", ""])
    for group in manifest["groups"]:
        upper = _python_group_identifier(group["name"])
        lines.extend([
            f"{upper} = ClosureGroup(",
            f"    name={json.dumps(group['name'])},",
            "    description=(",
            f"        {json.dumps(group['description'])}",
            "    ),",
            "    specs=(",
        ])
        lines.extend(f"        {_upper_identifier(member)}," for member in group["specs"])
        pin = json.dumps(group["source_closure_sha256"]) if is_pinned(group) else "None"
        lines.extend([
            "    ),",
            f"    root={json.dumps(group.get('root', ''))},",
            f"    pinned_source_closure_sha256={pin},",
            ")",
            "",
        ])
    lines.append("CLOSURE_GROUPS: Dict[str, ClosureGroup] = {")
    lines.extend(
        f"    {json.dumps(group['name'])}: {_python_group_identifier(group['name'])},"
        for group in manifest["groups"]
    )
    lines.extend(["}", ""])
    return "\n".join(lines)


def render_doc_block(manifest: dict[str, Any]) -> str:
    lines = [
        DOC_BEGIN,
        "<!-- Generated by tools/sbsbench/generate_host_sbs_shader_manifest.py. -->",
        "| Closure group | Ordered roots | Source-closure SHA-256 |",
        "| --- | ---: | --- |",
    ]
    lines.extend(
        f"| `{group['name']}` | {len(group['specs'])} | "
        f"`{group['source_closure_sha256']}` |"
        for group in manifest["groups"] if is_pinned(group)
    )
    lines.append(DOC_END)
    return "\n".join(lines)


def render_documentation(manifest: dict[str, Any], current: str) -> str:
    begin = current.find(DOC_BEGIN)
    end = current.find(DOC_END)
    if begin == -1 or end == -1 or end < begin:
        raise ValueError(f"{DOC_TARGET} is missing the generated shader-closure markers")
    end += len(DOC_END)
    return current[:begin] + render_doc_block(manifest) + current[end:]


def render_dvc_manifest(manifest: dict[str, Any], current: str) -> str:
    """Project the producer/preprocess closure identity into the DVC JSON contract."""

    try:
        contract = json.loads(current)
    except json.JSONDecodeError as exc:
        raise ValueError(f"cannot parse {DVC_MANIFEST}: {exc}") from exc
    if not isinstance(contract, dict):
        raise ValueError(f"{DVC_MANIFEST} must contain a JSON object")
    producer = closure_group(manifest, "parallax_v2_producer")
    specs = spec_index(manifest)
    contract["shader_implementation"] = {
        "source_closure_schema": manifest["source_closure_schema"],
        "source_compile_flags": manifest["shader_compile_flags"],
        "source_macro_count": manifest["source_macro_count"],
        "source_specs": [
            {
                "source_file": specs[name]["source_file"],
                "source_entrypoint": specs[name]["source_entrypoint"],
                "source_target": specs[name]["source_target"],
            }
            for name in producer["specs"]
        ],
        "source_closure_sha256": producer["source_closure_sha256"],
    }
    preprocess = closure_group(manifest, "preprocess")
    preprocess_spec = specs[preprocess["specs"][0]]
    calibrations = contract.get("model_calibrations")
    if not isinstance(calibrations, list) or not calibrations:
        raise ValueError(f"{DVC_MANIFEST} has no model calibrations")
    for index, calibration in enumerate(calibrations):
        if not isinstance(calibration, dict) or not isinstance(
                calibration.get("preprocess"), dict):
            raise ValueError(f"{DVC_MANIFEST} model_calibrations[{index}] is malformed")
        projection = calibration["preprocess"]
        projection["source_closure_schema"] = manifest["source_closure_schema"]
        projection["source_file"] = preprocess_spec["source_file"]
        projection["source_entrypoint"] = preprocess_spec["source_entrypoint"]
        projection["source_target"] = preprocess_spec["source_target"]
        projection["source_compile_flags"] = manifest["shader_compile_flags"]
        projection["source_macro_count"] = manifest["source_macro_count"]
        projection["source_closure_sha256"] = preprocess["source_closure_sha256"]
    return json.dumps(contract, indent=2, ensure_ascii=True) + "\n"


def _write_utf8_if_changed(path: Path, rendered: str) -> None:
    expected = rendered.encode("utf-8")
    try:
        if path.read_bytes() == expected:
            return
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def _write_or_check(path: Path, rendered: str, check: bool) -> bool:
    if check:
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == rendered:
            return True
        print(f"stale generated Host SBS shader manifest view: {path}", file=sys.stderr)
        return False
    _write_utf8_if_changed(path, rendered)
    return True


def _update_pins(manifest: dict[str, Any]) -> None:
    for name, digest in computed_group_pins(manifest, names=PINNED_GROUP_NAMES).items():
        closure_group(manifest, name)["source_closure_sha256"] = digest
    _write_utf8_if_changed(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")


def generate(check: bool, update_pins: bool) -> bool:
    manifest = load_manifest()
    if update_pins:
        _update_pins(manifest)
        manifest = load_manifest()
    validate_group_pins(manifest)
    try:
        documentation = render_documentation(
            manifest, DOC_TARGET.read_text(encoding="utf-8")
        )
        dvc_manifest = render_dvc_manifest(
            manifest, DVC_MANIFEST.read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise ValueError(f"cannot read a generated shader-manifest consumer: {exc}") from exc
    return all((
        _write_or_check(CPP_TARGET, render_cpp(manifest), check),
        _write_or_check(PYTHON_TARGET, render_python(manifest), check),
        _write_or_check(DOC_TARGET, documentation, check),
        _write_or_check(DVC_MANIFEST, dvc_manifest, check),
    ))


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if a generated view is stale")
    parser.add_argument(
        "--update-source-closure-pins",
        action="store_true",
        help="refresh the checked-in closure pins from the current shader source graph",
    )
    parser.add_argument(
        "--build-identity",
        type=Path,
        metavar="HEADER",
        help="write the build's closure header for the groups without a checked-in pin",
    )
    parser.add_argument(
        "--depfile",
        type=Path,
        help="with --build-identity, list the shader sources it read for Ninja",
    )
    args = parser.parse_args(argv)
    if args.check and args.update_source_closure_pins:
        parser.error("--check and --update-source-closure-pins are mutually exclusive")
    if (args.build_identity is None) != (args.depfile is None) or (
            args.build_identity is not None and (args.check or args.update_source_closure_pins)):
        parser.error("--build-identity needs --depfile and excludes the other modes")
    try:
        if args.build_identity is not None:
            write_build_identity(args.build_identity, args.depfile)
            return 0
        return 0 if generate(args.check, args.update_source_closure_pins) else 1
    except ValueError as exc:
        print(f"Host SBS shader manifest error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
