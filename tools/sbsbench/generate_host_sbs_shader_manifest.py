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
EXPECTED_GROUP_KEYS = {
    "name", "description", "specs", "source_closure_sha256",
}
EXPECTED_GROUP_NAMES = (
    "preprocess",
    "parallax_v2_producer",
    "parallax_v2_coordinate_diagnostic",
    "near_identical_detector",
    "gpu_trace",
    "parallax_v2_live_renderer",
    "parallax_v2_p010_y",
    "sbs_flat_fallback",
    "parallax_v2_live_diagnostic",
)
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
        _require_exact_keys(group, EXPECTED_GROUP_KEYS, f"manifest.groups[{index}]")
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
        if not isinstance(group["source_closure_sha256"], str) or \
                SHA256.fullmatch(group["source_closure_sha256"]) is None:
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


def spec_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {spec["name"]: spec for spec in manifest["specs"]}


def closure_group(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [group for group in manifest["groups"] if group["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"closure group {name!r} must appear exactly once")
    return matches[0]


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
    return hashlib.sha256(canonical).hexdigest()


def computed_group_pins(
        manifest: dict[str, Any], shader_root: Path = SHADER_ROOT) -> dict[str, str]:
    return {
        group["name"]: shader_source_closure_sha256(
            shader_root,
            group_specs(manifest, group["name"]),
            shader_compile_flags=manifest["shader_compile_flags"],
        )
        for group in manifest["groups"]
    }


def validate_group_pins(
        manifest: dict[str, Any], shader_root: Path = SHADER_ROOT) -> dict[str, str]:
    computed = computed_group_pins(manifest, shader_root)
    for group in manifest["groups"]:
        expected = computed[group["name"]]
        if group["source_closure_sha256"] != expected:
            raise ValueError(
                f"{group['name']} source closure pin is stale: expected {expected}, "
                f"found {group['source_closure_sha256']}"
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
        "#include <span>",
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
        lines.extend([
            "  };",
            f"  inline constexpr std::string_view {group['name']}_source_closure_sha256 =",
            f"    {json.dumps(group['source_closure_sha256'])};",
            "",
        ])
    lines.extend([
        "  struct closure_group_view {",
        "    std::string_view name;",
        "    std::span<const shader_spec> specs;",
        "    std::string_view source_closure_sha256;",
        "  };",
        "",
    ])
    for group in manifest["groups"]:
        name = group["name"]
        lines.extend([
            f"  inline constexpr closure_group_view {name}_closure_group {{",
            f"    {json.dumps(name)},",
            f"    std::span<const shader_spec> {{{name}_specs}},",
            f"    {name}_source_closure_sha256,",
            "  };",
        ])
    lines.extend([
        "",
        "  inline constexpr std::array closure_groups {",
    ])
    lines.extend(f"    {group['name']}_closure_group," for group in manifest["groups"])
    lines.extend([
        "  };",
        "}  // namespace models::host_sbs_shader_cache",
        "",
    ])
    return "\n".join(lines)


def render_python(manifest: dict[str, Any]) -> str:
    lines = [
        '"""Generated Host SBS shader closure registry; do not edit by hand."""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "from typing import Dict, Tuple",
        "",
        f"MANIFEST_SCHEMA = {manifest['schema']}",
        f"SOURCE_CLOSURE_SCHEMA = {manifest['source_closure_schema']}",
        f"SHADER_COMPILE_FLAGS = 0x{manifest['shader_compile_flags']:08X}",
        f"SOURCE_MACRO_COUNT = {manifest['source_macro_count']}",
        "",
        "@dataclass(frozen=True)",
        "class ShaderSpec:",
        "    source_file: str",
        "    source_entrypoint: str",
        "    source_target: str",
        "",
        "@dataclass(frozen=True)",
        "class ClosureGroup:",
        "    name: str",
        "    description: str",
        "    specs: Tuple[ShaderSpec, ...]",
        "    source_closure_sha256: str",
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
        upper = _upper_identifier(group["name"])
        lines.extend([
            f"{upper} = ClosureGroup(",
            f"    name={json.dumps(group['name'])},",
            f"    description={json.dumps(group['description'])},",
            "    specs=(",
        ])
        lines.extend(f"        {_upper_identifier(member)}," for member in group["specs"])
        lines.extend([
            "    ),",
            f"    source_closure_sha256={json.dumps(group['source_closure_sha256'])},",
            ")",
            "",
        ])
    lines.append("CLOSURE_GROUPS: Dict[str, ClosureGroup] = {")
    lines.extend(
        f"    {json.dumps(group['name'])}: {_upper_identifier(group['name'])},"
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
        for group in manifest["groups"]
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


def _write_or_check(path: Path, rendered: str, check: bool) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == rendered:
        return True
    if check:
        print(f"stale generated Host SBS shader manifest view: {path}", file=sys.stderr)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8", newline="\n")
    return True


def _update_pins(manifest: dict[str, Any]) -> None:
    computed = computed_group_pins(manifest)
    for group in manifest["groups"]:
        group["source_closure_sha256"] = computed[group["name"]]
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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
        help="refresh every closure pin from the current shader source graph",
    )
    args = parser.parse_args(argv)
    if args.check and args.update_source_closure_pins:
        parser.error("--check and --update-source-closure-pins are mutually exclusive")
    try:
        return 0 if generate(args.check, args.update_source_closure_pins) else 1
    except ValueError as exc:
        print(f"Host SBS shader manifest error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
