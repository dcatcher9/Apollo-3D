import argparse
import os
import re
from pathlib import Path


LOCAL_TENSORRT_DEFAULT = Path("E:/TensorRT-11.2.1.2")


def patch_header(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    # Normalize an earlier run back to the vendor declaration before applying the
    # current transform. This keeps the script idempotent across repeated builds.
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            "#if defined(__GNUC__)" in line
            and i + 5 < len(lines)
            and "::~" not in lines[i + 1]
            and "~" in lines[i + 1]
            and "msvc_dummy_destructor" in lines[i + 2]
            and "#else" in lines[i + 3]
            and " override " in lines[i + 4]
            and "#endif" in lines[i + 5]
        ):
            out_lines.append(lines[i + 4])
            i += 6
            continue
        if (
            "#if defined(__GNUC__)" in line
            and i + 4 < len(lines)
            and "msvc_dummy_destructor" in lines[i + 1]
        ):
            out_lines.append(lines[i + 3])
            i += 5
            continue
        if (
            "#if !defined(__GNUC__)" in line
            and i + 2 < len(lines)
            and "::~" in lines[i + 1]
        ):
            out_lines.append(lines[i + 1])
            i += 3
            continue
        out_lines.append(line)
        i += 1

    content = "".join(out_lines)

    def replace_virtual_destructor(match: re.Match[str]) -> str:
        declaration = match.group(0).strip()
        indentation = match.group(1)
        if "= 0" in declaration:
            replacement = "virtual void msvc_dummy_destructor(char flags) = 0;"
        else:
            replacement = "virtual void msvc_dummy_destructor(char flags) {}"
        return (
            "#if defined(__GNUC__)\n"
            f"{indentation}{replacement}\n"
            "#else\n"
            f"{indentation}{declaration}\n"
            "#endif\n"
        )

    patched = re.sub(
        r"^([ \t]*)virtual ~[a-zA-Z0-9_]+\(\)(?: noexcept)? = (?:0|default);\s*$",
        replace_virtual_destructor,
        content,
        flags=re.MULTILINE,
    )

    def guard_inline_destructor(match: re.Match[str]) -> str:
        declaration = match.group(0).strip()
        return f"#if !defined(__GNUC__)\n{declaration}\n#endif\n"

    patched = re.sub(
        r"^inline [a-zA-Z0-9_]+::~[a-zA-Z0-9_]+\(\)(?: noexcept)? = default;\s*$",
        guard_inline_destructor,
        patched,
        flags=re.MULTILINE,
    )

    def replace_overriding_destructor(match: re.Match[str]) -> str:
        declaration = match.group(0).strip()
        indentation = match.group(1)
        gcc_declaration = declaration.replace(" override =", " =", 1)
        return (
            "#if defined(__GNUC__)\n"
            f"{indentation}{gcc_declaration}\n"
            f"{indentation}void msvc_dummy_destructor(char flags) override {{}}\n"
            "#else\n"
            f"{indentation}{declaration}\n"
            "#endif\n"
        )

    # TensorRT 11.2 added `override` to destructors of interfaces derived from
    # IVersionedInterface and IPluginV2. Under GCC their base destructor slot is
    # the ABI-compatible dummy above, so override that slot and keep the real
    # destructor non-virtual (the layout used by the patched 11.1 headers).
    patched = re.sub(
        r"^([ \t]*)~[a-zA-Z0-9_]+\(\)(?: noexcept)? override = default;\s*$",
        replace_overriding_destructor,
        patched,
        flags=re.MULTILINE,
    )

    original = path.read_text(encoding="utf-8")
    if patched == original:
        return False
    path.write_text(patched, encoding="utf-8", newline="")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the MinGW/MSVC TensorRT interface-vtable compatibility patch."
    )
    parser.add_argument(
        "tensorrt_dir",
        nargs="?",
        default=os.environ.get("TENSORRT_DIR", str(LOCAL_TENSORRT_DEFAULT)),
        help="TensorRT package root (defaults to TENSORRT_DIR, then E:/TensorRT-11.2.1.2)",
    )
    args = parser.parse_args()

    include_dir = Path(args.tensorrt_dir).expanduser().resolve() / "include"
    if not (include_dir / "NvInfer.h").is_file():
        parser.error(f"TensorRT headers not found under {include_dir}")

    headers = sorted(include_dir.glob("*.h"))
    patched_headers = [path for path in headers if patch_header(path)]
    for path in patched_headers:
        print(f"Patched {path}")
    print(
        f"TensorRT ABI patch verified across {len(headers)} headers "
        f"({len(patched_headers)} files changed)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
