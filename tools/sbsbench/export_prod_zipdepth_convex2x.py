#!/usr/bin/env python3
"""Export the frozen production DAV2 + ZipDepth convex-2x TensorRT artifact.

The script intentionally writes the large ONNX and TensorRT products outside
the repository.  Source identities and the six point-profile order come from
``contracts/prod-zipdepth-convex2x-v2.json``; callers cannot silently substitute
another DAV2 graph, ZipDepth commit, or checkpoint under the production name.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

import onnx
from onnx import compose, helper, version_converter


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import prod_zipdepth_convex2x as contract_api  # noqa: E402


EXPORTER_NAME = "apollo-prod-zipdepth-convex2x-exporter"
EXPORTER_VERSION = "2"
OPSET_VERSION = 18
BRANCH_FILENAME = "zipdepth_mask_convex2x_dynamic_opset18.onnx"
FUSED_FILENAME = "prod_dav2_zipdepth_c2x_high_opset18.onnx"
PLAN_FILENAME = "prod_dav2_zipdepth_c2x_high_six_profiles.plan"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    return {
        "path": os.fspath(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def prepare_output_directory(path: Path) -> Path:
    """Create an empty output directory, refusing to reuse any prior run."""

    resolved = path.resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"output is not a directory: {resolved}")
        if any(resolved.iterdir()):
            raise ValueError(f"refusing nonempty output directory: {resolved}")
    else:
        resolved.mkdir(parents=True)
    return resolved


def require_file_hash(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    return observed


def require_artifact_identity(
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> dict[str, object]:
    observed = file_record(path)
    if observed["bytes"] != expected_bytes or observed["sha256"] != expected_sha256:
        raise ValueError(
            f"{label} is not the frozen release artifact: expected "
            f"{expected_bytes} bytes / {expected_sha256}, observed "
            f"{observed['bytes']} bytes / {observed['sha256']}"
        )
    return observed


def reconfigure_utf8_streams() -> None:
    """Allow the current PyTorch exporter to print Unicode status glyphs on Windows."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def git_identity(repository: Path, expected_commit: str) -> dict[str, object]:
    if not (repository / ".git").exists():
        raise ValueError(f"ZipDepth source is not a Git checkout: {repository}")
    head = subprocess.check_output(
        ["git", "-C", os.fspath(repository), "rev-parse", "HEAD"],
        text=True,
        encoding="utf-8",
    ).strip()
    if head != expected_commit:
        raise ValueError(
            f"ZipDepth commit mismatch: expected {expected_commit}, observed {head}"
        )
    status = subprocess.check_output(
        ["git", "-C", os.fspath(repository), "status", "--porcelain"],
        text=True,
        encoding="utf-8",
    ).strip()
    if status:
        raise ValueError("ZipDepth checkout must be clean before export")
    return {"path": os.fspath(repository.resolve()), "commit": head, "clean": True}


def shape_of(value_info: onnx.ValueInfoProto) -> list[object]:
    result: list[object] = []
    for dim in value_info.type.tensor_type.shape.dim:
        result.append(int(dim.dim_value) if dim.dim_value else dim.dim_param)
    return result


def set_shape(value_info: onnx.ValueInfoProto, dimensions: Sequence[object]) -> None:
    value_info.type.tensor_type.shape.ClearField("dim")
    for value in dimensions:
        dim = value_info.type.tensor_type.shape.dim.add()
        if isinstance(value, int):
            dim.dim_value = value
        else:
            dim.dim_param = str(value)


def rename_value(model: onnx.ModelProto, old: str, new: str) -> None:
    for collection in (
        model.graph.input,
        model.graph.output,
        model.graph.value_info,
        model.graph.initializer,
        model.graph.sparse_initializer,
    ):
        for item in collection:
            if item.name == old:
                item.name = new
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new
        for index, value in enumerate(node.output):
            if value == old:
                node.output[index] = new


def model_contract(model: onnx.ModelProto) -> dict[str, object]:
    return {
        "ir_version": model.ir_version,
        "opsets": [(item.domain, item.version) for item in model.opset_import],
        "inputs": {
            item.name: {
                "shape": shape_of(item),
                "onnx_elem_type": item.type.tensor_type.elem_type,
            }
            for item in model.graph.input
        },
        "outputs": {
            item.name: {
                "shape": shape_of(item),
                "onnx_elem_type": item.type.tensor_type.elem_type,
            }
            for item in model.graph.output
        },
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "operators": dict(sorted(Counter(node.op_type for node in model.graph.node).items())),
    }


def _require_model_boundary(
    model: onnx.ModelProto,
    expected_inputs: Sequence[str],
    expected_outputs: Sequence[str],
    label: str,
) -> None:
    inputs = tuple(item.name for item in model.graph.input)
    outputs = tuple(item.name for item in model.graph.output)
    if inputs != tuple(expected_inputs) or outputs != tuple(expected_outputs):
        raise ValueError(
            f"unexpected {label} boundary: inputs={inputs}, outputs={outputs}"
        )
    for value_info in (*model.graph.input, *model.graph.output):
        if value_info.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError(f"{label} tensor {value_info.name} must be FP32")


def deterministic_model_bytes(model: onnx.ModelProto) -> bytes:
    return model.SerializeToString(deterministic=True)


def canonicalize_export_metadata(model: onnx.ModelProto, graph_name: str) -> None:
    """Remove source-path/line metadata without touching executable graph data.

    PyTorch records the Python wrapper's absolute path and line numbers on most
    ONNX nodes.  Those annotations do not affect inference, but retaining them
    makes otherwise identical exports differ across checkouts or harmless source
    formatting changes.  Provenance is recorded in ``export_report.json``.
    """

    model.ClearField("metadata_props")
    model.doc_string = ""
    model.producer_name = EXPORTER_NAME
    model.producer_version = EXPORTER_VERSION
    model.domain = ""
    model.model_version = 1
    model.graph.name = graph_name
    model.graph.doc_string = ""
    for value_info in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        value_info.ClearField("metadata_props")
        value_info.doc_string = ""
    for node in model.graph.node:
        node.ClearField("metadata_props")
        node.doc_string = ""


def save_deterministic_model(model: onnx.ModelProto, path: Path) -> onnx.ModelProto:
    onnx.checker.check_model(model, full_check=True)
    path.write_bytes(deterministic_model_bytes(model))
    checked = onnx.load(os.fspath(path))
    onnx.checker.check_model(checked, full_check=True)
    return checked


def compose_fused_models(
    dav2_original: onnx.ModelProto,
    zip_branch_original: onnx.ModelProto,
) -> onnx.ModelProto:
    """Compose the frozen graphs behind one high-resolution public boundary.

    The host supplies only the exact 2x RGB tensor.  A typed FP32 AveragePool
    produces DAV2's original coarse input inside the graph; the untouched high
    tensor independently feeds the frozen ZipDepth mask branch.  DAV2's coarse
    prediction remains an internal edge into the convex reconstruction.
    """

    if [(item.domain, item.version) for item in dav2_original.opset_import] != [("", 14)]:
        raise ValueError("production DAV2 source must use the frozen opset-14 contract")
    _require_model_boundary(
        dav2_original,
        ("pixel_values",),
        ("predicted_depth",),
        "production DAV2",
    )
    if [(item.domain, item.version) for item in zip_branch_original.opset_import] != [
        ("", OPSET_VERSION)
    ]:
        raise ValueError(f"ZipDepth branch must use opset {OPSET_VERSION}")
    _require_model_boundary(
        zip_branch_original,
        ("zip_pixel_values", "predicted_depth"),
        ("refined_depth",),
        "ZipDepth convex branch",
    )

    dav2 = version_converter.convert_version(dav2_original, OPSET_VERSION)
    zip_branch = onnx.ModelProto()
    zip_branch.CopyFrom(zip_branch_original)

    # Retarget the existing DAV2 graph to an internal coarse tensor, then make
    # its former public input the single 2x tensor. AveragePool inherits FP32
    # from that input and has no padding, so every supported even-sized point
    # profile is an exact average over disjoint 2x2 cells.
    for node in dav2.graph.node:
        for index, value in enumerate(node.input):
            if value == "pixel_values":
                node.input[index] = "dav2_pixel_values"
    dav2.graph.input[0].name = "pixel_values"
    set_shape(dav2.graph.input[0], [1, 3, "2*height", "2*width"])
    dav2.graph.value_info.insert(
        0,
        helper.make_tensor_value_info(
            "dav2_pixel_values",
            onnx.TensorProto.FLOAT,
            [1, 3, "height", "width"],
        ),
    )
    dav2.graph.node.insert(
        0,
        helper.make_node(
            "AveragePool",
            ["pixel_values"],
            ["dav2_pixel_values"],
            name="single_high_io_average_pool_2x2",
            kernel_shape=[2, 2],
            strides=[2, 2],
            pads=[0, 0, 0, 0],
            ceil_mode=0,
            count_include_pad=0,
        ),
    )
    set_shape(dav2.graph.output[0], [1, "height", "width"])
    shared_ir = max(dav2.ir_version, zip_branch.ir_version)
    dav2.ir_version = shared_ir
    zip_branch.ir_version = shared_ir
    prefixed = compose.add_prefix(zip_branch, "zip_")
    fused = compose.merge_models(
        dav2,
        prefixed,
        io_map=[("predicted_depth", "zip_predicted_depth")],
        name="prod_dav2_plus_frozen_zipdepth_single_high_io_convex2x",
    )
    rename_value(fused, "zip_refined_depth", "refined_depth")

    # Both branches consume the same public high-resolution tensor. Do this as
    # an input alias rather than an Identity so TensorRT sees exactly one input
    # binding and can fuse/plan directly from it.
    zip_input_name = "zip_zip_pixel_values"
    for node in fused.graph.node:
        for index, value in enumerate(node.input):
            if value == zip_input_name:
                node.input[index] = "pixel_values"
    retained_inputs = [item for item in fused.graph.input if item.name != zip_input_name]
    fused.graph.ClearField("input")
    fused.graph.input.extend(retained_inputs)

    inputs = {item.name: item for item in fused.graph.input}
    outputs = {item.name: item for item in fused.graph.output}
    if set(inputs) != {"pixel_values"}:
        raise ValueError(f"unexpected fused inputs: {tuple(inputs)}")
    if set(outputs) != {"refined_depth"}:
        raise ValueError(f"unexpected fused outputs: {tuple(outputs)}")
    set_shape(inputs["pixel_values"], [1, 3, "2*height", "2*width"])
    set_shape(outputs["refined_depth"], [1, "2*height", "2*width"])

    # merge_models repeats the identical default-domain opset import. Canonicalize
    # it so repeated exports are byte-stable and downstream tools see one owner.
    fused.ClearField("opset_import")
    fused.opset_import.append(helper.make_opsetid("", OPSET_VERSION))
    canonicalize_export_metadata(
        fused, "prod_dav2_plus_frozen_zipdepth_single_high_io_convex2x"
    )
    _require_model_boundary(
        fused,
        ("pixel_values",),
        ("refined_depth",),
        "fused model",
    )
    onnx.checker.check_model(fused, full_check=True)
    return fused


def export_zipdepth_branch(
    zip_repository: Path,
    checkpoint: Path,
    output_path: Path,
    torch_site_packages: Path | None,
) -> dict[str, object]:
    """Export only ZipDepth's RGB feature/mask path plus exact convex2x."""

    if torch_site_packages is not None:
        if not torch_site_packages.is_dir():
            raise FileNotFoundError(f"missing torch site-packages: {torch_site_packages}")
        sys.path.insert(0, os.fspath(torch_site_packages.resolve()))
    import torch
    import torch.nn.functional as functional

    sys.path.insert(0, os.fspath(zip_repository.resolve()))
    from zipdepth.model.architecture import create_model
    from zipdepth.utils.model_utils import strip_state_dict_prefixes

    source = create_model(variant="base", upsample_unfold=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = strip_state_dict_prefixes(payload.get("model_state_dict", payload))
    source.load_state_dict(state, strict=True)
    source.float().eval().fuse_for_inference()
    convex = source.decoder.convex_up
    if (
        int(convex.scale) != 2
        or float(convex.temperature) != 1.0
        or not bool(convex.use_unfold)
    ):
        raise ValueError("checkpoint architecture does not match frozen convex2x contract")

    class DynamicZipConvex(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.encoder
            self.decoder = model.decoder
            neighbor_kernel = torch.zeros(9, 1, 3, 3, dtype=torch.float32)
            for row in range(3):
                for column in range(3):
                    neighbor_kernel[row * 3 + column, 0, row, column] = 1.0
            self.register_buffer("neighbor_kernel", neighbor_kernel)
            pixel_shuffle_kernel = torch.zeros(4, 1, 2, 2, dtype=torch.float32)
            pixel_shuffle_kernel[0, 0, 0, 0] = 1.0
            pixel_shuffle_kernel[1, 0, 0, 1] = 1.0
            pixel_shuffle_kernel[2, 0, 1, 0] = 1.0
            pixel_shuffle_kernel[3, 0, 1, 1] = 1.0
            self.register_buffer("pixel_shuffle_kernel", pixel_shuffle_kernel)

        def forward(self, pixel_values, depth):
            # The host supplies ImageNet-normalized guidance. Bypassing the
            # ZipDepth top-level normalizer permits reuse of rgb_to_nchw_cs.
            s_half, features = self.encoder(pixel_values)
            c1, c2, c3, c4 = features
            f4 = self.decoder.proj4(c4)
            f3 = self.decoder.fuse3(c3, f4)
            f2 = self.decoder.fuse2(c2, f3)
            f1 = self.decoder.fuse1(c1, f2)
            f_half = self.decoder.fuse_half(s_half, f1)
            logits = self.decoder.convex_up.mask_pred(f_half)

            batch, height, width = depth.shape
            weights = torch.softmax(
                logits.reshape(batch, 9, 4, height, width), dim=1
            )
            padded = functional.pad(
                depth.unsqueeze(1), (1, 1, 1, 1), mode="replicate"
            )
            # These one-hot convolutions are exact export forms of 3x3 unfold
            # and row-major pixel_shuffle(2), not learned replacements.
            neighbors = functional.conv2d(padded, self.neighbor_kernel).unsqueeze(2)
            subpixels = (weights * neighbors).sum(dim=1)
            refined = functional.conv_transpose2d(
                subpixels, self.pixel_shuffle_kernel, stride=2
            )
            return functional.relu(refined[:, 0])

    wrapper = DynamicZipConvex(source).float().eval()
    example_height, example_width = 434, 770
    guidance = torch.zeros(
        1, 3, 2 * example_height, 2 * example_width, dtype=torch.float32
    )
    depth = torch.zeros(1, example_height, example_width, dtype=torch.float32)
    height = torch.export.Dim("height", min=434, max=1036)
    width = torch.export.Dim("width", min=434, max=1036)
    program = torch.onnx.export(
        wrapper,
        (guidance, depth),
        os.fspath(output_path),
        input_names=["zip_pixel_values", "predicted_depth"],
        output_names=["refined_depth"],
        opset_version=OPSET_VERSION,
        dynamo=True,
        external_data=False,
        optimize=True,
        dynamic_shapes=(
            {2: 2 * height, 3: 2 * width},
            {1: height, 2: width},
        ),
    )
    del program
    branch = onnx.load(os.fspath(output_path))
    _require_model_boundary(
        branch,
        ("zip_pixel_values", "predicted_depth"),
        ("refined_depth",),
        "exported ZipDepth branch",
    )
    expected_shapes = {
        "zip_pixel_values": [1, 3, "2*height", "2*width"],
        "predicted_depth": [1, "height", "width"],
        "refined_depth": [1, "2*height", "2*width"],
    }
    for value_info in (*branch.graph.input, *branch.graph.output):
        if shape_of(value_info) != expected_shapes[value_info.name]:
            raise ValueError(
                f"unexpected exported shape for {value_info.name}: {shape_of(value_info)}"
            )
    canonicalize_export_metadata(branch, "zipdepth_mask_convex2x_dynamic")
    branch = save_deterministic_model(branch, output_path)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "contract": model_contract(branch),
        "artifact": file_record(output_path),
    }


def point_profile_build_arguments(
    trtexec: Path,
    onnx_path: Path,
    plan_path: Path,
    builder_optimization_level: int,
    high_shapes: Sequence[contract_api.Shape] | None = None,
) -> list[str]:
    if builder_optimization_level < 0 or builder_optimization_level > 5:
        raise ValueError("TensorRT builder optimization level must be in [0,5]")
    shapes = tuple(high_shapes or contract_api.supported_high_shapes())
    arguments = [
        os.fspath(trtexec),
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        "--skipInference",
        f"--builderOptimizationLevel={builder_optimization_level}",
        "--memPoolSize=workspace:4096",
    ]
    for index, shape in enumerate(shapes):
        binding_shapes = f"pixel_values:1x3x{shape.height}x{shape.width}"
        arguments.extend(
            [
                f"--profile={index}",
                f"--minShapes={binding_shapes}",
                f"--optShapes={binding_shapes}",
                f"--maxShapes={binding_shapes}",
            ]
        )
    return arguments


def build_tensorrt_engine(
    trtexec: Path,
    onnx_path: Path,
    plan_path: Path,
    log_path: Path,
    builder_optimization_level: int,
) -> dict[str, object]:
    if not trtexec.is_file():
        raise FileNotFoundError(f"missing trtexec: {trtexec}")
    command = point_profile_build_arguments(
        trtexec, onnx_path, plan_path, builder_optimization_level
    )
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=os.fspath(plan_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not plan_path.is_file() or plan_path.stat().st_size == 0:
        raise RuntimeError(
            f"TensorRT build failed with exit {completed.returncode}; see {log_path}\n"
            + completed.stdout[-4000:]
        )
    return {
        "builder_optimization_level": builder_optimization_level,
        "seconds": elapsed,
        "command": command,
        "profiles": [
            {
                "index": index,
                "high_input_output_wh": [shape.width, shape.height],
                "internal_dav2_wh": [shape.width // 2, shape.height // 2],
            }
            for index, shape in enumerate(contract_api.supported_high_shapes())
        ],
        "log": file_record(log_path),
        "plan": file_record(plan_path),
    }


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dav2-onnx", type=Path, required=True)
    parser.add_argument("--zipdepth-repo", type=Path, required=True)
    parser.add_argument("--zipdepth-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--torch-site-packages",
        type=Path,
        help="optional site-packages directory containing the required PyTorch build",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trtexec", type=Path)
    parser.add_argument("--skip-engine", action="store_true")
    parser.add_argument("--builder-optimization-level", type=int, default=5)
    args = parser.parse_args(argv)
    if args.skip_engine and args.trtexec is not None:
        parser.error("--trtexec and --skip-engine are mutually exclusive")
    if not args.skip_engine and args.trtexec is None:
        parser.error("--trtexec is required unless --skip-engine is selected")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure_utf8_streams()
    args = parse_arguments(argv)
    output = prepare_output_directory(args.output)
    contract = contract_api.load_contract()
    source_contract = contract["sources"]
    export_contract = contract["export"]
    dav2_contract = source_contract["dav2"]
    zip_contract = source_contract["zipdepth"]
    fused_contract = source_contract["fused_onnx"]
    if int(export_contract["opset"]) != OPSET_VERSION:
        raise ValueError("exporter and frozen contract disagree about ONNX opset")
    release_builder_level = int(export_contract["release_builder_optimization_level"])
    if not args.skip_engine and args.builder_optimization_level != release_builder_level:
        raise ValueError(
            f"release TensorRT build requires builder optimization level {release_builder_level}"
        )

    dav2_path = args.dav2_onnx.resolve()
    zip_repository = args.zipdepth_repo.resolve()
    checkpoint = args.zipdepth_checkpoint.resolve()
    require_file_hash(dav2_path, str(dav2_contract["onnx_sha256"]), "production DAV2 ONNX")
    if checkpoint.name != str(zip_contract["checkpoint"]):
        raise ValueError(
            f"ZipDepth checkpoint filename must be {zip_contract['checkpoint']}: {checkpoint}"
        )
    require_file_hash(
        checkpoint, str(zip_contract["checkpoint_sha256"]), "ZipDepth checkpoint"
    )
    repository_identity = git_identity(
        zip_repository, str(zip_contract["repository_commit"])
    )

    report: dict[str, object] = {
        "schema": 1,
        "exporter": {
            "name": EXPORTER_NAME,
            "version": EXPORTER_VERSION,
            "script": file_record(Path(__file__)),
        },
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "onnx": onnx.__version__,
        },
        "contract": file_record(contract_api.CONTRACT_PATH),
        "sources": {
            "dav2": file_record(dav2_path),
            "zipdepth_repository": repository_identity,
            "zipdepth_checkpoint": file_record(checkpoint),
        },
    }
    license_path = zip_repository / "LICENSE"
    if license_path.is_file():
        report["sources"]["zipdepth_license"] = file_record(license_path)

    branch_path = output / BRANCH_FILENAME
    report["zipdepth_branch"] = export_zipdepth_branch(
        zip_repository,
        checkpoint,
        branch_path,
        args.torch_site_packages.resolve() if args.torch_site_packages else None,
    )
    report["zipdepth_branch"]["frozen_identity"] = require_artifact_identity(
        branch_path,
        int(export_contract["zipdepth_branch_onnx_bytes"]),
        str(export_contract["zipdepth_branch_onnx_sha256"]),
        "ZipDepth branch ONNX",
    )
    dav2_model = onnx.load(os.fspath(dav2_path))
    branch_model = onnx.load(os.fspath(branch_path))
    fused = compose_fused_models(dav2_model, branch_model)
    fused_path = output / FUSED_FILENAME
    fused = save_deterministic_model(fused, fused_path)
    report["fused"] = {
        "artifact": file_record(fused_path),
        "contract": model_contract(fused),
    }
    report["fused"]["frozen_identity"] = require_artifact_identity(
        fused_path,
        int(fused_contract["bytes"]),
        str(fused_contract["sha256"]),
        "fused ONNX",
    )
    partial_path = output / "export_report.partial.json"
    partial_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.skip_engine:
        report["tensorrt"] = {"skipped": True}
    else:
        plan_path = output / PLAN_FILENAME
        report["tensorrt"] = build_tensorrt_engine(
            args.trtexec.resolve(),
            fused_path,
            plan_path,
            output / "trtexec_build.log",
            args.builder_optimization_level,
        )
    report_path = output / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "schema": 1,
        "report": file_record(report_path),
        "branch_onnx": file_record(branch_path),
        "fused_onnx": file_record(fused_path),
    }
    if not args.skip_engine:
        manifest["tensorrt_plan"] = file_record(output / PLAN_FILENAME)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
