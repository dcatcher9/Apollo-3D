#!/usr/bin/env python3
"""Graph edits for a fine-tuned DAV2 + ZipDepth convex-2x composite.

The production composite feeds DAV2 an exact 2x2 average of the high-resolution input and uses the
stock DAV2 and ZipDepth weights. A tuned composite keeps the same topology and public tensors and
changes four things, each of which is one function here:

  swap_head_initializers        tuned DAV2 decoder-head weights, matched by initializer name
  swap_vit_linear_initializers  tuned ViT linear weights (folded low-rank adapters), matched by the
                                stock weight they replace, because the ONNX stores them as unnamed
                                transposed MatMul constants
  retarget_vit_input            the ViT sees a bilinear resize of the high grid to
                                round(size * scale / 14) * 14 instead of the 2x2 average, and
                                DAV2's final feature resize targets the half grid, so the decoder
                                output and everything after it keep their shapes
  apply_output_gain             one constant gain on the public output, so the raw range that the
                                depth-coordinate calibration was measured with is preserved

All sizes are computed from the bound input shape, so one graph serves every point profile. The
functions take and return plain ONNX models and NumPy arrays; nothing here loads PyTorch except
load_state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

VIT_PATCH = 14
AVERAGE_POOL_NODE = "single_high_io_average_pool_2x2"
VIT_RESIZE_NODE = "single_high_io_vit_bilinear_resize"
OUTPUT_GAIN_NODE = "tuned_output_gain"
PUBLIC_INPUT = "pixel_values"
PUBLIC_OUTPUT = "refined_depth"
UNGAINED_OUTPUT = "refined_depth_before_gain"
HEAD_PREFIX = "depth_head."
VIT_BLOCK_PREFIX = "pretrained.blocks."
FINAL_CONV_WEIGHT = "depth_head.scratch.output_conv2.0.weight"
_PREFIX = "single_high_io_vit_"


def vit_input_size(high: int, scale: float) -> int:
    """ViT input extent for a high-grid extent: a whole number of patches, half rounded to even
    (Python's round and ONNX Round agree)."""
    return int(round(high * scale / VIT_PATCH)) * VIT_PATCH


def load_state(path: Path) -> dict[str, np.ndarray]:
    """A PyTorch state dict as NumPy arrays (weights only)."""
    import torch  # deferred: only the exporter process needs PyTorch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return {name: value.detach().cpu().numpy() for name, value in state.items()}


def split_checkpoint(path: Path, zipdepth_state_path: Path) -> dict[str, np.ndarray]:
    """A tuned checkpoint holds the DAV2 state under "dav2" and the ZipDepth state under "zip".
    Writes the ZipDepth state as its own file for the branch exporter and returns the DAV2 state."""
    import torch  # deferred: only the exporter process needs PyTorch

    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"dav2", "zip"} <= set(payload):
        raise ValueError(f"{path} is not a tuned checkpoint with 'dav2' and 'zip' states")
    torch.save(payload["zip"], str(zipdepth_state_path))
    return {name: value.detach().cpu().numpy() for name, value in payload["dav2"].items()}


def _initializers(model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
    result: dict[str, onnx.TensorProto] = {}
    for initializer in model.graph.initializer:
        if initializer.name in result:
            raise ValueError(f"duplicate initializer {initializer.name}")
        result[initializer.name] = initializer
    return result


def _producer(model: onnx.ModelProto, value: str) -> onnx.NodeProto:
    found = [node for node in model.graph.node if value in node.output]
    if len(found) != 1:
        raise ValueError(f"expected one producer of {value}, found {len(found)}")
    return found[0]


def swap_head_initializers(
        dav2: onnx.ModelProto, tuned: Mapping[str, np.ndarray]) -> tuple[int, list[str]]:
    """Write every tuned decoder-head tensor over the initializer of the same name. Returns the
    number written and the tuned names the graph does not hold (constant-folded away)."""
    initializers = _initializers(dav2)
    written, absent = 0, []
    for name, value in tuned.items():
        if not name.startswith(HEAD_PREFIX):
            continue
        target = initializers.get(name)
        if target is None:
            absent.append(name)
            continue
        old = numpy_helper.to_array(target)
        if tuple(value.shape) != tuple(old.shape):
            raise ValueError(f"{name}: tuned shape {value.shape} != graph shape {old.shape}")
        target.CopyFrom(numpy_helper.from_array(np.asarray(value).astype(old.dtype), name))
        written += 1
    return written, absent


def swap_vit_linear_initializers(
        dav2: onnx.ModelProto,
        stock: Mapping[str, np.ndarray],
        tuned: Mapping[str, np.ndarray],
        tolerance: float = 1e-6) -> int:
    """Write every changed two-dimensional ViT block weight over the one initializer that equals the
    transposed stock weight. A weight that matches no initializer, or more than one, is an error:
    a silent miss would export a partly tuned model."""
    initializers = _initializers(dav2)
    by_shape: dict[tuple[int, ...], list[str]] = {}
    for name, initializer in initializers.items():
        if len(initializer.dims) == 2:
            by_shape.setdefault(tuple(initializer.dims), []).append(name)
    written = 0
    for name, new in tuned.items():
        if not (name.startswith(VIT_BLOCK_PREFIX) and name.endswith(".weight") and new.ndim == 2):
            continue
        if name not in stock:
            raise ValueError(f"{name}: the stock checkpoint has no such weight")
        old = stock[name]
        if old.shape != new.shape:
            raise ValueError(f"{name}: tuned shape {new.shape} != stock shape {old.shape}")
        if np.array_equal(old, new):
            continue
        wanted = old.T
        matches = []
        for candidate in by_shape.get(tuple(wanted.shape), []):
            stored = numpy_helper.to_array(initializers[candidate])
            reference = wanted.astype(stored.dtype).astype(np.float32)
            if np.allclose(stored.astype(np.float32), reference, atol=tolerance):
                matches.append(candidate)
        if len(matches) != 1:
            raise ValueError(f"{name}: expected one matching initializer, found {len(matches)}")
        stored = numpy_helper.to_array(initializers[matches[0]])
        initializers[matches[0]].CopyFrom(
            numpy_helper.from_array(new.T.astype(stored.dtype), matches[0]))
        written += 1
    return written


def _final_feature_resize(fused: onnx.ModelProto) -> onnx.NodeProto:
    final_conv = [node for node in fused.graph.node if FINAL_CONV_WEIGHT in node.input]
    if len(final_conv) != 1:
        raise ValueError("the graph has no single final DAV2 output convolution")
    node = _producer(fused, final_conv[0].input[0])
    for _ in range(4):
        if node.op_type == "Resize":
            return node
        node = _producer(fused, node.input[0])
    if node.op_type != "Resize":
        raise ValueError("DAV2's final feature Resize was not found before its output convolution")
    return node


def retarget_vit_input(fused: onnx.ModelProto, scale: float) -> dict[str, object]:
    """Replace the 2x2 average with a bilinear resize to the ViT grid and point DAV2's final
    feature resize at the half grid. Returns the names involved, for the export report."""
    if not 0.0 < scale <= 1.0:
        raise ValueError(f"ViT input scale {scale} is outside (0, 1]")
    graph = fused.graph
    pools = [node for node in graph.node if node.name == AVERAGE_POOL_NODE]
    if len(pools) != 1 or pools[0].op_type != "AveragePool" or list(pools[0].input) != [PUBLIC_INPUT]:
        raise ValueError("the composite's 2x2 input average was not found")
    pool = pools[0]
    final_resize = _final_feature_resize(fused)
    initializers = _initializers(fused)
    feature_channels = int(initializers[FINAL_CONV_WEIGHT].dims[1])

    def constant(name: str, value: np.ndarray) -> str:
        if _PREFIX + name in initializers:
            raise ValueError(f"the graph was already retargeted ({_PREFIX + name})")
        graph.initializer.append(numpy_helper.from_array(value, _PREFIX + name))
        return _PREFIX + name

    high = _PREFIX + "high_hw"
    nodes = [
        helper.make_node("Shape", [PUBLIC_INPUT], [high], name=_PREFIX + "high_hw_shape", start=2, end=4),
        helper.make_node("Cast", [high], [high + "_float"], name=_PREFIX + "high_hw_cast", to=TensorProto.FLOAT),
        helper.make_node(
            "Mul", [high + "_float", constant("scale_over_patch", np.array(scale / VIT_PATCH, np.float32))],
            [_PREFIX + "patches_float"], name=_PREFIX + "patches_mul"),
        helper.make_node("Round", [_PREFIX + "patches_float"], [_PREFIX + "patches"], name=_PREFIX + "patches_round"),
        helper.make_node(
            "Mul", [_PREFIX + "patches", constant("patch", np.array(float(VIT_PATCH), np.float32))],
            [_PREFIX + "hw_float"], name=_PREFIX + "hw_mul"),
        helper.make_node("Cast", [_PREFIX + "hw_float"], [_PREFIX + "hw"], name=_PREFIX + "hw_cast", to=TensorProto.INT64),
        helper.make_node(
            "Concat", [constant("batch_rgb", np.array([1, 3], np.int64)), _PREFIX + "hw"],
            [_PREFIX + "sizes"], name=_PREFIX + "sizes_concat", axis=0),
        helper.make_node(
            "Div", [high, constant("two", np.array(2, np.int64))],
            [_PREFIX + "half_hw"], name=_PREFIX + "half_hw_div"),
        helper.make_node(
            "Concat", [constant("batch_features", np.array([1, feature_channels], np.int64)), _PREFIX + "half_hw"],
            [_PREFIX + "half_sizes"], name=_PREFIX + "half_sizes_concat", axis=0),
        helper.make_node(
            "Resize", [PUBLIC_INPUT, "", "", _PREFIX + "sizes"], list(pool.output),
            name=VIT_RESIZE_NODE, mode="linear", coordinate_transformation_mode="half_pixel"),
    ]
    index = list(graph.node).index(pool)
    graph.node.remove(pool)
    for offset, node in enumerate(nodes):
        graph.node.insert(index + offset, node)
    while len(final_resize.input) < 4:
        final_resize.input.append("")
    final_resize.input[2] = ""
    final_resize.input[3] = _PREFIX + "half_sizes"
    # Shapes recorded for the removed average no longer hold; inference recomputes them.
    del graph.value_info[:]
    return {
        "scale": scale,
        "vit_resize_node": VIT_RESIZE_NODE,
        "final_feature_resize_node": final_resize.name,
        "feature_channels": feature_channels,
    }


def apply_output_gain(fused: onnx.ModelProto, gain: float) -> None:
    """Multiply the public output by one constant."""
    if not gain > 0.0:
        raise ValueError(f"output gain {gain} is not positive")
    graph = fused.graph
    if any(node.name == OUTPUT_GAIN_NODE for node in graph.node):
        raise ValueError("the graph already has an output gain")
    last = _producer(fused, PUBLIC_OUTPUT)
    for index, name in enumerate(last.output):
        if name == PUBLIC_OUTPUT:
            last.output[index] = UNGAINED_OUTPUT
    graph.initializer.append(numpy_helper.from_array(np.array(gain, np.float32), OUTPUT_GAIN_NODE + "_value"))
    graph.node.append(helper.make_node(
        "Mul", [UNGAINED_OUTPUT, OUTPUT_GAIN_NODE + "_value"], [PUBLIC_OUTPUT], name=OUTPUT_GAIN_NODE))
