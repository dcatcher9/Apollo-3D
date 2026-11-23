"""Export a Direct Glyph graph with FP16 convolutions and FP32 protection scores."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def export(source: Path, destination: Path, *, dynamic_batch: bool = False) -> None:
    if source.resolve() == destination.resolve():
        raise ValueError("Choose a separate destination for the mixed-precision graph")
    model = onnx.load(source)
    graph = model.graph
    nodes = list(graph.node)
    # The two classifier tensors feed the original FP32 probability calculation.
    split = next(i for i, node in enumerate(nodes) if node.op_type == "LogSoftmax")
    fill = next(node for node in nodes[:split] if "fill.weight" in node.input)
    decoration = nodes[split - 1]
    if (
        decoration.op_type != "Conv"
        or "decor.weight" not in decoration.input
        or nodes[split].input[0] != decoration.output[0]
        or [tensor.name for tensor in graph.input] != ["rgb"]
        or [tensor.name for tensor in graph.output] != ["logits"]
    ):
        raise ValueError("Expected the stroke-anchored Direct Glyph graph")

    boundary = (decoration.output[0], fill.output[0])
    weights = {name for node in nodes[:split] if node.op_type == "Conv" for name in node.input[1:]}
    for tensor in graph.initializer:
        if tensor.name in weights:
            converted = numpy_helper.to_array(tensor).astype(np.float16)
            tensor.CopyFrom(numpy_helper.from_array(converted, tensor.name))

    # ONNX Resize takes FP32 scale factors even when its image tensor is FP16.
    scales = {name for node in nodes[:split] if node.op_type == "Resize" for name in node.input[1:]}
    updated = [helper.make_node("Cast", ["rgb"], ["half_input"], to=TensorProto.FLOAT16)]
    for i, node in enumerate(nodes):
        if i == split:
            for name in boundary:
                updated.append(helper.make_node("Cast", [name], [name + "_float"], to=TensorProto.FLOAT))
        for j, name in enumerate(node.input):
            if i < split and name == "rgb":
                node.input[j] = "half_input"
            elif i >= split and name in boundary:
                node.input[j] = name + "_float"
        if i < split and node.op_type == "Constant" and node.output[0] not in scales:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT:
                    attr.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(attr.t).astype(np.float16)))
        updated.append(node)
    del graph.node[:]
    graph.node.extend(updated)
    del graph.value_info[:]
    if dynamic_batch:
        for tensor in (graph.input[0], graph.output[0]):
            tensor.type.tensor_type.shape.dim[0].dim_param = "tiles"
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--dynamic-batch", action="store_true", help="Export the shipping encoder with a symbolic tile batch axis")
    args = parser.parse_args()
    export(args.source, args.destination, dynamic_batch=args.dynamic_batch)
