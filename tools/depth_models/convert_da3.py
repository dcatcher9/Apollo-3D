#!/usr/bin/env python3
"""Produce single-file DA-V3 ONNX (fp32 and fp16) for Apollo's SBS depth pipeline.

The onnx-community DA-V3 exports ship as a two-file fp32 model (model.onnx graph +
model.onnx_data external weights) whose data file is referenced by the literal name
"model.onnx_data" -- so multiple models can't share an assets dir, and TRT 11 (strongly
typed) would build an fp32 engine from it. This tool:

  1. Loads the two-file fp32 and re-saves it single-file as <stem>_fp32.onnx.
  2. Converts to fp16 (keep_io_types=True -> FP32 pixel_values/predicted_depth I/O, fp16
     weights, so the D3D pipeline binds FP32 buffers and TRT builds an fp16 engine).

fp16 conversion notes (hard-won -- DA-V3 uses a DINOv3 backbone with RoPE):
  - onnxconverter_common.float16.convert_float_to_float16 produces an INVALID graph on this
    model (type mismatch at .../attn/rope/Cast_*), and auto_convert_mixed_precision is built
    on it so it fails identically. Do NOT use them here.
  - onnxruntime.transformers' converter (needs `sympy`) works, but emits two redundant
    cast-to-fp32 nodes at the head with a COLLIDING output-tensor name. We drop the duplicate
    producers (keep one) to make the graph valid. Result: predicted_depth corr 1.0 vs fp32.

Usage:
  python convert_da3.py --repo onnx-community/depth-anything-v3-small --stem depth_anything_v3_small
  # writes depth_anything_v3_small_fp32.onnx and depth_anything_v3_small_fp16.onnx

Deps: onnx, onnxruntime, sympy, huggingface_hub (or pre-download the two files).
"""
import argparse
import os
import tempfile
from collections import defaultdict

import onnx


def load_two_file_fp32(repo: str, cache_dir: str) -> onnx.ModelProto:
    from huggingface_hub import hf_hub_download
    graph = hf_hub_download(repo, "onnx/model.onnx", cache_dir=cache_dir)
    # The graph references "model.onnx_data" by that literal name; place it alongside.
    hf_hub_download(repo, "onnx/model.onnx_data", cache_dir=cache_dir)
    return onnx.load(graph, load_external_data=True)


def to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    from onnxruntime.transformers.onnx_model import OnnxModel
    om = OnnxModel(model)
    om.convert_float_to_float16(keep_io_types=True)
    m = om.model
    g = m.graph
    # Drop redundant duplicate-producer nodes (ORT converter bug: two cast nodes emit the same
    # output tensor name -> "Duplicate definition of name" on load). Keep the first producer.
    producers = defaultdict(list)
    for i, n in enumerate(g.node):
        for o in n.output:
            producers[o].append(i)
    remove = {j for idxs in producers.values() if len(idxs) > 1 for j in idxs[1:]}
    if remove:
        keep = [n for i, n in enumerate(g.node) if i not in remove]
        del g.node[:]
        g.node.extend(keep)
    # De-duplicate node names (harmless, but keeps the graph strictly valid).
    seen = {}
    for n in g.node:
        if not n.name or n.name in seen:
            base = n.name or n.op_type
            seen[base] = seen.get(base, 0) + 1
            n.name = f"{base}__u{seen[base]}"
        else:
            seen[n.name] = 0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo, e.g. onnx-community/depth-anything-v3-small")
    ap.add_argument("--stem", required=True, help="output stem, e.g. depth_anything_v3_small")
    ap.add_argument("--fp32-onnx", help="use this local two-file fp32 graph instead of downloading")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    if args.fp32_onnx:
        model = onnx.load(args.fp32_onnx, load_external_data=True)
    else:
        model = load_two_file_fp32(args.repo, tempfile.mkdtemp())

    fp32_out = os.path.join(args.outdir, f"{args.stem}_fp32.onnx")
    onnx.save_model(model, fp32_out, save_as_external_data=False)
    print(f"wrote {fp32_out} ({os.path.getsize(fp32_out)/1e6:.1f} MB)")

    fp16 = to_fp16(onnx.load(fp32_out))
    fp16_out = os.path.join(args.outdir, f"{args.stem}_fp16.onnx")
    onnx.save_model(fp16, fp16_out, save_as_external_data=False)
    print(f"wrote {fp16_out} ({os.path.getsize(fp16_out)/1e6:.1f} MB)")

    # Validate predicted_depth agreement fp16 vs fp32 on random input.
    try:
        import numpy as np
        import onnxruntime as ort
        x = np.random.rand(1, 1, 3, 336, 798).astype(np.float32)
        a = ort.InferenceSession(fp32_out, providers=["CPUExecutionProvider"]).run(
            ["predicted_depth"], {"pixel_values": x})[0].ravel()
        b = ort.InferenceSession(fp16_out, providers=["CPUExecutionProvider"]).run(
            ["predicted_depth"], {"pixel_values": x})[0].ravel()
        print(f"fp16-vs-fp32 predicted_depth corr = {np.corrcoef(a, b)[0, 1]:.5f}")
    except Exception as e:  # noqa: BLE001
        print(f"(skipped validation: {e})")


if __name__ == "__main__":
    main()
