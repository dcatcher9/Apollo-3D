"""Emit the small trained color-vote correction into the research native shader."""
import hashlib
from pathlib import Path
import torch

def insert(shader, checkpoint, strength):
    if not 0 < strength <= 1:
        raise ValueError("Invalid correction strength")
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    indices = (1, 2, 3, 4, 5, 7, 35, 36, 38, 40, 41, 45)
    rows = [f"// Vote correction SHA-256 {hashlib.sha256(checkpoint.read_bytes()).hexdigest()}"]
    for key, name, shape in (("0.weight", "GateW0", (8, 12)), ("0.bias","GateB0",(8,)),
                             ("2.weight", "GateW1", (1, 8)), ("2.bias","GateB1",(1,))):
        v = state[key]
        if tuple(v.shape) != shape or not torch.isfinite(v).all():
            raise ValueError(f"Invalid gate tensor {key}")
        rows.append("static const float %s[%d] = {%s};" % (name, v.numel(), ", ".join(f"{x:.9e}f" for x in v.flatten().tolist())))
    rows.append("static const uint GateFeatures[12] = {" + ", ".join(map(str, indices)) + "};")
    original = shader.read_text(encoding="utf8")
    marker = "float PaletteMask(float features[50]) {"
    if original.count(marker) != 1:
        raise ValueError("Incompatible native decoder")
    rows.extend([marker,
        "    float gateLogit = GateB1[0];",
        "    [unroll] for (uint g=0; g<8; ++g) {",
        "        float activation = GateB0[g];",
        "        [unroll] for (uint j=0; j<12; ++j) activation += GateW0[g*12+j] * features[GateFeatures[j]];",
        "        gateLogit += max(activation, 0.0f) * GateW1[g];",
        "    }",
        f"    float mix = {strength:.9e}f / (1.0f + exp(-gateLogit));",
        "    features[37] *= 1.0f-mix;",
        "    features[38] = (1.0f-mix)*features[38]+mix;",
        "    features[39] *= 1.0f-mix;"])
    shader.write_text(original.replace(marker, "\n".join(rows)), encoding="utf8", newline="\n")

