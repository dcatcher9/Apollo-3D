"""Offline reference for the November 22 mixed H28 line-palette model."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
import torch

from background import Net
from geometry import windows
from line import native
from tracking import Tracker


ROOT = Path(__file__).resolve().parent


class Masker:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(ROOT / "model/encoder.onnx"), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.output = self.session.get_outputs()[0].name
        self.net = Net(42)
        self.net.load_state_dict(torch.load(ROOT / "model/palette.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        self.tracker = Tracker(28, True, True)

    def encode(self, tensor):
        return self.session.run([self.output], {self.input: tensor})[0]

    def score(self, crop):
        if crop.dtype != np.uint8 or crop.ndim != 3 or crop.shape[2] != 3 or not min(crop.shape[:2]):
            raise ValueError("Expected a nonempty RGB8 line crop")
        alpha, tiles = native(crop, self.encode, self.net)
        return None, alpha, tiles

    def frame(self, rgb, work, covers):
        return self.tracker.predict(rgb, windows(work, covers, "small"), covers, self)


def verify():
    for name, wanted in json.loads((ROOT / "sha256.json").read_bytes()).items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != wanted:
            raise ValueError("Checkpoint hash mismatch for " + name)
    records = json.loads((ROOT / "restore-expected.json").read_bytes())
    masker = Masker()
    with np.load(ROOT / "restore-inputs.npz", allow_pickle=False) as fixture:
        for i, row in enumerate(records):
            mask, _, history = masker.frame(fixture["rgb" + str(i)], row["work"], row["covers"])
            if list(map(list,history)) != row["history"] or sha256(mask.tobytes()).hexdigest() != row["mask_sha256"]:
                raise ValueError("Different temporal mask on " + row["event"])
            if row["work"]:
                raw = masker.tracker.previous[0][3]
                pixels = np.rint(raw*255).astype(np.uint8)
                if sha256(pixels.tobytes()).hexdigest() != row["native_alpha_q8_sha256"]:
                    raise ValueError("Different native opacity on " + row["event"])
    print("Verified every file, five native opacity maps and seven consecutive masks including cue replacement and clearing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    for name in ("alpha", "mask"):
        cmd = commands.add_parser(name)
        cmd.add_argument("crop", type=Path)
        cmd.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "verify":
        verify()
    else:
        with Image.open(args.crop) as image:
            rgb = np.asarray(image.convert("RGB"))
        _, alpha, _ = Masker().score(rgb)
        pixels = np.rint(alpha*255).astype("u1")
        Image.fromarray(pixels if args.command=="alpha" else (pixels>=38).astype("u1")*255).save(args.output)
