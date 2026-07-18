#!/usr/bin/env python3
"""Optional Apple iSQoE stereoscopic preference oracle.

iSQoE is the only optional oracle in this directory trained directly from headset-viewed stereo
preferences.  It is therefore useful as an independent holistic cross-check, but not as a
teacher label: its single score mixes fidelity, comfort, and depth taste, and that taste is not
necessarily Apollo's deliberately strong-pop style.  Every result remains diagnostic-only.

The official model currently loads DINOv2 through ``torch.hub``.  Newer DINOv2 revisions expose
dropout probabilities as floats instead of ``nn.Dropout`` modules.  ``CompatibleDinoV2`` keeps
the official architecture and weights while accepting either public API representation; no
model parameter or inference equation is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any

from PIL import Image


SCHEMA = 1
ORACLE = "apple-isqoe"
ROLE = "optional_eval_only_experimental_diagnostic"
OFFICIAL_RESOLUTION = (720, 1280)
OFFICIAL_REPOSITORY_URL = "https://github.com/apple/ml-isqoe"
OFFICIAL_CHECKPOINT_ID = "isqoe_1_1"
OFFICIAL_CHECKPOINT_URL = (
    "https://ml-site.cdn-apple.com/models/isqoe/isqoe_1_1.ckpt"
)
# Apple does not publish a checksum beside the download.  This is the SHA-256 of the official
# isqoe_1_1.ckpt fetched by evaluation/download_checkpoint.sh on 2026-07-17.  Keep the actual
# digest in every result even if Apple replaces the object at the same URL in the future.
KNOWN_OFFICIAL_CHECKPOINT_SHA256 = (
    "1a4a367ac2bb03125cd5df9e507856dc50338f06315def4224d59a5ab55b5ed3"
)


class IsqoeUnavailable(RuntimeError):
    """The official checkout, checkpoint, or Python runtime is unavailable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _repository_revision(repo: Path) -> str | None:
    """Read a checkout revision without invoking git or changing safe-directory config."""
    git_dir = repo / ".git"
    if not git_dir.is_dir():
        return None
    try:
        head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
        if not head.startswith("ref: "):
            return head if len(head) == 40 else None
        reference = head[5:]
        loose_ref = git_dir / reference
        if loose_ref.is_file():
            revision = loose_ref.read_text(encoding="ascii").strip()
            return revision if len(revision) == 40 else None
        packed_refs = git_dir / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text(encoding="ascii").splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                revision, name = line.split(" ", 1)
                if name == reference and len(revision) == 40:
                    return revision
    except (OSError, UnicodeError, ValueError):
        return None
    return None


def _dropout_probability(value: Any) -> float:
    """Read both the old ``nn.Dropout.p`` and the current float DINOv2 contract."""
    return float(value.p) if hasattr(value, "p") else float(value)


class IsqoeModel:
    """Load the official Apple model once and evaluate packed SBS PNGs."""

    def __init__(self, repo: Path, checkpoint: Path, device: str | None = None):
        self.repo = Path(repo).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not (self.repo / "isqoe" / "lightning_model.py").is_file():
            raise IsqoeUnavailable(f"not an Apple ml-isqoe checkout: {self.repo}")
        if not self.checkpoint.is_file():
            raise IsqoeUnavailable(f"iSQoE checkpoint is missing: {self.checkpoint}")
        checkpoint_sha256 = _sha256(self.checkpoint)

        try:
            import torch
            from torchvision import transforms
        except Exception as error:
            raise IsqoeUnavailable(f"PyTorch/torchvision import failed: {error}") from error
        self.torch = torch
        self.transforms = transforms
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise IsqoeUnavailable("CUDA was requested but torch.cuda.is_available() is false")
        self.device = device

        repo_text = str(self.repo)
        inserted = repo_text not in sys.path
        if inserted:
            sys.path.insert(0, repo_text)
        try:
            from isqoe.lightning_model import LightningPerceptualModel
            from isqoe.pretrained_models.feature_extractor import DINOv2
            from isqoe.pretrained_models.dinov2.layers.attention import MemEffCrossAttention

            class CompatibleDinoV2(DINOv2):
                def __init__(self):
                    torch.nn.Module.__init__(self)
                    self.backbone = "dinov2_vits14"
                    self.feature = "patches"
                    self.xatten = "concat"
                    self.xatten_layers = [2, 5, 8, 11]
                    self.model = self.build_model()
                    self.patch_size = self.model.patch_embed.proj.kernel_size[0]
                    self.embed_dim = self.model.patch_embed.proj.out_channels
                    for block_index, block in enumerate(self.model.blocks):
                        if block_index not in self.xatten_layers:
                            continue
                        cross_attention = MemEffCrossAttention(
                            block.attn.qkv.in_features,
                            block.attn.num_heads,
                            block.attn.qkv.bias is not None,
                            block.attn.proj.bias is not None,
                            _dropout_probability(block.attn.attn_drop),
                            _dropout_probability(block.attn.proj_drop),
                            self.xatten,
                        )
                        cross_attention.load_state_dict(block.attn.state_dict())
                        block.attn = cross_attention
                    self.preprocess = self.get_preprocess()

            model = LightningPerceptualModel(
                feature_extractor=CompatibleDinoV2(),
                dataset_dir=self.repo / "dataset" / "scope",
                use_lora=True,
                lora_r=8,
                lora_alpha=32,
                lora_dropout=0.1,
            )
            checkpoint_payload = torch.load(
                self.checkpoint, map_location=device, weights_only=True)
            state = (checkpoint_payload.get("state_dict", checkpoint_payload)
                     if isinstance(checkpoint_payload, dict) else checkpoint_payload)
            model.load_state_dict(state, strict=True)
            model.to(device)
            model.requires_grad_(False)
            model.eval()
            self.model = model.stereo_perceptual_model
        except IsqoeUnavailable:
            raise
        except Exception as error:
            raise IsqoeUnavailable(f"official iSQoE model load failed: {error}") from error
        finally:
            if inserted:
                try:
                    sys.path.remove(repo_text)
                except ValueError:  # pragma: no cover - defensive against imported hooks
                    pass

        self.preprocess = transforms.Compose([
            transforms.Resize(
                OFFICIAL_RESOLUTION,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
        ])
        self._provenance = {
            "implementation": "official Apple ml-isqoe architecture/checkpoint",
            "official_repository_url": OFFICIAL_REPOSITORY_URL,
            "repository_checkout": str(self.repo),
            "repository_revision": _repository_revision(self.repo),
            "official_checkpoint_id": OFFICIAL_CHECKPOINT_ID,
            "official_checkpoint_url": OFFICIAL_CHECKPOINT_URL,
            "checkpoint": str(self.checkpoint),
            "checkpoint_filename": self.checkpoint.name,
            "checkpoint_bytes": self.checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_matches_known_official_sha256": (
                checkpoint_sha256 == KNOWN_OFFICIAL_CHECKPOINT_SHA256
            ),
            "known_official_checkpoint_sha256": KNOWN_OFFICIAL_CHECKPOINT_SHA256,
            "device": self.device,
            "input_resolution_per_eye": list(OFFICIAL_RESOLUTION),
            "torch_version": _version("torch"),
            "torchvision_version": _version("torchvision"),
            "lightning_version": _version("lightning"),
            "peft_version": _version("peft"),
            "license": "Apple ML Research Model Terms; research use only",
            "dinov2_api_adapter": "dropout representation only; inference unchanged",
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def _tensor(self, image: Image.Image):
        return self.preprocess(image.convert("RGB"))[None].to(self.device)

    def evaluate(self, left: Image.Image, right: Image.Image) -> dict[str, float]:
        """Measure both eye orders and expose disagreement instead of hiding it."""
        left_tensor = self._tensor(left)
        right_tensor = self._tensor(right)
        with self.torch.inference_mode():
            forward_output = self.model(left_tensor, right_tensor)
            swapped_output = self.model(right_tensor, left_tensor)
        if forward_output.numel() != 1 or swapped_output.numel() != 1:
            raise ValueError(
                "official iSQoE model must return one scalar per stereo pair, got "
                f"{tuple(forward_output.shape)} and {tuple(swapped_output.shape)}"
            )
        forward = float(forward_output.item())
        swapped = float(swapped_output.item())
        if not math.isfinite(forward) or not math.isfinite(swapped):
            raise ValueError(
                f"official iSQoE model returned non-finite scores: {forward}, {swapped}"
            )
        return {
            "isqoe_score": forward,
            "isqoe_swapped_score": swapped,
            "isqoe_mean_score": 0.5 * (forward + swapped),
            "isqoe_worst_score": max(forward, swapped),
            "isqoe_eye_order_delta": abs(forward - swapped),
        }

    def evaluate_path(self, path: Path) -> dict[str, Any]:
        path = Path(path).expanduser().resolve()
        with Image.open(path) as packed:
            packed = packed.convert("RGB")
            if packed.width % 2:
                raise ValueError(f"SBS width must be even, got {packed.width}")
            eye_width = packed.width // 2
            left = packed.crop((0, 0, eye_width, packed.height))
            right = packed.crop((eye_width, 0, packed.width, packed.height))
            metrics = self.evaluate(left, right)
        return {
            "schema": SCHEMA,
            "oracle": ORACLE,
            "status": "ok",
            "role": ROLE,
            "qualification": "experimental_diagnostic_only",
            "training_label_eligible": False,
            "path": str(path),
            "input_sha256": _sha256(path),
            "metrics": metrics,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    model = IsqoeModel(args.repo, args.checkpoint, args.device)
    payload = {
        "schema": SCHEMA,
        "oracle": ORACLE,
        "status": "ok",
        "role": ROLE,
        "qualification": "experimental_diagnostic_only",
        "training_label_eligible": False,
        "provenance": model.provenance,
        "frames": [model.evaluate_path(path) for path in args.images],
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
