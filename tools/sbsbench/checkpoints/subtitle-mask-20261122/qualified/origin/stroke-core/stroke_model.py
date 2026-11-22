"""Three-scale RGB glyph matte with a learned stroke-interior anchor."""

import torch
from torch import nn
from torch.nn import functional as F


class _ResidualContext(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.spatial = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation, groups=channels // 4
        )
        self.mix = nn.Conv2d(channels, channels, 1)

    def forward(self, features):
        return features + self.mix(self.spatial(features).relu()).relu()


class StrokeAnchoredGlyph(nn.Module):
    """Predict background, visible fill, visible outline and visible shadow log scores.

    The RGB input is in [0, 1]. Height and width are multiples of four. Align a
    crop to the full-source four-pixel pooling grid before calling this model.
    """

    receptive_halo = 40

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1)
        self.native_context = _ResidualContext(16)
        self.half_project = nn.Conv2d(16, 24, 1)
        self.half_context = _ResidualContext(24)
        self.quarter_project = nn.Conv2d(24, 32, 1)
        self.quarter_context = nn.Sequential(
            _ResidualContext(32),
            _ResidualContext(32, dilation=2),
            _ResidualContext(32, dilation=3),
        )
        self.quarter_to_half = nn.Conv2d(32, 24, 1)
        self.half_to_native = nn.Conv2d(24, 16, 1)
        self.fuse = nn.Conv2d(32, 16, 1)

        self.fill = nn.Conv2d(16, 1, 3, padding=1)
        self.decor_input = nn.Conv2d(17, 16, 1)
        self.decor_context = _ResidualContext(16)
        self.decor = nn.Conv2d(16, 3, 1)

    def forward(self, rgb):
        native = self.native_context(self.stem(rgb * 2 - 1).relu())
        half = self.half_context(self.half_project(F.avg_pool2d(native, 2)).relu())
        quarter = self.quarter_context(self.quarter_project(F.avg_pool2d(half, 2)).relu())
        half = half + F.interpolate(
            self.quarter_to_half(quarter), scale_factor=2, mode="bilinear", align_corners=False
        )
        context = F.interpolate(
            self.half_to_native(half), scale_factor=2, mode="bilinear", align_corners=False
        )
        features = self.fuse(torch.cat((native, context), dim=1)).relu()

        fill = self.fill(features)
        anchored = self.decor_input(torch.cat((features, fill.sigmoid()), dim=1)).relu()
        conditional = F.log_softmax(self.decor(self.decor_context(anchored)), dim=1)
        nonfill = F.logsigmoid(-fill) + conditional
        return torch.cat((nonfill[:, :1], F.logsigmoid(fill), nonfill[:, 1:]), dim=1)
