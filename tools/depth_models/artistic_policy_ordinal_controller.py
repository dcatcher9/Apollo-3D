"""Causal controller for experimental ordinal artistic-policy predictions.

The controller deliberately remains offline-only until the ordinal policy has
passed rendered evaluation.  It consumes one desired safe scale per completed
depth update, lowers immediately, and requires sustained evidence before any
increase.  The same state machine can therefore be replayed during evaluation
without using future frames from a scene.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from artistic_policy_ordinal_contract import SCALES, scale_index


IDENTITY_SCALE = SCALES[0]


def _positive_updates(seconds, depth_fps, description):
    if (not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or
            not math.isfinite(float(seconds)) or float(seconds) < 0.0):
        raise RuntimeError(f"{description} is invalid")
    if (not isinstance(depth_fps, (int, float)) or
            isinstance(depth_fps, bool) or
            not math.isfinite(float(depth_fps)) or float(depth_fps) <= 0.0):
        raise RuntimeError("depth fps is invalid")
    return max(1, int(math.ceil(float(seconds) * float(depth_fps))))


def _canonical_scale(value):
    return SCALES[scale_index(value)]


def mix_style_scale(safe_cap, style, balanced_fraction=0.5):
    """Resolve a presentation style only after the safety cap is known."""
    safe_cap = _canonical_scale(safe_cap)
    if style == "clean":
        return IDENTITY_SCALE
    if style == "immersive":
        return safe_cap
    if style != "balanced":
        raise RuntimeError("unknown artistic presentation style")
    if (not isinstance(balanced_fraction, (int, float)) or
            isinstance(balanced_fraction, bool) or
            not math.isfinite(float(balanced_fraction)) or
            not 0.0 <= float(balanced_fraction) <= 1.0):
        raise RuntimeError("balanced style fraction is outside [0,1]")
    mixed = IDENTITY_SCALE + (
        safe_cap - IDENTITY_SCALE
    ) * float(balanced_fraction)
    # Style mixing is presentation-only and need not land on an ordinal bin.
    return mixed


@dataclass(frozen=True)
class ControllerOutput:
    desired_scale: float
    applied_safe_cap: float
    changed: bool
    change_kind: str
    safe_raise_updates: int
    cooldown_updates: int


class CausalOrdinalController:
    """Immediate-down/delayed-up controller measured in depth updates."""

    def __init__(self, depth_fps, safe_hold_seconds=0.5,
                 upward_cooldown_seconds=2.0):
        self.safe_hold_updates = _positive_updates(
            safe_hold_seconds, depth_fps, "safe hold duration"
        )
        self.upward_cooldown_updates = _positive_updates(
            upward_cooldown_seconds, depth_fps, "upward cooldown duration"
        )
        self.reset()

    def reset(self):
        """Reset at a stream start or authenticated production hard cut."""
        self.applied = IDENTITY_SCALE
        self.raise_minimum = None
        self.raise_updates = 0
        self.cooldown = 0

    def _output(self, desired, changed=False, change_kind="hold"):
        return ControllerOutput(
            desired_scale=desired,
            applied_safe_cap=self.applied,
            changed=changed,
            change_kind=change_kind,
            safe_raise_updates=self.raise_updates,
            cooldown_updates=self.cooldown,
        )

    def update(self, desired_scale, hard_cut=False):
        """Apply one causal decision for one completed depth update."""
        desired = _canonical_scale(desired_scale)
        if hard_cut:
            changed = self.applied != IDENTITY_SCALE
            self.reset()
            return self._output(
                desired, changed=changed,
                change_kind="cut_reset" if changed else "cut_hold",
            )

        was_in_cooldown = self.cooldown > 0
        if was_in_cooldown:
            self.cooldown -= 1

        if desired < self.applied:
            self.applied = desired
            self.raise_minimum = None
            self.raise_updates = 0
            # A safety reduction starts a fresh observation period.  It does
            # not need an additional cooldown because the hold itself is the
            # conservative upward guard.
            self.cooldown = 0
            return self._output(desired, changed=True,
                                change_kind="safety_lower")

        if desired <= self.applied or was_in_cooldown:
            self.raise_minimum = None
            self.raise_updates = 0
            return self._output(desired)

        if self.raise_minimum is None:
            self.raise_minimum = desired
        else:
            self.raise_minimum = min(self.raise_minimum, desired)
        self.raise_updates += 1
        if self.raise_updates < self.safe_hold_updates:
            return self._output(desired)

        self.applied = self.raise_minimum
        self.raise_minimum = None
        self.raise_updates = 0
        self.cooldown = self.upward_cooldown_updates
        return self._output(desired, changed=True, change_kind="safe_raise")


def replay_desired_scales(desired_scales, depth_fps, hard_cut_indices=(),
                          safe_hold_seconds=0.5,
                          upward_cooldown_seconds=2.0):
    """Replay a trace without looking ahead at later desired scales."""
    hard_cuts = set(hard_cut_indices)
    if any(not isinstance(index, int) or index < 0 for index in hard_cuts):
        raise RuntimeError("hard-cut indices are invalid")
    controller = CausalOrdinalController(
        depth_fps,
        safe_hold_seconds=safe_hold_seconds,
        upward_cooldown_seconds=upward_cooldown_seconds,
    )
    return [
        controller.update(value, hard_cut=index in hard_cuts)
        for index, value in enumerate(desired_scales)
    ]
