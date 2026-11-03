"""CPU reference for UIP4 presentation timing, not OCR/geometry or GPU qualification.

Inputs are already-authorized current observations. Coordinates are one-eye native
source U; goal is the current conflict-histogram candidate. No pixel mask is made here.
"""
from dataclasses import dataclass
import math

try:
    from .depth_coordinate_v2_contract import SUBTITLE_OCR
except ImportError:
    from depth_coordinate_v2_contract import SUBTITLE_OCR


@dataclass(frozen=True)
class PlaneTiming:
    applied: float = 0.0
    goal: float = 0.0
    accepted_us: int = 0
    presented_active: bool = False


def advance(previous: PlaneTiming, *, goal: float, now_us: int, source_width: int,
            visible: bool = True, ready: bool = True, valid_ocr: bool = True,
            valid_probe: bool = True, distinct: bool = True, reset: bool = False,
            limit: float = 0.04) -> PlaneTiming:
    """Model only eligible source-time integration; unknown supplies no empty cue.

    A shrinking representation limit clamps immediately for validity. This is the
    explicit exception to slew, along with first entry/reset and authoritativeEMPTY.
    """
    if not distinct and not reset:
        return previous
    if reset:
        previous = PlaneTiming()
    applied = min(previous.applied, limit)
    held = PlaneTiming(applied, previous.goal, 0, previous.presented_active)
    if not ready or not valid_ocr:
        return held
    clock_ok = now_us > 0 and (previous.accepted_us == 0 or now_us > previous.accepted_us)
    if not visible:
        return PlaneTiming(accepted_us=now_us if clock_ok else 0)
    if not valid_probe or not clock_ok:
        return held
    goal = min(max(goal, 0.0), limit)
    if not previous.presented_active:
        return PlaneTiming(goal, goal, now_us, True)
    if previous.accepted_us:
        seconds = min(now_us - previous.accepted_us,
                      SUBTITLE_OCR.plane_max_tick_gap_ms * 1000) * 1e-6
        alpha = -math.expm1(-seconds / (SUBTITLE_OCR.plane_smoothing_time_constant_ms * .001))
        slew = SUBTITLE_OCR.plane_slew_source_pixels_per_second * seconds / source_width
        update = applied + max(-slew, min(slew, (goal - applied) * alpha))
        applied = max(min(applied, goal), min(max(applied, goal), update))
    return PlaneTiming(applied, goal, now_us, True)
