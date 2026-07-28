#!/usr/bin/env python3
"""Streaming, auditable scene planning for offline Host SBS conversion.

The production cut detector remains causal.  This module delays commitment by a small number of
future depth updates, refines a proposed boundary with two-sided evidence, and then chooses one
stable camera plan from the complete finalized scene.  It deliberately does not call its result
ground truth or claim that an automatically selected camera is comfort-optimal.

Boundary ``b`` is always the first source frame of the next scene.  A finalized scene therefore
owns ``[start_sequence, end_sequence_exclusive)`` using the native harness's one-based global
sequence numbers.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SCENE_PLAN_SCHEMA = 1
SCENE_PLAN_VERSION = "scene-plan-v1"
CACHE_CONTRACT_SCHEMA = 1

DEPTH_CUT_HIGH = 0.60
DEPTH_CUT_CORROBORATE = 0.25
RAW_RGB_CUT_HIGH = 0.70
STRUCTURAL_COLOR_CUT_HIGH = 0.03
STRUCTURAL_COLOR_MIN_SUPPORT = 0.01
POP_RISK_LOW = 0.04
POP_RISK_HIGH = 0.20

# Optional analysis_flags bits.  The planner also accepts identically named boolean fields,
# allowing a future trace schema to expose the flags without coupling policy to their packing.
ANALYSIS_APPEARANCE_PROPOSAL = 1 << 0
ANALYSIS_EXPOSURE_LIKE = 1 << 1
ANALYSIS_STRUCTURELESS_TRANSITION = 1 << 2
ANALYSIS_SAME_SCENE_RETURN = 1 << 3
ANALYSIS_APPEARANCE_VETO = 1 << 4
ANALYSIS_RELATIVE_GEOMETRY_SPIKE = 1 << 5


class ScenePlanError(ValueError):
    """The trace, policy, or bounded cache cannot produce a trustworthy scene plan."""


class SceneCacheBudgetExceeded(ScenePlanError):
    """The current semantic scene exceeded the configured exact-cache hard cap."""

    def __init__(
        self,
        *,
        limit_bytes: int,
        live_bytes: int,
        open_start_sequence: int,
        current_sequence: int,
    ):
        self.limit_bytes = limit_bytes
        self.live_bytes = live_bytes
        self.open_start_sequence = open_start_sequence
        self.current_sequence = current_sequence
        super().__init__(
            "scene cache budget exceeded before a semantic boundary was finalized: "
            f"{live_bytes} > {limit_bytes} bytes for sequences "
            f"[{open_start_sequence},{current_sequence}]. "
            "Increase the budget or explicitly opt into administrative splitting."
        )


@dataclass(frozen=True)
class ScenePlannerConfig:
    pop_strength: float
    adaptive_pop: bool
    adaptive_pop_max: float
    zero_plane: str
    lookbehind_depth_updates: int = 4
    lookahead_depth_updates: int = 8
    duplicate_pulse_distance_updates: int = 2
    settle_depth_updates: int = 8
    minimum_scene_frames: int = 2
    risk_quantile: float = 0.90
    pop_risk_low: float = POP_RISK_LOW
    pop_risk_high: float = POP_RISK_HIGH
    max_open_cache_bytes: int = 8 * 1024 * 1024 * 1024
    budget_policy: str = "fail"

    def __post_init__(self) -> None:
        finite = (
            self.pop_strength,
            self.adaptive_pop_max,
            self.risk_quantile,
            self.pop_risk_low,
            self.pop_risk_high,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ScenePlanError("scene planner numeric configuration must be finite")
        if not 0.25 <= self.pop_strength <= 2.0:
            raise ScenePlanError("pop_strength must be in [0.25, 2.0]")
        if not self.pop_strength <= self.adaptive_pop_max <= 2.0:
            raise ScenePlanError(
                "adaptive_pop_max must be at least pop_strength and at most 2.0"
            )
        if self.zero_plane not in {"subject", "median", "background"}:
            raise ScenePlanError(
                "zero_plane must be subject, median, or background"
            )
        for name in (
            "lookbehind_depth_updates",
            "lookahead_depth_updates",
            "duplicate_pulse_distance_updates",
            "settle_depth_updates",
            "minimum_scene_frames",
        ):
            if getattr(self, name) < 0:
                raise ScenePlanError(f"{name} must be non-negative")
        if self.minimum_scene_frames < 1:
            raise ScenePlanError("minimum_scene_frames must be at least one")
        if not 0.0 < self.risk_quantile <= 1.0:
            raise ScenePlanError("risk_quantile must be in (0, 1]")
        if not 0.0 <= self.pop_risk_low < self.pop_risk_high:
            raise ScenePlanError("pop-risk endpoints must be ordered and non-negative")
        if self.max_open_cache_bytes < 0:
            raise ScenePlanError("max_open_cache_bytes must be non-negative")
        if self.budget_policy not in {"fail", "split"}:
            raise ScenePlanError("budget_policy must be fail or split")


@dataclass
class _ProposalCluster:
    proposal_indices: list[int]
    proposal_frame_ids: list[str]
    first_update_ordinal: int
    last_update_ordinal: int


def _finite(value: Any, default: float = -1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    result = float(value)
    return result if math.isfinite(result) else default


def _flag(frame: Mapping[str, Any], name: str, bit: int) -> bool:
    direct = frame.get(name)
    if isinstance(direct, bool):
        return direct
    packed = frame.get("analysis_flags")
    if (
        isinstance(packed, (int, float)) and not isinstance(packed, bool) and
        math.isfinite(float(packed)) and float(packed).is_integer() and
        0 <= int(packed) < (1 << 24)
    ):
        return bool(int(packed) & bit)
    return False


def _quantile(values: Sequence[float], q: float) -> float:
    """Deterministic linear quantile matching NumPy's default interpolation."""
    if not values:
        raise ScenePlanError("cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _smoothstep(low: float, high: float, value: float) -> float:
    t = min(max((value - low) / (high - low), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _appearance_qualified(frame: Mapping[str, Any]) -> bool:
    if _flag(frame, "appearance_veto", ANALYSIS_APPEARANCE_VETO):
        return False
    for name in (
        "current_structural_support_fraction",
        "previous_structural_support_fraction",
        "common_structural_support_fraction",
    ):
        if name in frame and _finite(frame.get(name)) < STRUCTURAL_COLOR_MIN_SUPPORT:
            return False
    return (
        _finite(frame.get("current_depth_change_fraction")) >=
        DEPTH_CUT_CORROBORATE and
        _finite(frame.get("raw_rgb_change_fraction")) >= RAW_RGB_CUT_HIGH and
        _finite(frame.get("structural_change_fraction")) >=
        STRUCTURAL_COLOR_CUT_HIGH
    )


def _geometry_qualified(frame: Mapping[str, Any]) -> bool:
    return (
        not _flag(frame, "appearance_veto", ANALYSIS_APPEARANCE_VETO) and
        _finite(frame.get("current_depth_change_fraction")) >= DEPTH_CUT_HIGH
    )


def _evidence_score(frame: Mapping[str, Any]) -> float:
    depth = max(_finite(frame.get("current_depth_change_fraction"), 0.0), 0.0)
    raw = max(_finite(frame.get("raw_rgb_change_fraction"), 0.0), 0.0)
    structural = max(
        _finite(frame.get("structural_change_fraction"), 0.0), 0.0
    )
    geometry = depth / DEPTH_CUT_HIGH
    appearance = min(
        depth / DEPTH_CUT_CORROBORATE,
        raw / RAW_RGB_CUT_HIGH,
        structural / STRUCTURAL_COLOR_CUT_HIGH,
    )
    return max(geometry, appearance)


def _frame_summary(frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sequence": int(frame["source_index"]) + 1,
        "frame_id": str(frame["frame_id"]),
        "depth_change_fraction": _finite(
            frame.get("current_depth_change_fraction")
        ),
        "raw_rgb_change_fraction": _finite(
            frame.get("raw_rgb_change_fraction")
        ),
        "structural_change_fraction": _finite(
            frame.get("structural_change_fraction")
        ),
        "score": _evidence_score(frame),
        "appearance_qualified": _appearance_qualified(frame),
        "geometry_qualified": _geometry_qualified(frame),
        "appearance_veto": _flag(
            frame, "appearance_veto", ANALYSIS_APPEARANCE_VETO
        ),
    }


class StreamingScenePlanner:
    """Finalize scenes incrementally while retaining only the current unresolved scene."""

    def __init__(self, config: ScenePlannerConfig):
        self.config = config
        self._frames: list[dict[str, Any]] = []
        self._pending: list[_ProposalCluster] = []
        self._next_source_index = 0
        self._depth_update_ordinal = -1
        self._scene_number = 0
        self._semantic_scene_number = 1
        self._open_cache_bytes = 0
        self._boundary_revisions: list[dict[str, Any]] = []
        self._closed = False

    @property
    def open_start_sequence(self) -> int:
        if self._frames:
            return int(self._frames[0]["source_index"]) + 1
        return self._next_source_index + 1

    @property
    def open_cache_bytes(self) -> int:
        return self._open_cache_bytes

    @property
    def pending_proposal_count(self) -> int:
        return sum(len(cluster.proposal_indices) for cluster in self._pending)

    @property
    def boundary_revisions(self) -> tuple[dict[str, Any], ...]:
        """Every accepted, moved, merged, rejected, and administrative decision so far."""
        return tuple(copy.deepcopy(item) for item in self._boundary_revisions)

    def feed(
        self,
        frame: Mapping[str, Any],
        *,
        frame_cache_bytes: int = 0,
    ) -> list[dict[str, Any]]:
        if self._closed:
            raise ScenePlanError("cannot feed a finalized scene planner")
        if frame_cache_bytes < 0:
            raise ScenePlanError("frame_cache_bytes must be non-negative")
        normalized = self._validate_frame(frame)
        normalized["_planner_cache_bytes"] = frame_cache_bytes
        if normalized["depth_updated"]:
            self._depth_update_ordinal += 1
        normalized["_planner_depth_update_ordinal"] = self._depth_update_ordinal
        self._frames.append(normalized)
        self._open_cache_bytes += frame_cache_bytes
        self._next_source_index += 1

        # Held-depth source frames reuse the complete SubjectState, including its one-update pulse.
        # A proposal is an estimator update event, never every color frame that happens to reuse it.
        if normalized["depth_updated"] and normalized["hard_cut_pulse"]:
            self._add_proposal(normalized)

        finalized: list[dict[str, Any]] = []
        finalized.extend(self._resolve_mature_clusters(eof=False))
        finalized.extend(self._enforce_budget())
        return finalized

    def finish(self) -> list[dict[str, Any]]:
        if self._closed:
            raise ScenePlanError("scene planner was already finalized")
        if not self._frames and self._next_source_index == 0:
            raise ScenePlanError("cannot finalize an empty clip")
        finalized = self._resolve_mature_clusters(eof=True)
        if self._frames:
            finalized.append(
                self._finalize_prefix(
                    len(self._frames),
                    boundary={
                        "proposal_sequences": [],
                        "proposal_frame_ids": [],
                        "final_sequence": None,
                        "accepted": True,
                        "decision": "end_of_stream",
                        "reason": "the final scene ends at the source EOF",
                        "confidence": None,
                        "evidence_window": None,
                        "revision_updates": 0,
                        "truncated": False,
                        "budget_forced": False,
                        "semantic_cut": False,
                    },
                )
            )
        self._pending.clear()
        self._closed = True
        return finalized

    def _validate_frame(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(frame, Mapping):
            raise ScenePlanError("scene planner frame must be a mapping")
        source_index = frame.get("source_index")
        if (
            not isinstance(source_index, int) or isinstance(source_index, bool) or
            source_index != self._next_source_index
        ):
            raise ScenePlanError(
                "scene planner requires contiguous source_index values: "
                f"got {source_index!r}, expected {self._next_source_index}"
            )
        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id.isdigit():
            raise ScenePlanError("scene planner frame_id must be a decimal string")
        for name in ("depth_updated", "hard_cut_pulse"):
            if not isinstance(frame.get(name), bool):
                raise ScenePlanError(f"scene planner {name} must be boolean")
        return dict(frame)

    def _add_proposal(self, frame: Mapping[str, Any]) -> None:
        index = int(frame["source_index"])
        ordinal = int(frame["_planner_depth_update_ordinal"])
        if ordinal < 0:
            raise ScenePlanError("a cut pulse cannot precede the first depth update")
        # Merge every proposal whose two-sided evidence window overlaps the current cluster.
        # Maturity below waits the same lookbehind+lookahead reach, so a future overlapping pulse
        # cannot arrive after an immutable boundary was already emitted.
        overlap_reach = (
            self.config.lookbehind_depth_updates +
            self.config.lookahead_depth_updates
        )
        if (
            self._pending and
            ordinal - self._pending[-1].last_update_ordinal <=
            max(self.config.duplicate_pulse_distance_updates, overlap_reach)
        ):
            cluster = self._pending[-1]
            cluster.proposal_indices.append(index)
            cluster.proposal_frame_ids.append(str(frame["frame_id"]))
            cluster.last_update_ordinal = ordinal
        else:
            self._pending.append(
                _ProposalCluster(
                    proposal_indices=[index],
                    proposal_frame_ids=[str(frame["frame_id"])],
                    first_update_ordinal=ordinal,
                    last_update_ordinal=ordinal,
                )
            )

    def _resolve_mature_clusters(self, *, eof: bool) -> list[dict[str, Any]]:
        finalized: list[dict[str, Any]] = []
        while self._pending:
            cluster = self._pending[0]
            mature = (
                self._depth_update_ordinal >=
                cluster.last_update_ordinal +
                self.config.lookahead_depth_updates +
                self.config.lookbehind_depth_updates
            )
            if not eof and not mature:
                break
            self._pending.pop(0)
            scene = self._resolve_cluster(cluster, eof=eof and not mature)
            if scene is not None:
                finalized.append(scene)
        return finalized

    def _resolve_cluster(
        self,
        cluster: _ProposalCluster,
        *,
        eof: bool,
    ) -> dict[str, Any] | None:
        if not self._frames:
            return None
        open_start = int(self._frames[0]["source_index"])
        next_cluster_start = (
            self._pending[0].proposal_indices[0] if self._pending else None
        )
        minimum_boundary = open_start + self.config.minimum_scene_frames
        left_ordinal = (
            cluster.first_update_ordinal -
            self.config.lookbehind_depth_updates
        )
        right_ordinal = (
            cluster.last_update_ordinal +
            self.config.lookahead_depth_updates
        )

        candidates: list[dict[str, Any]] = []
        cluster_indices = set(cluster.proposal_indices)
        for position, frame in enumerate(self._frames):
            source_index = int(frame["source_index"])
            ordinal = int(frame["_planner_depth_update_ordinal"])
            if not frame["depth_updated"]:
                continue
            if ordinal < left_ordinal or ordinal > right_ordinal:
                continue
            if source_index < minimum_boundary:
                continue
            if next_cluster_start is not None:
                # Leave enough frames for a distinct nearby scene instead of allowing the later
                # production pulse to steal this proposal's boundary.
                if source_index > (
                    next_cluster_start - self.config.minimum_scene_frames
                ):
                    continue
            is_proposal = source_index in cluster_indices
            appearance = _appearance_qualified(frame)
            geometry = _geometry_qualified(frame)
            relative = _flag(
                frame,
                "relative_geometry_spike",
                ANALYSIS_RELATIVE_GEOMETRY_SPIKE,
            )
            evidence_available = all(
                _finite(frame.get(name)) >= 0.0
                for name in (
                    "current_depth_change_fraction",
                    "raw_rgb_change_fraction",
                    "structural_change_fraction",
                )
            )
            if not (appearance or geometry or relative):
                # A legacy trace with no exported evidence may retain its production pulse as a
                # conservative fallback. Schema-3 evidence, when present, must support the pulse.
                if not (is_proposal and not evidence_available):
                    continue
            if not is_proposal and not (appearance or geometry):
                continue
            if _flag(frame, "appearance_veto", ANALYSIS_APPEARANCE_VETO):
                continue
            score = _evidence_score(frame)
            nearest_proposal_distance = min(
                abs(source_index - proposal)
                for proposal in cluster.proposal_indices
            )
            candidates.append({
                "position": position,
                "source_index": source_index,
                "frame": frame,
                "score": score,
                "is_proposal": is_proposal,
                "appearance_qualified": appearance,
                "geometry_qualified": geometry,
                "relative_geometry_spike": relative,
                "nearest_proposal_distance": nearest_proposal_distance,
            })

        return_evidence = any(
            _flag(frame, "same_scene_gap_return", ANALYSIS_SAME_SCENE_RETURN)
            for frame in self._frames
            if left_ordinal <=
               int(frame["_planner_depth_update_ordinal"]) <= right_ordinal and
               int(frame["source_index"]) > cluster.proposal_indices[-1]
        )
        all_proposals_vetoed = all(
            any(
                int(frame["source_index"]) == proposal and
                _flag(frame, "appearance_veto", ANALYSIS_APPEARANCE_VETO)
                for frame in self._frames
            )
            for proposal in cluster.proposal_indices
        )
        if all_proposals_vetoed and return_evidence:
            # A supported flash/strobe return is the one rejection the offline policy can prove
            # from the currently exported evidence.  Other uncertain pulses remain conservative.
            self._boundary_revisions.append(copy.deepcopy({
                "proposal_sequences": [
                    index + 1 for index in cluster.proposal_indices
                ],
                "proposal_frame_ids": list(cluster.proposal_frame_ids),
                "final_sequence": None,
                "accepted": False,
                "decision": "rejected_supported_flash_return",
                "reason": (
                    "every proposal was appearance-vetoed and future evidence "
                    "returned to the supported left endpoint"
                ),
                "truncated": eof,
                "budget_forced": False,
                "semantic_cut": False,
            }))
            return None

        if not candidates:
            proposal_evidence_available = all(
                any(
                    int(frame["source_index"]) == proposal and
                    all(
                        _finite(frame.get(name)) >= 0.0
                        for name in (
                            "current_depth_change_fraction",
                            "raw_rgb_change_fraction",
                            "structural_change_fraction",
                        )
                    )
                    for frame in self._frames
                )
                for proposal in cluster.proposal_indices
            )
            if proposal_evidence_available:
                self._boundary_revisions.append(copy.deepcopy({
                    "proposal_sequences": [
                        index + 1 for index in cluster.proposal_indices
                    ],
                    "proposal_frame_ids": list(cluster.proposal_frame_ids),
                    "final_sequence": None,
                    "accepted": False,
                    "decision": "rejected_unsupported_proposal",
                    "reason": (
                        "exported schema-3 evidence supported neither geometry, "
                        "appearance, nor the relative-geometry escape"
                    ),
                    "truncated": eof,
                    "budget_forced": False,
                    "semantic_cut": False,
                }))
                return None
            # Do not silently discard a production cut merely because optional diagnostics were
            # unavailable.  If minimum-length constraints make the pulse illegal, merge it into
            # the current scene and let the next distinct proposal decide.
            legal_proposals = [
                proposal for proposal in cluster.proposal_indices
                if proposal >= minimum_boundary
            ]
            if not legal_proposals:
                self._boundary_revisions.append(copy.deepcopy({
                    "proposal_sequences": [
                        index + 1 for index in cluster.proposal_indices
                    ],
                    "proposal_frame_ids": list(cluster.proposal_frame_ids),
                    "final_sequence": None,
                    "accepted": False,
                    "decision": "rejected_minimum_scene_length",
                    "reason": (
                        "the proposed boundary could not leave a legal scene prefix"
                    ),
                    "truncated": eof,
                    "budget_forced": False,
                    "semantic_cut": False,
                }))
                return None
            boundary_index = legal_proposals[0]
            selected = next(
                frame for frame in self._frames
                if int(frame["source_index"]) == boundary_index
            )
            confidence = None
            decision = (
                "merged_duplicate_proposals"
                if len(cluster.proposal_indices) > 1 else
                "confirmed_causal_fallback"
            )
            reason = (
                "no stronger non-vetoed correlated transition was available; "
                "the production pulse was retained conservatively"
            )
        else:
            # Highest score, then nearest to a production proposal, then earliest source frame.
            best = min(
                candidates,
                key=lambda item: (
                    -item["score"],
                    item["nearest_proposal_distance"],
                    item["source_index"],
                ),
            )
            boundary_index = int(best["source_index"])
            selected = best["frame"]
            confidence = None
            if boundary_index not in cluster_indices:
                decision = "moved_to_correlated_evidence"
                reason = (
                    "lookahead found stronger depth-corroborated structural/RGB replacement"
                )
            elif len(cluster.proposal_indices) > 1:
                decision = "merged_duplicate_proposals"
                reason = "nearby production pulses resolved to one deterministic boundary"
            else:
                decision = "confirmed"
                reason = "the production pulse remained the strongest legal transition"

        prefix_count = boundary_index - open_start
        right_count = len(self._frames) - prefix_count
        if (
            prefix_count < self.config.minimum_scene_frames or
            right_count < self.config.minimum_scene_frames
        ):
            self._boundary_revisions.append(copy.deepcopy({
                "proposal_sequences": [
                    index + 1 for index in cluster.proposal_indices
                ],
                "proposal_frame_ids": list(cluster.proposal_frame_ids),
                "final_sequence": None,
                "accepted": False,
                "decision": "rejected_minimum_scene_length",
                "reason": (
                    "the refined boundary could not leave legal scene prefixes "
                    "on both sides"
                ),
                "truncated": eof,
                "budget_forced": False,
                "semantic_cut": False,
            }))
            return None
        proposal_sequence = cluster.proposal_indices[0] + 1
        final_sequence = boundary_index + 1
        window_frames = [
            frame for frame in self._frames
            if left_ordinal <=
               int(frame["_planner_depth_update_ordinal"]) <= right_ordinal
        ]
        boundary = {
            "proposal_sequences": [
                index + 1 for index in cluster.proposal_indices
            ],
            "proposal_frame_ids": list(cluster.proposal_frame_ids),
            "final_sequence": final_sequence,
            "final_frame_id": str(selected["frame_id"]),
            "accepted": True,
            "decision": decision,
            "reason": reason,
            "confidence": confidence,
            "evidence_score": _evidence_score(selected),
            "confidence_kind": "uncalibrated-normalized-threshold-margin",
            "evidence_window": {
                "first_sequence": (
                    int(window_frames[0]["source_index"]) + 1
                    if window_frames else proposal_sequence
                ),
                "last_sequence": (
                    int(window_frames[-1]["source_index"]) + 1
                    if window_frames else proposal_sequence
                ),
                "lookbehind_depth_updates":
                    self.config.lookbehind_depth_updates,
                "lookahead_depth_updates":
                    self.config.lookahead_depth_updates,
                "candidate_count": len(candidates),
                "selected": _frame_summary(selected),
            },
            "revision_updates": (
                int(selected["_planner_depth_update_ordinal"]) -
                cluster.first_update_ordinal
            ),
            "revision_source_frames": final_sequence - proposal_sequence,
            "truncated": eof,
            "budget_forced": False,
            "semantic_cut": True,
        }
        self._boundary_revisions.append(copy.deepcopy(boundary))
        return self._finalize_prefix(prefix_count, boundary=boundary)

    def _enforce_budget(self) -> list[dict[str, Any]]:
        limit = self.config.max_open_cache_bytes
        if not limit or self._open_cache_bytes <= limit:
            return []
        if self.config.budget_policy == "fail":
            raise SceneCacheBudgetExceeded(
                limit_bytes=limit,
                live_bytes=self._open_cache_bytes,
                open_start_sequence=self.open_start_sequence,
                current_sequence=self._next_source_index,
            )
        if len(self._frames) <= self.config.minimum_scene_frames:
            raise SceneCacheBudgetExceeded(
                limit_bytes=limit,
                live_bytes=self._open_cache_bytes,
                open_start_sequence=self.open_start_sequence,
                current_sequence=self._next_source_index,
            )
        if self._pending:
            # Splitting away the proposal's source/evidence window would either erase an event
            # from the audit or force a decision without its promised lookahead. Even the opt-in
            # administrative policy therefore fails closed while a semantic proposal is open.
            raise SceneCacheBudgetExceeded(
                limit_bytes=limit,
                live_bytes=self._open_cache_bytes,
                open_start_sequence=self.open_start_sequence,
                current_sequence=self._next_source_index,
            )
        # Administrative splitting is explicit and does not increment the semantic-scene id.
        prefix_count = len(self._frames) - 1
        boundary_index = int(self._frames[prefix_count]["source_index"])
        boundary = {
            "proposal_sequences": [],
            "proposal_frame_ids": [],
            "final_sequence": boundary_index + 1,
            "final_frame_id": str(
                self._frames[prefix_count]["frame_id"]
            ),
            "accepted": False,
            "decision": "administrative_cache_split",
            "reason": (
                "the opt-in split policy bounded storage before a semantic cut"
            ),
            "confidence": None,
            "evidence_window": None,
            "revision_updates": 0,
            "truncated": True,
            "budget_forced": True,
            "semantic_cut": False,
        }
        self._boundary_revisions.append(copy.deepcopy(boundary))
        return [
            self._finalize_prefix(
                prefix_count,
                boundary=boundary,
                increment_semantic_scene=False,
            )
        ]

    def _finalize_prefix(
        self,
        prefix_count: int,
        *,
        boundary: dict[str, Any],
        increment_semantic_scene: bool = True,
    ) -> dict[str, Any]:
        if prefix_count <= 0 or prefix_count > len(self._frames):
            raise ScenePlanError("finalized scene prefix is empty or out of range")
        scene_frames = self._frames[:prefix_count]
        del self._frames[:prefix_count]
        released_bytes = sum(
            int(frame["_planner_cache_bytes"]) for frame in scene_frames
        )
        self._open_cache_bytes -= released_bytes
        self._scene_number += 1
        camera, evidence = self._camera_plan(scene_frames)
        start_sequence = int(scene_frames[0]["source_index"]) + 1
        end_sequence_exclusive = int(scene_frames[-1]["source_index"]) + 2
        first_pts = _finite(
            scene_frames[0].get(
                "source_pts_seconds",
                scene_frames[0].get("timestamp_seconds"),
            ),
            math.nan,
        )
        last_pts = _finite(
            scene_frames[-1].get(
                "source_pts_seconds",
                scene_frames[-1].get("timestamp_seconds"),
            ),
            math.nan,
        )
        last_duration = _finite(
            scene_frames[-1].get("duration_seconds"), math.nan)
        timing = None
        if (
            math.isfinite(first_pts) and math.isfinite(last_pts) and
            math.isfinite(last_duration) and last_duration > 0.0 and
            last_pts >= first_pts
        ):
            timing = {
                "start_pts_seconds": first_pts,
                "end_pts_seconds_exclusive": last_pts + last_duration,
                "duration_seconds": last_pts + last_duration - first_pts,
            }
        result = {
            "schema": SCENE_PLAN_SCHEMA,
            "version": SCENE_PLAN_VERSION,
            "scene_id": self._scene_number,
            "semantic_scene_id": self._semantic_scene_number,
            "start_sequence": start_sequence,
            "end_sequence_exclusive": end_sequence_exclusive,
            "frame_count": len(scene_frames),
            "first_frame_id": str(scene_frames[0]["frame_id"]),
            "last_frame_id": str(scene_frames[-1]["frame_id"]),
            "timing": timing,
            "cache_bytes": released_bytes,
            "boundary": boundary,
            "evidence": evidence,
            "absolute_pop_strength": camera["absolute_pop_strength"],
            "zero_anchor_shift_px": camera["zero_anchor_shift_px"],
            "render": camera,
            "semantics": {
                "depth": "causal-production-processed-r32f",
                "cut_state": "unrevised-v1",
                "known_limit": (
                    "a rejected or shifted causal cut can retain its original subject "
                    "recenter/stretch reset; scene pop and resolved anchor are corrected"
                ),
                "ground_truth": False,
                "comfort_optimal": False,
            },
        }
        if increment_semantic_scene and boundary.get("semantic_cut"):
            self._semantic_scene_number += 1
        return result

    def _camera_plan(
        self,
        frames: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        depth_updates = 0
        settled: list[Mapping[str, Any]] = []
        for frame in frames:
            if not frame["depth_updated"]:
                continue
            if depth_updates >= self.config.settle_depth_updates:
                settled.append(frame)
            depth_updates += 1

        def usable_depth(frame: Mapping[str, Any]) -> bool:
            return (
                _finite(frame.get("initialized"), 0.0) > 0.5 and
                _finite(frame.get("depth_ready"), 0.0) > 0.5 and
                _finite(frame.get("valid_depth_fraction"), 0.0) > 0.0 and
                _finite(frame.get("range_collapsed"), 1.0) < 0.5 and
                _finite(frame.get("scene_age"), -1.0) >=
                    self.config.settle_depth_updates
            )

        usable_settled = [frame for frame in settled if usable_depth(frame)]
        edge_values = [
            value
            for frame in usable_settled
            for value in [_finite(frame.get("current_edge_fraction"))]
            if value >= 0.0
        ]
        anchor_values = [
            value
            for frame in usable_settled
            if _finite(frame.get("zero_anchor_valid"), 0.0) > 0.5
            for value in [
                _finite(frame.get("current_zero_anchor_candidate_shift_px"))
            ]
            if -1.39635933 <= value <= 8.58230571
        ]

        if not self.config.adaptive_pop:
            absolute_pop = self.config.pop_strength
            pop_origin = "configured-fixed"
            risk_value = None
            pop_fallback = None
        elif edge_values:
            risk_value = _quantile(edge_values, self.config.risk_quantile)
            confidence = 1.0 - _smoothstep(
                self.config.pop_risk_low,
                self.config.pop_risk_high,
                risk_value,
            )
            absolute_pop = (
                self.config.pop_strength +
                (self.config.adaptive_pop_max - self.config.pop_strength) *
                confidence
            )
            absolute_pop = min(
                max(absolute_pop, self.config.pop_strength),
                self.config.adaptive_pop_max,
            )
            pop_origin = "whole-finalized-scene-edge-risk"
            pop_fallback = None
        else:
            absolute_pop = self.config.pop_strength
            risk_value = None
            pop_origin = "conservative-floor-fallback"
            pop_fallback = "no settled valid instantaneous edge-risk samples"

        if anchor_values:
            anchor = _quantile(anchor_values, 0.50)
            anchor_origin = "whole-finalized-scene-median-candidate"
            anchor_fallback = None
        else:
            production = [
                _finite(frame.get("zero_anchor_shift_px"))
                for frame in frames
                if _finite(frame.get("zero_anchor_valid"), 0.0) > 0.5
            ]
            production = [
                value for value in production
                if -1.39635933 <= value <= 8.58230571
            ]
            anchor = production[-1] if production else 0.0
            anchor_origin = (
                "production-latched-fallback" if production else
                "neutral-fallback"
            )
            anchor_fallback = "no settled valid zero-anchor candidate samples"

        valid_depth_changes = [
            value
            for frame in frames
            for value in [
                _finite(frame.get("current_depth_change_fraction"))
            ]
            if value >= 0.0
        ]
        veto_count = sum(
            _flag(frame, "appearance_veto", ANALYSIS_APPEARANCE_VETO)
            for frame in frames
        )
        evidence = {
            "source_frame_count": len(frames),
            "depth_update_count": depth_updates,
            "settled_depth_update_count": len(settled),
            "usable_settled_depth_update_count": len(usable_settled),
            "valid_edge_sample_count": len(edge_values),
            "valid_anchor_sample_count": len(anchor_values),
            "excluded_edge_sample_count": len(settled) - len(edge_values),
            "excluded_anchor_sample_count": len(settled) - len(anchor_values),
            "appearance_veto_count": veto_count,
            "edge_fraction": (
                {
                    "p50": _quantile(edge_values, 0.50),
                    "p90": _quantile(edge_values, 0.90),
                    "p95": _quantile(edge_values, 0.95),
                    "max": max(edge_values),
                }
                if edge_values else None
            ),
            "zero_anchor_candidate_shift_px": (
                {
                    "p10": _quantile(anchor_values, 0.10),
                    "p50": _quantile(anchor_values, 0.50),
                    "p90": _quantile(anchor_values, 0.90),
                }
                if anchor_values else None
            ),
            "depth_change_max": (
                max(valid_depth_changes) if valid_depth_changes else None
            ),
        }
        camera = {
            "absolute_pop_strength": absolute_pop,
            "pop_objective": (
                "stable scene-wide gain derived from a conservative high quantile of "
                "settled production edge risk"
            ),
            "pop_origin": pop_origin,
            "pop_fallback": pop_fallback,
            "risk_quantile": (
                self.config.risk_quantile if risk_value is not None else None
            ),
            "risk_value": risk_value,
            "pop_range": {
                "floor": self.config.pop_strength,
                "ceiling": self.config.adaptive_pop_max,
            },
            "zero_plane_mode": self.config.zero_plane,
            "zero_anchor_shift_px": anchor,
            "zero_objective": (
                "one resolved source-pixel convergence shift minimizing L1 movement "
                "across settled candidates in the finalized scene"
            ),
            "zero_origin": anchor_origin,
            "zero_fallback": anchor_fallback,
            "override_pop": True,
            "override_anchor": True,
        }
        return camera, evidence


def native_scene_plan_document(scene: Mapping[str, Any]) -> dict[str, Any]:
    """Create the immutable one-scene document accepted by native cache replay."""
    required = {
        "start_sequence",
        "end_sequence_exclusive",
        "absolute_pop_strength",
        "zero_anchor_shift_px",
    }
    if not isinstance(scene, Mapping) or not required.issubset(scene):
        raise ScenePlanError("finalized scene lacks native render fields")
    return {
        "schema": SCENE_PLAN_SCHEMA,
        "version": SCENE_PLAN_VERSION,
        "cache_contract_schema": CACHE_CONTRACT_SCHEMA,
        "scenes": [
            {
                "start_sequence": int(scene["start_sequence"]),
                "end_sequence_exclusive":
                    int(scene["end_sequence_exclusive"]),
                "absolute_pop_strength":
                    float(scene["absolute_pop_strength"]),
                "zero_anchor_shift_px":
                    float(scene["zero_anchor_shift_px"]),
            }
        ],
        "audit": dict(scene),
    }
