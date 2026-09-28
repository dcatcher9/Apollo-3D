"""Offline UI discovery: observations, pixels and human review are separate evidence.

This module reads no files and changes no rendering policy. The inspector supplies
decoded native artifacts and optional SHA-256 identities. SDK numeric tags are
deliberately not decoded here: the dump's catalog is the source of candidate names.
"""

import html
import math
import re

import numpy as np


SCHEMA = "sunshine.game3d.ui-discovery.v1"
REVIEW_SCHEMA = "sunshine.game3d.ui-review.v1"
PRIORITY = {"explicit_ui_alpha": 1, "explicit_ui_layer": 2, "backbuffer_alpha": 3,
            "hudless_support": 4, "non_authoritative_hint": 5, "unknown": 6}
SEMANTICS = {"ui_alpha": "explicit_ui_alpha", "ui_color_and_alpha": "explicit_ui_layer",
             "final_game_color_before_fg_candidate_not_verified_ui_mask": "backbuffer_alpha",
             "uninterpreted_source_alpha": "backbuffer_alpha", "color_without_ui": "hudless_support"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
OPACITY_FORMATS = {"R": {41, 54, 56, 61}, "A": {2, 10, 24, 28, 29, 87, 91}}


def _object(value):
    return value if isinstance(value, dict) else {}


def _rows(value):
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _matches(row, item):
    stem = row.get("file_stem")
    identity = row.get("artifact_id")
    return ((stem is not None and stem == item.get("kind", item.get("file_stem"))) or
            (identity is not None and identity == item.get("artifact_id")))


def _channel_facts(value):
    finite = value[np.isfinite(value)]
    count, invalid = int(finite.size), int(value.size - finite.size)
    low = float(finite.min()) if count else None
    high = float(finite.max()) if count else None
    positive = int(np.count_nonzero(finite > 0))
    opaque = int(np.count_nonzero(finite == 1))
    if invalid:
        classification = "nonfinite"
    elif not count:
        classification = "empty_allocation"
    elif low < 0 or high > 1:
        classification = "out_of_range"
    elif low == high == 0:
        classification = "empty"
    elif positive == count:
        classification = "full"
    else:
        classification = "selective"
    return dict(classification=classification, finite_count=count, nonfinite_count=invalid,
                min=low, max=high, mean=float(finite.mean(dtype=np.float64)) if count else None,
                constant=bool(count and low == high), positive_count=positive, opaque_count=opaque,
                coverage=positive / int(value.size) if value.size else None,
                note="Coverage counts finite positive samples. Full includes translucent positive coverage; "
                     "numeric selectivity does not establish UI semantics.")


def artifact_facts(descriptor, decoded_value=None, error=None, status=None, artifact_sha256=None):
    """Summarize already-decoded native channels without thresholds or color transforms.

    The caller's decoder must expose only real components (no synthesized alpha).
    A failed file read can use status='missing' or 'unreadable'; errors retain text.
    """
    result = {key: descriptor.get(key) for key in
              ("kind", "artifact_id", "width", "height", "dxgi_format")}
    result["artifact_sha256"] = artifact_sha256
    result["status"] = status or ("captured" if decoded_value is not None else
                                  "missing" if isinstance(error, FileNotFoundError) else "unreadable")
    if error is not None:
        result["reason"] = str(error)
    if decoded_value is not None:
        value = np.asarray(decoded_value)
        if value.ndim != 3 or not 1 <= value.shape[-1] <= 4:
            raise ValueError("Decoded UI artifacts require one to four native channels")
        result["channels"] = {name: _channel_facts(value[:, :, i])
                              for i, name in enumerate("RGBA"[:value.shape[-1]])}
    return result


def _catalog(metadata):
    latest = _object(metadata.get("latest_observations"))
    if "ui_resources" in latest:
        return _rows(latest["ui_resources"]), latest, ("latest_observations.ui_resources" if
                                                      isinstance(latest["ui_resources"], list) else "malformed")
    if isinstance(metadata.get("ui_resources"), list):
        return _rows(metadata["ui_resources"]), latest, "ui_resources"
    return [], latest, "unavailable"


def _category(row):
    semantic = row.get("semantic", "")
    if semantic in SEMANTICS:
        return SEMANTICS[semantic]
    if semantic:
        return "non_authoritative_hint"
    return "unknown"


def _observation(row, raw_color=False):
    raw = row.get("state")
    state = {"unobserved": "not_observed", "null": "explicit_null",
             "non_null": "observed_non_null", "snapshot_unavailable": "metadata_unavailable"}.get(raw, raw)
    if raw_color:
        state = "render_artifact"
    observations = _rows(row.get("observations"))
    def order(item):
        stamp = _object(item.get("observation"))
        return tuple(value if isinstance(value, (int, float)) else 0 for value in
                     (stamp.get("tick_ms", 0), stamp.get("sequence", 0)))
    newest = max(observations, key=order) if observations else {}
    return dict(status=state or "metadata_unavailable", producer_state=raw,
                runtime_support=row.get("runtime_support", "unknown"),
                resource_capture_status=newest.get("resource_capture_status", row.get("resource_capture_status")),
                observations=observations)


def _capture(row, captures, artifacts, facts, errors, raw_color=False):
    captured = next((item for item in captures if _matches(row, item)), {})
    descriptor = next((item for item in artifacts if _matches(row, item)), {})
    fact = next((item for item in facts if _matches(row, item)), {})
    host_errors = [item for item in errors if _matches(row, item)]
    raw = captured.get("status")
    if fact.get("status") in ("captured", "missing", "unreadable"):
        state = fact["status"]
    elif host_errors:
        state = "capture_rejected"
    elif descriptor or raw == "captured":
        state = "missing"
    elif raw in ("pending_gpu", "recorded", "submitted", "gpu_completion_timeout"):
        state = "pending"
    elif raw in ("not_observed", "explicit_null"):
        state = raw
    elif raw == "unreadable_or_query_failed":
        state = "query_failed"
    elif raw:
        state = "capture_rejected"
    else:
        state = "metadata_unavailable"
    return dict(status=state, producer_status=raw, artifact_status=fact.get("status", "not_listed"),
                metadata=captured, descriptor=descriptor, host_errors=host_errors,
                acquisition_scope="render_snapshot" if raw_color else "optional_api_snapshot"), fact


def _content(category, fact):
    channel = "R" if category == "explicit_ui_alpha" else "A" if category in (
        "explicit_ui_layer", "backbuffer_alpha") else None
    channels = _object(fact.get("channels"))
    if channel is None:
        return dict(channel=None, classification="supporting_evidence", channels=channels)
    result = channels.get(channel)
    return dict(channel=channel, **(result if isinstance(result, dict) else
                                   dict(classification="channel_unavailable")))


def _review(candidate, review, manifest_sha256, source_color_sha256):
    """Validate explicitly supplied snapshot review; never infer it from pixels."""
    key = candidate["candidate"]
    entries = [item for item in _rows(_object(review).get("candidates")) if item.get("candidate") == key]
    if not entries:
        return dict(status="pending", reasons=["No explicit human review for this snapshot."])
    reasons = []
    entry = entries[0]
    if len(entries) != 1:
        reasons.append("Multiple reviews target this candidate.")
    if review.get("schema") != REVIEW_SCHEMA:
        reasons.append("Unknown human review schema.")
    if not isinstance(manifest_sha256, str) or not _SHA256.fullmatch(manifest_sha256):
        reasons.append("The inspector did not supply a valid manifest SHA-256.")
    elif review.get("manifest_sha256") != manifest_sha256:
        reasons.append("Review belongs to a different manifest.")
    digest = candidate.get("artifact_sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        reasons.append("The inspector did not supply a valid artifact SHA-256.")
    elif entry.get("artifact_sha256") != digest:
        reasons.append("Review belongs to different artifact bytes.")
    if entry.get("verdict") not in ("usable", "not_ui", "inconclusive"):
        reasons.append("Review verdict must be usable, not_ui or inconclusive.")
    if entry.get("scene_context") not in ("gameplay", "menu", "no_ui"):
        reasons.append("Review must identify gameplay, menu or no_ui context.")
    if not isinstance(entry.get("notes"), str) or len(entry["notes"].strip()) < 20:
        reasons.append("Review needs notes describing the observed UI and scene evidence.")
    if reasons:
        return dict(status="review_invalid", reasons=reasons, review=entry)
    if entry["verdict"] != "usable":
        return dict(status="not_ui" if entry["verdict"] == "not_ui" else "inconclusive",
                    reasons=[entry["notes"]], review=entry)
    for name in ("mask_alignment_checked", "scene_exclusion_checked"):
        if entry.get(name) is not True:
            reasons.append(f"Human review must explicitly confirm {name}.")
    if not isinstance(source_color_sha256, str) or not _SHA256.fullmatch(source_color_sha256):
        reasons.append("Readable source color with a verified SHA-256 is required for scene review.")
    elif review.get("source_color_sha256") != source_color_sha256:
        reasons.append("Review belongs to different source color bytes.")
    classification = candidate["content"]["classification"]
    if classification == "full" and entry["scene_context"] == "gameplay" and entry.get("full_screen_ui") is not True:
        reasons.append("Full coverage during gameplay requires verified full_screen_ui context.")
    if classification == "full" and entry["scene_context"] == "no_ui":
        reasons.append("Full coverage contradicts a no_ui scene review.")
    if classification == "empty" and entry["scene_context"] != "no_ui":
        reasons.append("Empty coverage needs a no_ui scene review; it cannot establish coverage of visible UI.")
    if reasons:
        return dict(status="inconclusive", reasons=reasons, review=entry)
    return dict(status="human_reviewed_snapshot", reasons=[entry["notes"]], review=entry)


def _qualification(candidate, review, manifest_sha256, source_color_sha256):
    checked = _review(candidate, review, manifest_sha256, source_color_sha256)
    blockers = []
    if candidate["category"] not in ("explicit_ui_alpha", "explicit_ui_layer", "backbuffer_alpha"):
        blockers.append("This resource is supporting evidence, not an admitted UI-opacity candidate.")
    if candidate["capture"]["status"] != "captured":
        blockers.append("Readable native candidate pixels are unavailable.")
    if candidate["capture"]["host_errors"]:
        blockers.append("The host reported an acquisition failure for this artifact; "
                        "surviving readable bytes cannot override that failure.")
    if candidate["content"]["classification"] not in ("empty", "full", "selective"):
        blockers.append("The required opacity channel is unavailable, nonfinite or outside [0,1].")
    channel = candidate["content"]["channel"]
    if candidate["capture"]["descriptor"].get("dxgi_format") not in OPACITY_FORMATS.get(channel, set()):
        blockers.append("The native format is not admitted for this opacity channel; "
                        "integer hints are not normalized opacity.")
    if candidate["sdk_contract"]["status"] in ("ambiguous", "unsupported", "incompatible"):
        blockers.append("The SDK catalog binding is explicitly ambiguous or unsupported.")
    capture = candidate["capture"]["metadata"]
    if _object(capture.get("observation")).get("sdk_success") is False:
        blockers.append("The captured SDK operation failed.")
    # A file can survive a failed transport or capture decision. Keep its content
    # inspectable without silently treating the contradictory capture as safe.
    if capture.get("status") not in (None, "captured"):
        blockers.append("The producer did not publish a completed capture decision for these pixels.")
    if blockers and checked["status"] == "human_reviewed_snapshot":
        checked["status"] = "inconclusive"
    checked["blockers"] = blockers
    checked["scope"] = "this_artifact_in_this_dump_only"
    checked["live_authorization"] = False
    return checked


def _next_action(candidate):
    observation = candidate["observation"]["status"]
    capture = candidate["capture"]["status"]
    if candidate["observation"].get("resource_capture_status") == "not_attempted_depth_owned":
        return ("The interface was queried; optional capture was suppressed by SL depth ownership. "
                "Capture eligibility needs a separate provider change.")
    if observation in ("query_failed", "getter_unavailable"):
        return ("Inspect the recorded getter and SDK result for this feature; "
                "a failed query does not establish absent UI.")
    if capture == "capture_rejected":
        return ("Resolve this capture's admission/transport failure using its diagnostic and host error; "
                "then repeat the dump.")
    if capture == "pending":
        return "Repeat the dump after GPU completion; this timed snapshot has no completed readable candidate."
    if capture in ("missing", "unreadable"):
        return ("Recover or recapture the native artifact; a producer capture record is not proof "
                "that readable pixels reached this package.")
    if observation == "explicit_null":
        return "Record another relevant gameplay/menu mode; this API observation explicitly supplied no resource."
    if capture != "captured":
        return "Verify hook coverage and an armed observation window, then capture the relevant feature and game mode."
    if candidate["category"] == "hudless_support":
        return ("Inspect with verified corresponding final color, rectangle and composition evidence; "
                "HUDless color alone is not an opacity mask.")
    if candidate["category"] in ("non_authoritative_hint", "unknown"):
        return "Keep as diagnostic evidence; establish a separate UI-specific contract before considering protection."
    if candidate["qualification"]["status"] == "human_reviewed_snapshot":
        return ("Snapshot meaning was reviewed. Test UI movement, hidden UI and menus in additional captures "
                "before considering any game-wide policy.")
    if candidate["content"]["classification"] == "full":
        return ("Review against the game image: full coverage may be a fullscreen menu or opaque scene data; "
                "neither the tag nor statistics decides.")
    if candidate["content"]["classification"] == "empty":
        return ("Review the scene context: empty coverage can correctly describe no UI; "
                "obtain a visible-UI capture to test coverage.")
    return ("Review mask alignment and scene exclusion against corresponding game evidence, "
            "then record an explicit snapshot review.")


def select_qualified_source(candidates):
    """Choose a reviewed offline snapshot only; this is never a live selection."""
    eligible = [item for item in candidates if
                item.get("category") in ("explicit_ui_alpha", "explicit_ui_layer", "backbuffer_alpha") and
                _object(item.get("qualification")).get("status") == "human_reviewed_snapshot" and
                not _object(item.get("qualification")).get("blockers") and
                _object(item.get("capture")).get("status") == "captured" and
                _object(item.get("content")).get("classification") in ("empty", "full", "selective")]
    return (min(eligible, key=lambda item: (PRIORITY[item["category"]], item["candidate"]))["candidate"]
            if eligible else None)


def discover_ui(manifest, facts, review=None, *, manifest_sha256=None):
    """Build a JSON-safe report from a manifest and artifact_facts results."""
    metadata = _object(manifest.get("producer_metadata"))
    rows, latest, location = _catalog(metadata)
    artifacts = _rows(manifest.get("artifacts"))
    captures = _rows(metadata.get("optional_captures"))
    errors = _rows(manifest.get("optional_capture_errors"))
    facts = _rows(facts)
    source_color = next((item for item in facts if item.get("kind") == "source_color" and
                         item.get("status") == "captured"), {})
    source_color_sha256 = source_color.get("artifact_sha256") if any(
        item.get("kind") == "source_color" for item in artifacts) else None
    replay = _object(metadata.get("replay"))
    fg = _object(replay.get("source_alpha_ui_fg_mode"))
    gaps = []
    if location in ("unavailable", "malformed"):
        gaps.append("UI observation catalog is unavailable; "
                    "missing provider entries are not evidence of absent resources.")
    if latest.get("status") in ("busy", "serialization-limit", "not-armed"):
        gaps.append(f"Middleware observation window is {latest['status']}.")
    if latest.get("truncated") or latest.get("dropped_updates"):
        gaps.append("Middleware observations were truncated or dropped; unobserved candidates remain inconclusive.")
    if any(item.get("stage") == "producer_metadata" for item in errors):
        gaps.append("Optional capture metadata was omitted by the producer.")
    # Legacy packages can expose optional artifacts/copy metadata without their
    # catalog. Preserve evidence without reverse-engineering SDK names from IDs.
    known = list(rows)
    for item in captures:
        if not any(_matches(row, item) for row in known):
            known.append(dict(item, state="snapshot_unavailable"))
    for artifact in artifacts:
        if artifact.get("kind") not in ("source_color", "ui_source_color", "raw_depth", "candidate",
                                        "vertical_majorant", "vertical_field", "final_field", "sbs", "linear_color"):
            if not any(_matches(row, artifact) for row in known):
                known.append(dict(file_stem=artifact.get("kind"), artifact_id=artifact.get("artifact_id"),
                                  state="snapshot_unavailable"))
    if any(item.get("kind") == "source_color" for item in artifacts):
        known.append(dict(file_stem="source_color", name="Presented game color alpha", provider="render",
                          semantic="uninterpreted_source_alpha", role="color_alpha"))
    candidates = []
    for row in known:
        key = row.get("file_stem")
        if not isinstance(key, str) or not key:
            gaps.append("A catalog entry has no usable file_stem; its candidate identity cannot be resolved.")
            continue
        raw_color = key == "source_color"
        category = _category(row)
        captured, fact = _capture(row, captures, artifacts, facts, errors, raw_color)
        candidate = dict(candidate=key, provider=row.get("provider", "unknown"), name=row.get("name", key),
                         category=category, priority=PRIORITY[category], semantic=row.get("semantic", "unknown"),
                         catalog=row, observation=_observation(row, raw_color), capture=captured,
                         content=_content(category, fact), artifact_sha256=fact.get("artifact_sha256"),
                         sdk_contract=dict(status=row.get("sdk_contract_status", "unverified"),
                                           authority="producer catalog; numeric SDK tags are not decoded here"),
                         frame_association=dict(status="same_color_allocation" if raw_color else "unverified",
                                                evidence=_object(captured["metadata"].get("observation")),
                                                note=("Raw color alpha shares its RGB allocation; "
                                                      "real/generated frame identity still needs evidence.")
                                                if raw_color else
                                                ("Independent API snapshot; matching dimensions, sequences or a human "
                                                 "semantic review do not prove association with rendered color.")),
                         consumption=dict(status="see_consumed_input" if raw_color else
                                          "optional_snapshot_not_consumed"))
        candidate["live_eligibility"] = dict(
            status=("unavailable_generated_color" if fg.get("known") is True and fg.get("enabled") is True else
                    "requires_live_policy" if fg.get("known") is True and fg.get("enabled") is False else
                    "unverified_real_frame") if raw_color else "not_established_by_optional_snapshot",
            note="Offline review never admits this snapshot for live use; "
                 "provider eligibility, scope and freshness need independent checks.")
        candidate["qualification"] = _qualification(candidate, review, manifest_sha256, source_color_sha256)
        candidate["next_action"] = _next_action(candidate)
        candidates.append(candidate)
    candidates.sort(key=lambda item: (item["priority"], item["candidate"]))
    consumed_kind = replay.get("ui_alpha_source", "unknown")
    consumed_fact = next((item for item in facts if item.get("kind") == consumed_kind), {})
    consumed_channel = _object(replay.get("ui_constant_binding")).get("mask_channel", "alpha")
    selected = select_qualified_source(candidates)
    readable = [item for item in candidates if item["priority"] <= 3 and item["capture"]["status"] == "captured"]
    blocked = [item for item in candidates if item["priority"] <= 3 and item["capture"]["status"] in
               ("capture_rejected", "pending", "missing", "unreadable")]
    state = ("human_reviewed_snapshot_available" if selected else "semantic_inconclusive" if readable else
             "capture_blocked" if blocked else "metadata_unavailable" if gaps else
             "no_qualified_mask_in_observed_window")
    enabled = replay.get("source_alpha_ui")
    return dict(schema=SCHEMA, manifest_sha256=manifest_sha256, source_color_sha256=source_color_sha256,
                catalog_location=location, evidence_gaps=gaps,
                provider_evidence=dict(modules=latest.get("modules", []),
                                       observation_status=latest.get("status", "unavailable"),
                                       streamline=latest.get("streamline", []), ngx=latest.get("ngx", []),
                                       frame_generation=latest.get("frame_generation", []),
                                       note="Provider inventory describes this dump's observer coverage only; "
                                            "omitted interfaces are unassessed."),
                candidates=candidates,
                consumed_input=dict(artifact=consumed_kind, enabled=enabled if isinstance(enabled, bool) else None,
                                    channel=consumed_channel, artifact_status=consumed_fact.get("status", "not_listed"),
                                    content=_object(consumed_fact.get("channels")).get(
                                        "R" if consumed_channel == "red" else "A"),
                                    provenance=metadata.get("ui_source", {}),
                                    frame_association="unverified" if consumed_kind == "ui_source_color" else
                                    "same_color_allocation" if consumed_kind == "source_color" else "not_applicable",
                                    note="Consumed pixels and constants are render facts, "
                                         "not proof of semantic qualification or same-game-frame pairing."),
                summary=dict(status=state, selected_snapshot=selected, live_policy_changed=False,
                             qualified_scope="this_artifact_in_this_dump_only", candidate_count=len(candidates),
                             note="No qualified source means insufficient evidence in this dump, "
                                  "not proof that this game has no UI mask."))


def review_template(report):
    """Create an intentionally incomplete, digest-bound human review worksheet."""
    return dict(schema=REVIEW_SCHEMA, manifest_sha256=report.get("manifest_sha256"),
                source_color_sha256=report.get("source_color_sha256"), candidates=[
        dict(candidate=item["candidate"], artifact_sha256=item.get("artifact_sha256"), verdict="inconclusive",
             scene_context=None, mask_alignment_checked=False, scene_exclusion_checked=False,
             full_screen_ui=False, notes="")
        for item in report.get("candidates", []) if item.get("priority", 6) <= 3 and
        _object(item.get("capture")).get("status") == "captured"])


def _markdown(value):
    text = "unknown" if value is None else str(value)
    # Provider names, errors and reviewer notes are data, not Markdown or HTML.
    return re.sub(r"([\\`*_{}\[\]()#+.!|>-])", r"\\\1", html.escape(text)).replace("\r", " ").replace("\n", " ")


def _coverage(value):
    return f"{100 * value:.2f}%" if isinstance(value, (int, float)) and math.isfinite(value) else "-"


def format_report(report):
    """Render the discovery facts as a readable, non-authorizing Markdown report."""
    summary = _object(report.get("summary"))
    consumed = _object(report.get("consumed_input"))
    lines = ["# Game 3D UI discovery", "", f"Result: {_markdown(summary.get('status'))}.", "",
             "This report qualifies individual native snapshots only. It does not change rendering, "
             "authorize a live source, or establish frame correspondence.", "",
             f"Offline reviewed snapshot: {_markdown(summary.get('selected_snapshot') or 'none')}.",
             f"Render input: {_markdown(consumed.get('artifact'))}; "
             f"protection enabled: {_markdown(consumed.get('enabled'))}; "
             f"channel: {_markdown(consumed.get('channel'))}; file: {_markdown(consumed.get('artifact_status'))}.",
             f"Consumed mask content: {_markdown(_object(consumed.get('content')).get('classification'))}; "
             f"positive coverage: {_coverage(_object(consumed.get('content')).get('coverage'))}.",
             f"Render input association: {_markdown(consumed.get('frame_association'))}.", "",
             "## Candidate order", "",
             "| Candidate | Observation | Capture | Channel | Content | Coverage | Qualification |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for item in _rows(report.get("candidates")):
        cells = (item.get("candidate"), _object(item.get("observation")).get("status"),
                 _object(item.get("capture")).get("status"), _object(item.get("content")).get("channel"),
                 _object(item.get("content")).get("classification"),
                 _coverage(_object(item.get("content")).get("coverage")),
                 _object(item.get("qualification")).get("status"))
        lines.append("| " + " | ".join(_markdown(value) for value in cells) + " |")
    for item in _rows(report.get("candidates")):
        lines.extend(["", f"### {_markdown(item.get('candidate'))}", "",
                      f"Declared meaning: {_markdown(item.get('semantic'))}. "
                      f"SDK binding: {_markdown(_object(item.get('sdk_contract')).get('status'))}.", ""])
        qualification = _object(item.get("qualification"))
        reasons = qualification.get("reasons", []) + qualification.get("blockers", [])
        for reason in reasons:
            lines.append("- " + _markdown(reason))
        lines.extend(["", "Next action: " + _markdown(item.get("next_action")),
                      "", "Live eligibility: " + _markdown(_object(item.get("live_eligibility")).get("status"))])
    lines.extend(["", "## Evidence gaps", ""])
    for gap in report.get("evidence_gaps", []):
        lines.append("- " + _markdown(gap))
    if not report.get("evidence_gaps"):
        lines.append("No package-level omission was reported. "
                     "Unobserved interfaces and unverified frame correspondence remain unassessed.")
    lines.extend(["", "## Review workflow", "",
                  "Inspect native mask/alpha previews with the game image in gameplay, hidden-UI and menu captures. "
                  "Record alignment, scene exclusion and scene context in a review template bound to this manifest, "
                  "candidate bytes and source-color bytes. Full coverage can be legitimate fullscreen UI; "
                  "empty coverage can be legitimate absence of UI. "
                  "Neither content pattern proves meaning by itself.", "",
                  "A human-reviewed snapshot is limited to this artifact in this dump. Additional game modes, "
                  "UI movement, SDK binding, live freshness and actual frame correspondence require separate evidence.",
                  ""])
    return "\n".join(lines)
