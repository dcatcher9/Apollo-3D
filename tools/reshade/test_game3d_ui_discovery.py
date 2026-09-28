import copy
import json
import unittest

import numpy as np

import game3d_ui_discovery as discovery


MANIFEST_SHA = "a" * 64
ARTIFACT_SHA = "b" * 64
SOURCE_SHA = "c" * 64


class UIDiscoveryTest(unittest.TestCase):
    def candidate(self, semantic="ui_alpha", kind="sl_ui_alpha", values=(0, 1), state="non_null",
                  capture="captured", role="alpha", provider="streamline"):
        row = dict(file_stem=kind, name=kind, semantic=semantic, state=state, role=role, provider=provider)
        channels = 4 if role == "color_alpha" else 1
        pixels = np.zeros((1, len(values), channels), dtype=np.float32)
        pixels[:, :, -1] = values
        descriptor = dict(kind=kind, width=len(values), height=1, dxgi_format=2 if channels == 4 else 41)
        source = dict(kind="source_color", width=2, height=1, dxgi_format=28)
        manifest = dict(artifacts=[descriptor, source], producer_metadata=dict(
            latest_observations=dict(status="observed-window", ui_resources=[row]),
            optional_captures=[dict(file_stem=kind, status=capture)]))
        facts = [discovery.artifact_facts(descriptor, pixels, artifact_sha256=ARTIFACT_SHA)]
        facts.append(discovery.artifact_facts(source, np.ones((1, 2, 4), np.float32), artifact_sha256=SOURCE_SHA))
        return manifest, facts

    def report(self, manifest, facts, review=None):
        return discovery.discover_ui(manifest, facts, review, manifest_sha256=MANIFEST_SHA)

    def review(self, kind="sl_ui_alpha", context="gameplay", verdict="usable", **changes):
        item = dict(candidate=kind, artifact_sha256=ARTIFACT_SHA, verdict=verdict,
                    scene_context=context, mask_alignment_checked=True, scene_exclusion_checked=True,
                    notes="Compared HUD icons, scene edges and mask overlay in this snapshot.")
        item.update(changes)
        return dict(schema=discovery.REVIEW_SCHEMA, manifest_sha256=MANIFEST_SHA,
                    source_color_sha256=SOURCE_SHA, candidates=[item])

    def test_hogwarts_opaque_scene_is_never_qualified_by_tag_or_capture(self):
        manifest, facts = self.candidate("ui_color_and_alpha", "sl_ui_color_alpha", (1, 1), role="color_alpha")
        report = self.report(manifest, facts)
        item = report["candidates"][0]
        self.assertEqual(item["content"]["classification"], "full")
        self.assertEqual(item["qualification"]["status"], "pending")
        self.assertIsNone(report["summary"]["selected_snapshot"])
        reviewed = self.report(manifest, facts, self.review("sl_ui_color_alpha"))
        self.assertEqual(reviewed["candidates"][0]["qualification"]["status"], "inconclusive")
        self.assertIn("full_screen_ui", reviewed["candidates"][0]["qualification"]["reasons"][0])

    def test_full_menu_and_empty_no_ui_can_be_reviewed_without_numeric_rejection(self):
        for values, context in (((1, 1), "menu"), ((0, 0), "no_ui")):
            with self.subTest(context=context):
                manifest, facts = self.candidate(values=values)
                report = self.report(manifest, facts, self.review(context=context))
                self.assertEqual(report["summary"]["selected_snapshot"], "sl_ui_alpha")
                qualification = report["candidates"][0]["qualification"]
                self.assertEqual(qualification["status"], "human_reviewed_snapshot")
                self.assertFalse(qualification["live_authorization"])
                self.assertEqual(qualification["scope"], "this_artifact_in_this_dump_only")

    def test_selective_mask_requires_review_and_full_gameplay_requires_context(self):
        manifest, facts = self.candidate()
        self.assertIsNone(self.report(manifest, facts)["summary"]["selected_snapshot"])
        self.assertEqual(self.report(manifest, facts, self.review())["summary"]["selected_snapshot"], "sl_ui_alpha")
        manifest, facts = self.candidate(values=(1, 1))
        reviewed = self.report(manifest, facts, self.review(full_screen_ui=True))
        self.assertEqual(reviewed["summary"]["selected_snapshot"], "sl_ui_alpha")
        self.assertIsNone(self.report(manifest, facts, self.review(context="no_ui"))["summary"]["selected_snapshot"])

    def test_observation_and_capture_failures_are_not_collapsed_into_absence(self):
        expected = (("unobserved", "not_observed", "not_observed", "not_observed"),
                    ("null", "explicit_null", "explicit_null", "explicit_null"),
                    ("query_failed", "unreadable_or_query_failed", "query_failed", "query_failed"),
                    ("getter_unavailable", None, "getter_unavailable", "metadata_unavailable"),
                    ("non_null", "unsupported_descriptor", "observed_non_null", "capture_rejected"),
                    ("non_null", "gpu_completion_timeout", "observed_non_null", "pending"),
                    ("snapshot_unavailable", None, "metadata_unavailable", "metadata_unavailable"))
        for observed, captured, observation_result, capture_result in expected:
            with self.subTest(observed=observed, captured=captured):
                manifest, _ = self.candidate(state=observed, capture=captured)
                manifest["artifacts"] = []
                item = self.report(manifest, [])["candidates"][0]
                self.assertEqual(item["observation"]["status"], observation_result)
                self.assertEqual(item["capture"]["status"], capture_result)
                self.assertNotEqual(item["qualification"]["status"], "not_ui")

    def test_capture_claim_missing_file_and_unreadable_file_remain_different(self):
        manifest, facts = self.candidate()
        descriptor = manifest["artifacts"][0]
        missing = discovery.artifact_facts(descriptor, error=FileNotFoundError("gone"))
        unreadable = discovery.artifact_facts(descriptor, error=ValueError("bad row bytes"))
        for fact, state in ((missing, "missing"), (unreadable, "unreadable")):
            with self.subTest(state=state):
                item = self.report(manifest, [fact])["candidates"][0]
                self.assertEqual(item["capture"]["status"], state)
                self.assertEqual(item["capture"]["producer_status"], "captured")
        manifest["artifacts"] = []
        report = self.report(manifest, [])
        self.assertEqual(report["candidates"][0]["capture"]["status"], "missing")

    def test_host_readback_failure_retains_capture_metadata(self):
        manifest, _ = self.candidate()
        manifest["artifacts"] = []
        manifest["optional_capture_errors"] = [dict(kind="sl_ui_alpha", stage="readback", reason="device lost")]
        item = self.report(manifest, [])["candidates"][0]
        self.assertEqual(item["capture"]["host_errors"][0]["stage"], "readback")
        self.assertEqual(item["capture"]["producer_status"], "captured")
        self.assertIsNone(discovery.select_qualified_source([item]))

    def test_surviving_pixels_cannot_override_host_acquisition_failure(self):
        manifest, facts = self.candidate()
        manifest["optional_capture_errors"] = [dict(kind="sl_ui_alpha", stage="readback", reason="device lost")]
        report = self.report(manifest, facts, self.review())
        item = report["candidates"][0]
        self.assertEqual(item["capture"]["status"], "captured")
        self.assertEqual(item["content"]["classification"], "selective")
        self.assertTrue(any("host reported" in text for text in item["qualification"]["blockers"]))
        self.assertIsNone(report["summary"]["selected_snapshot"])

    def test_hints_and_hudless_are_never_promoted_even_with_usable_review(self):
        for semantic in ("color_without_ui", "skip_warp_hint", "generic_alpha_not_ui_mask",
                         "transparent_effect_opacity_not_ui_mask", "future_semantic"):
            with self.subTest(semantic=semantic):
                manifest, facts = self.candidate(semantic=semantic)
                report = self.report(manifest, facts, self.review())
                self.assertIsNone(report["summary"]["selected_snapshot"])
                item = next(item for item in report["candidates"] if item["candidate"] == "sl_ui_alpha")
                self.assertTrue(item["qualification"]["blockers"])

    def test_capture_optional_and_consumed_inputs_are_separate(self):
        manifest, facts = self.candidate()
        consumed = dict(kind="ui_source_color", width=2, height=1, dxgi_format=41)
        manifest["artifacts"].append(consumed)
        manifest["producer_metadata"].update(
            replay=dict(ui_alpha_source="ui_source_color", ui_constant_binding=dict(mask_channel="red")),
            ui_source=dict(sequence=99, association="latest_completed_real_input_approximation"))
        facts.append(discovery.artifact_facts(consumed, np.ones((1, 2, 1), np.float32)))
        report = self.report(manifest, facts, self.review())
        candidate = report["candidates"][0]
        self.assertEqual(candidate["consumption"]["status"], "optional_snapshot_not_consumed")
        self.assertEqual(candidate["content"]["classification"], "selective")
        self.assertEqual(report["consumed_input"]["content"]["classification"], "full")
        self.assertEqual(report["consumed_input"]["provenance"]["sequence"], 99)
        self.assertEqual(report["consumed_input"]["frame_association"], "unverified")

    def test_optional_matching_frame_numbers_never_prove_pairing(self):
        manifest, facts = self.candidate()
        manifest["producer_metadata"]["render_identity"] = dict(presentation_ordinal=42)
        manifest["producer_metadata"]["optional_captures"][0]["observation"] = dict(sequence=42, frame_numeric=42)
        report = self.report(manifest, facts, self.review())
        self.assertEqual(report["candidates"][0]["frame_association"]["status"], "unverified")
        self.assertEqual(report["candidates"][0]["qualification"]["status"], "human_reviewed_snapshot")

    def test_ambiguous_sdk_contract_is_preserved_and_blocks_review(self):
        manifest, facts = self.candidate()
        row = manifest["producer_metadata"]["latest_observations"]["ui_resources"][0]
        row.update(tag_type=68, sdk_contract_status="ambiguous")
        report = self.report(manifest, facts, self.review())
        self.assertEqual(report["candidates"][0]["catalog"]["tag_type"], 68)
        self.assertEqual(report["candidates"][0]["sdk_contract"]["status"], "ambiguous")
        self.assertIsNone(report["summary"]["selected_snapshot"])
        # Changing a numeric value alone cannot relabel a producer's semantic.
        row["tag_type"] = 69
        self.assertEqual(self.report(manifest, facts)["candidates"][0]["category"], "explicit_ui_alpha")

    def test_review_is_digest_bound_and_does_not_trust_qualified_label(self):
        manifest, facts = self.candidate()
        for mutation in (lambda review: review.update(manifest_sha256="c" * 64),
                         lambda review: review["candidates"][0].update(artifact_sha256="c" * 64),
                         lambda review: review["candidates"][0].update(verdict="qualified"),
                         lambda review: review["candidates"][0].update(notes="yes"),
                         lambda review: review.update(schema="future")):
            review = self.review()
            mutation(review)
            result = self.report(manifest, facts, review)
            self.assertEqual(result["candidates"][0]["qualification"]["status"], "review_invalid")
            self.assertIsNone(result["summary"]["selected_snapshot"])
        result = self.report(manifest, facts, self.review(mask_alignment_checked=False))
        self.assertEqual(result["candidates"][0]["qualification"]["status"], "inconclusive")

    def test_nonfinite_out_of_range_and_missing_channels_cannot_qualify(self):
        for values in ((0, np.nan), (0, np.inf), (0, 255)):
            with self.subTest(values=values):
                manifest, facts = self.candidate(values=values)
                report = self.report(manifest, facts, self.review())
                self.assertIsNone(report["summary"]["selected_snapshot"])
                json.dumps(report, allow_nan=False)
        manifest, facts = self.candidate(semantic="ui_color_and_alpha")
        self.assertEqual(self.report(manifest, facts)["candidates"][0]["content"]["classification"],
                         "channel_unavailable")

    def test_legacy_missing_catalog_keeps_unknown_evidence_and_gap(self):
        manifest, facts = self.candidate()
        del manifest["producer_metadata"]["latest_observations"]
        manifest["producer_metadata"]["optional_captures"] = []
        report = self.report(manifest, facts)
        self.assertEqual(report["catalog_location"], "unavailable")
        self.assertTrue(report["evidence_gaps"])
        candidate = next(item for item in report["candidates"] if item["candidate"] == "sl_ui_alpha")
        self.assertEqual(candidate["category"], "unknown")
        self.assertEqual(candidate["capture"]["status"], "captured")
        self.assertEqual(candidate["observation"]["status"], "metadata_unavailable")

    def test_nested_catalog_wins_over_legacy_and_suppressed_capture_is_explicit(self):
        manifest, facts = self.candidate()
        manifest["producer_metadata"]["ui_resources"] = [dict(file_stem="wrong")]
        row = manifest["producer_metadata"]["latest_observations"]["ui_resources"][0]
        row["observations"] = [dict(observation=dict(tick_ms=42, sequence=2),
                                    resource_capture_status="not_attempted_depth_owned"),
                               dict(observation=dict(tick_ms=41, sequence=1), resource_capture_status="attempted")]
        report = self.report(manifest, facts)
        self.assertEqual(report["catalog_location"], "latest_observations.ui_resources")
        self.assertEqual(len(report["candidates"]), 2)
        self.assertIn("depth ownership", report["candidates"][0]["next_action"])

    def test_selection_prefers_explicit_alpha_and_does_not_mutate_input(self):
        manifests, facts, reviews = [], [], []
        for kind, semantic, role in (("layer", "ui_color_and_alpha", "color_alpha"),
                                     ("alpha", "ui_alpha", "alpha"),
                                     ("color", "final_game_color_before_fg_candidate_not_verified_ui_mask",
                                      "color_alpha")):
            manifest, these = self.candidate(kind=kind, semantic=semantic, role=role)
            manifests.append(manifest)
            facts.extend(these)
            reviews.extend(self.review(kind)["candidates"])
        combined = dict(artifacts=sum((m["artifacts"] for m in manifests), []), producer_metadata=dict(
            latest_observations=dict(ui_resources=sum((m["producer_metadata"]["latest_observations"]["ui_resources"]
                                                       for m in manifests), [])),
            optional_captures=sum((m["producer_metadata"]["optional_captures"] for m in manifests), [])))
        before = copy.deepcopy(combined)
        report = self.report(combined, facts, dict(schema=discovery.REVIEW_SCHEMA, manifest_sha256=MANIFEST_SHA,
                                                source_color_sha256=SOURCE_SHA, candidates=reviews))
        self.assertEqual([item["candidate"] for item in report["candidates"]],
                         ["alpha", "layer", "color", "source_color"])
        self.assertEqual(report["summary"]["selected_snapshot"], "alpha")
        self.assertEqual(combined, before)

    def test_template_is_incomplete_and_bound_to_dump(self):
        manifest, facts = self.candidate()
        report = self.report(manifest, facts)
        template = discovery.review_template(report)
        self.assertEqual(template["manifest_sha256"], MANIFEST_SHA)
        self.assertEqual(template["source_color_sha256"], SOURCE_SHA)
        self.assertEqual(template["candidates"][0]["artifact_sha256"], ARTIFACT_SHA)
        self.assertIsNone(self.report(manifest, facts, template)["summary"]["selected_snapshot"])

    def test_positive_translucent_pixels_cover_full_frame(self):
        manifest, facts = self.candidate(values=(0.5, 0.5))
        report = self.report(manifest, facts, self.review())
        content = report["candidates"][0]["content"]
        self.assertEqual(content["classification"], "full")
        self.assertEqual(content["coverage"], 1)
        self.assertEqual(content["opaque_count"], 0)
        self.assertIsNone(report["summary"]["selected_snapshot"])

    def test_uint_channel_is_not_admitted_opacity_despite_zero_one_values(self):
        manifest, facts = self.candidate()
        manifest["artifacts"][0]["dxgi_format"] = 62
        report = self.report(manifest, facts, self.review())
        self.assertEqual(report["candidates"][0]["content"]["classification"], "selective")
        self.assertIsNone(report["summary"]["selected_snapshot"])
        self.assertTrue(any("native format" in text for text in report["candidates"][0]["qualification"]["blockers"]))

    def test_source_color_review_is_bound_and_source_pixels_are_required(self):
        manifest, facts = self.candidate()
        review = self.review()
        review["source_color_sha256"] = "d" * 64
        self.assertIsNone(self.report(manifest, facts, review)["summary"]["selected_snapshot"])
        self.assertIsNone(self.report(manifest, facts[:1], self.review())["summary"]["selected_snapshot"])

    def test_enabled_uses_exact_replay_flag_and_raw_fg_live_eligibility_is_separate(self):
        manifest, facts = self.candidate()
        manifest["producer_metadata"]["replay"] = dict(ui_alpha_source="source_color", source_alpha_ui=False,
                                                      source_alpha_ui_fg_mode=dict(known=True, enabled=True))
        report = self.report(manifest, facts)
        self.assertIs(report["consumed_input"]["enabled"], False)
        raw = next(item for item in report["candidates"] if item["candidate"] == "source_color")
        self.assertEqual(raw["live_eligibility"]["status"], "unavailable_generated_color")
        del manifest["producer_metadata"]["replay"]["source_alpha_ui"]
        self.assertIsNone(self.report(manifest, facts)["consumed_input"]["enabled"])

    def test_malformed_canonical_catalog_cannot_revive_flat_stale_catalog(self):
        manifest, facts = self.candidate()
        manifest["producer_metadata"]["latest_observations"]["ui_resources"] = "bad"
        manifest["producer_metadata"]["ui_resources"] = [dict(file_stem="stale", semantic="ui_alpha")]
        report = self.report(manifest, facts)
        self.assertEqual(report["catalog_location"], "malformed")
        self.assertNotIn("stale", [item["candidate"] for item in report["candidates"]])
        self.assertTrue(report["evidence_gaps"])

    def test_markdown_report_escapes_untrusted_metadata_and_labels_scope(self):
        manifest, facts = self.candidate(kind="bad|name\n<img src=x>")
        report = self.report(manifest, facts)
        markdown = discovery.format_report(report)
        self.assertNotIn("<img", markdown)
        self.assertNotIn("bad|name", markdown)
        self.assertIn("Offline reviewed snapshot", markdown)
        self.assertIn("Render input", markdown)
        self.assertIn("this artifact in this dump", markdown)
        self.assertIn("Coverage", markdown)
        self.assertIn("50\\.00%", markdown)


if __name__ == "__main__":
    unittest.main()
