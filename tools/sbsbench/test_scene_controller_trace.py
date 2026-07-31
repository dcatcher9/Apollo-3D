import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import scene_controller_trace as trace  # noqa: E402


class SceneControllerTraceTests(unittest.TestCase):
    @staticmethod
    def _header(*, interval=2):
        return {
            "record": "header",
            "trace_schema": trace.TRACE_SCHEMA,
            "source": trace.TRACE_SOURCE,
            "capture": trace.TRACE_CAPTURE,
            "backend": "shadow_rules",
            "controller_schema":
                trace.controller_contract.SCHEMA_VERSION,
            "rule_revision":
                trace.controller_contract.RULE_REVISION,
            "ordered_abi_hash":
                trace.controller_contract.ORDERED_ABI_HASH,
            "global_out_fields": list(trace.GLOBAL_OUT_FIELDS),
            "rule_state_fields": list(trace.RULE_STATE_FIELDS),
            "config": {
                "model": "Depth Anything V2 Small",
                "depth_reuse_interval": interval,
                "active_roi_authority": False,
            },
        }

    @staticmethod
    def _rule_state(*, generation=1):
        values = []
        for field in trace.RULE_STATE_CONTRACT:
            initial = field["initial"]
            values.append(
                int(initial) if field["type"] == "uint32" else float(initial)
            )
        generation_index = trace.controller_contract.RULE_STATE_NAMES.index(
            "backend_generation")
        values[generation_index] = generation
        return values

    @classmethod
    def _frame(
        cls,
        index,
        *,
        interval=2,
        generation=1,
        available=True,
        force_update=False,
    ):
        depth_updated = index % interval == 0 or force_update
        controller_frame_id = (
            index if force_update else index - (index % interval)
        )
        if not available:
            return {
                "record": "frame",
                "frame_id": f"{index + 1:05d}",
                "source_index": index,
                "depth_updated": depth_updated,
                "snapshot_available": False,
                "controller_frame_id": None,
                "backend_generation": None,
                "shadow": True,
                "global_out": None,
                "rule_state": None,
            }
        return {
            "record": "frame",
            "frame_id": f"{index + 1:05d}",
            "source_index": index,
            "depth_updated": depth_updated,
            "snapshot_available": True,
            "controller_frame_id": controller_frame_id,
            "backend_generation": generation,
            "shadow": True,
            "global_out": [0.0] * len(trace.GLOBAL_OUT_CONTRACT),
            "rule_state": cls._rule_state(generation=generation),
        }

    @staticmethod
    def _write(path, header, frames):
        path.write_text(
            "\n".join(
                json.dumps(value, separators=(",", ":"), allow_nan=True)
                for value in [header, *frames]
            ) + "\n",
            encoding="utf-8",
        )

    def test_loads_exact_typed_trace_and_held_snapshot_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / trace.TRACE_NAME
            frames = [
                self._frame(index, force_update=index == 3)
                for index in range(4)
            ]
            self._write(path, self._header(), frames)
            header, decoded = trace.load_trace(
                path,
                expected_frame_ids=[frame["frame_id"] for frame in frames],
                expected_frame_count=4,
            )
        self.assertEqual(header["ordered_abi_hash"],
                         trace.controller_contract.ORDERED_ABI_HASH)
        self.assertEqual(decoded[1]["controller_frame_id"], 0)
        self.assertEqual(decoded[2]["controller_frame_id"], 2)
        self.assertEqual(decoded[3]["controller_frame_id"], 3)
        self.assertTrue(decoded[3]["depth_updated"])

    def test_terminal_non_cadence_frame_is_the_only_forced_update(self):
        expected_ids = ["00001", "00002", "00003", "00004"]
        decoder = trace.IncrementalSceneControllerTraceDecoder(expected_ids)
        decoder.feed_line(json.dumps(self._header(interval=2)))
        decoder.feed_line(json.dumps(self._frame(0, interval=2)))
        decoder.feed_line(json.dumps(self._frame(1, interval=2)))
        decoder.feed_line(json.dumps(self._frame(2, interval=2)))
        terminal = decoder.feed_line(json.dumps(
            self._frame(3, interval=2, force_update=True)
        ))
        self.assertIsNotNone(terminal)
        self.assertEqual(terminal["controller_frame_id"], 3)
        decoder.finalize(len(expected_ids))

        stale_terminal = trace.IncrementalSceneControllerTraceDecoder(
            expected_ids
        )
        stale_terminal.feed_line(json.dumps(self._header(interval=2)))
        for index in range(3):
            stale_terminal.feed_line(json.dumps(
                self._frame(index, interval=2)
            ))
        with self.assertRaisesRegex(
            trace.SceneControllerTraceError, "depth_reuse_interval"
        ):
            stale_terminal.feed_line(json.dumps(
                self._frame(3, interval=2)
            ))

        early_force = trace.IncrementalSceneControllerTraceDecoder(
            expected_ids
        )
        early_force.feed_line(json.dumps(self._header(interval=2)))
        early_force.feed_line(json.dumps(self._frame(0, interval=2)))
        with self.assertRaisesRegex(
            trace.SceneControllerTraceError, "depth_reuse_interval"
        ):
            early_force.feed_line(json.dumps(
                self._frame(1, interval=2, force_update=True)
            ))

    def test_shadow_trace_rejects_any_unavailable_snapshot(self):
        decoder = trace.IncrementalSceneControllerTraceDecoder(["00001"])
        decoder.feed_line(json.dumps(self._header(interval=1)))
        with self.assertRaisesRegex(
                trace.SceneControllerTraceError, "is unavailable"):
            decoder.feed_line(json.dumps(self._frame(
                0, interval=1, available=False)))

        decoder = trace.IncrementalSceneControllerTraceDecoder(
            ["00001", "00002"])
        decoder.feed_line(json.dumps(self._header(interval=1)))
        decoder.feed_line(json.dumps(self._frame(0, interval=1)))
        with self.assertRaisesRegex(
                trace.SceneControllerTraceError, "is unavailable"):
            decoder.feed_line(json.dumps(self._frame(
                1, interval=1, available=False)))

    def test_rejects_schema_hash_and_order_drift(self):
        mutations = []
        wrong_schema = self._header()
        wrong_schema["trace_schema"] += 1
        mutations.append(("trace_schema", wrong_schema))
        wrong_hash = self._header()
        wrong_hash["ordered_abi_hash"] = "0" * 64
        mutations.append(("ordered_abi_hash", wrong_hash))
        global_reordered = self._header()
        global_reordered["global_out_fields"][0:2] = reversed(
            global_reordered["global_out_fields"][0:2])
        mutations.append(("global_out_fields", global_reordered))
        rule_reordered = self._header()
        rule_reordered["rule_state_fields"][0:2] = reversed(
            rule_reordered["rule_state_fields"][0:2])
        mutations.append(("rule_state_fields", rule_reordered))
        for label, header in mutations:
            with self.subTest(label=label):
                decoder = trace.IncrementalSceneControllerTraceDecoder()
                with self.assertRaisesRegex(
                        trace.SceneControllerTraceError, "header contract"):
                    decoder.feed_line(json.dumps(header))

    def test_rejects_header_scalar_type_coercions(self):
        mutations = [
            ("trace_schema", True),
            ("controller_schema", float(
                trace.controller_contract.SCHEMA_VERSION)),
            ("source", 1),
        ]
        for key, value in mutations:
            with self.subTest(key=key):
                header = self._header()
                header[key] = value
                decoder = trace.IncrementalSceneControllerTraceDecoder()
                with self.assertRaisesRegex(
                        trace.SceneControllerTraceError, "header contract"):
                    decoder.feed_line(json.dumps(header))

    def test_rejects_field_count_finite_and_type_corruption(self):
        cases = []
        short_global = self._frame(0)
        short_global["global_out"].pop()
        cases.append(("global_out", short_global))
        nonfinite_global = self._frame(0)
        nonfinite_global["global_out"][0] = math.nan
        cases.append(("finite float32", nonfinite_global))
        wrong_uint_type = self._frame(0)
        generation_index = trace.controller_contract.RULE_STATE_NAMES.index(
            "backend_generation")
        wrong_uint_type["rule_state"][generation_index] = 1.0
        cases.append(("uint32 JSON integer", wrong_uint_type))
        non_integral_float = self._frame(0)
        state_kind_index = trace.controller_contract.RULE_STATE_NAMES.index(
            "state_kind")
        non_integral_float["rule_state"][state_kind_index] = 0.5
        cases.append(("uint32-valued float", non_integral_float))
        for message, frame in cases:
            with self.subTest(message=message):
                decoder = trace.IncrementalSceneControllerTraceDecoder(["00001"])
                decoder.feed_line(json.dumps(self._header()))
                with self.assertRaisesRegex(
                        trace.SceneControllerTraceError, message):
                    decoder.feed_line(json.dumps(frame))

    def test_rejects_frame_backend_and_generation_identity_drift(self):
        cases = []
        wrong_frame = self._frame(0)
        wrong_frame["frame_id"] = "99999"
        cases.append(("frame identity", wrong_frame))
        wrong_controller = self._frame(1, force_update=True)
        wrong_controller["controller_frame_id"] = 0
        cases.append(("source identity", wrong_controller))
        wrong_generation = self._frame(0)
        generation_index = trace.controller_contract.RULE_STATE_NAMES.index(
            "backend_generation")
        wrong_generation["rule_state"][generation_index] = 2
        cases.append(("disagrees with rule_state", wrong_generation))
        for message, frame in cases:
            with self.subTest(message=message):
                expected_ids = ["00001"] if frame["source_index"] == 0 else [
                    "00001", "00002"
                ]
                decoder = trace.IncrementalSceneControllerTraceDecoder(
                    expected_ids)
                decoder.feed_line(json.dumps(self._header()))
                if frame["source_index"] == 1:
                    decoder.feed_line(json.dumps(self._frame(0)))
                with self.assertRaisesRegex(
                        trace.SceneControllerTraceError, message):
                    decoder.feed_line(json.dumps(frame))

    def test_descriptor_enforces_shadow_trace_and_off_means_no_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "active_roi_authority": False,
                "controller_schema":
                    trace.controller_contract.SCHEMA_VERSION,
                "rule_revision":
                    trace.controller_contract.RULE_REVISION,
                "ordered_abi_hash":
                    trace.controller_contract.ORDERED_ABI_HASH,
            }
            disabled = {
                **common,
                "enabled": False,
                "backend": "off",
                "transport": None,
                "file": None,
                "header_file": None,
                "frame_file": None,
                "retained_history": False,
                "trace_schema": None,
                "frame_count": 0,
            }
            value = trace.validate_descriptor(
                disabled,
                root,
                expected_frame_ids=["00001"],
                expected_model="Depth Anything V2 Small",
                expected_depth_reuse_interval=1,
                expected_backend="off",
            )
            self.assertFalse(value["enabled"])

            frame = self._frame(0, interval=1)
            self._write(
                root / trace.TRACE_NAME,
                self._header(interval=1),
                [frame],
            )
            enabled = {
                **common,
                "enabled": True,
                "backend": "shadow_rules",
                "transport": trace.TRACE_TRANSPORT,
                "file": trace.TRACE_NAME,
                "header_file": None,
                "frame_file": None,
                "retained_history": True,
                "trace_schema": trace.TRACE_SCHEMA,
                "frame_count": 1,
            }
            value = trace.validate_descriptor(
                enabled,
                root,
                expected_frame_ids=["00001"],
                expected_model="Depth Anything V2 Small",
                expected_depth_reuse_interval=1,
                expected_backend="shadow_rules",
            )
            self.assertEqual(value["validated_frame_count"], 1)

            wrong_type = dict(enabled)
            wrong_type["frame_count"] = True
            with self.assertRaisesRegex(
                    trace.SceneControllerTraceError,
                    "descriptor mismatch"):
                trace.validate_descriptor(
                    wrong_type,
                    root,
                    expected_frame_ids=["00001"],
                    expected_model="Depth Anything V2 Small",
                    expected_depth_reuse_interval=1,
                    expected_backend="shadow_rules",
                )
            wrong_type = dict(enabled)
            wrong_type["active_roi_authority"] = 0
            with self.assertRaisesRegex(
                    trace.SceneControllerTraceError,
                    "descriptor mismatch"):
                trace.validate_descriptor(
                    wrong_type,
                    root,
                    expected_frame_ids=["00001"],
                    expected_model="Depth Anything V2 Small",
                    expected_depth_reuse_interval=1,
                    expected_backend="shadow_rules",
                )

            disabled["backend"] = "off"
            with self.assertRaisesRegex(
                    trace.SceneControllerTraceError, "unexpectedly produced"):
                trace.validate_descriptor(
                    disabled,
                    root,
                    expected_frame_ids=["00001"],
                    expected_model="Depth Anything V2 Small",
                    expected_depth_reuse_interval=1,
                    expected_backend="off",
                )

    def test_atomic_latest_transport_validates_terminal_identity_without_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / trace.ATOMIC_HEADER_NAME).write_text(
                json.dumps(self._header(interval=2)) + "\n",
                encoding="utf-8",
            )
            (root / trace.ATOMIC_FRAME_NAME).write_text(
                json.dumps(
                    self._frame(3, interval=2, force_update=True)
                ) + "\n",
                encoding="utf-8",
            )
            descriptor = {
                "enabled": True,
                "backend": "shadow_rules",
                "active_roi_authority": False,
                "transport": trace.ATOMIC_TRACE_TRANSPORT,
                "file": None,
                "header_file": trace.ATOMIC_HEADER_NAME,
                "frame_file": trace.ATOMIC_FRAME_NAME,
                "retained_history": False,
                "trace_schema": trace.TRACE_SCHEMA,
                "controller_schema":
                    trace.controller_contract.SCHEMA_VERSION,
                "rule_revision":
                    trace.controller_contract.RULE_REVISION,
                "ordered_abi_hash":
                    trace.controller_contract.ORDERED_ABI_HASH,
                "frame_count": 4,
            }
            value = trace.validate_descriptor(
                descriptor,
                root,
                expected_frame_ids=[
                    "00001", "00002", "00003", "00004"
                ],
                expected_model="Depth Anything V2 Small",
                expected_depth_reuse_interval=2,
                expected_backend="shadow_rules",
            )
        self.assertFalse(value["retained_history"])
        self.assertEqual(value["validated_frame_count"], 4)


if __name__ == "__main__":
    unittest.main()
