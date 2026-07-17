from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import build_full_cadence_artistic_split as builder  # noqa: E402


def _active(native=True):
    rows = [{
        "production_id": "mono_training", "source_kind": "mono-video",
        "dataset_manifest": "train.json", "dataset_manifest_sha256": "a" * 64,
    }, {
        "production_id": "mono_development", "source_kind": "mono-video",
        "dataset_manifest": "dev.json", "dataset_manifest_sha256": "b" * 64,
    }, {
        "production_id": "mono_test", "source_kind": "mono-video",
        "dataset_manifest": "test.json", "dataset_manifest_sha256": "c" * 64,
    }]
    if native:
        rows.extend([{
            "production_id": "chug_native_pq_v1_training",
            "source_kind": "native-hdr-video",
            "dataset_manifest": "old-train.json",
            "dataset_manifest_sha256": "d" * 64,
        }, {
            "production_id": "chug_native_pq_v1_development",
            "source_kind": "native-hdr-video",
            "dataset_manifest": "old-dev.json",
            "dataset_manifest_sha256": "e" * 64,
        }])
    return {
        "schema": 1,
        "catalog": "catalog.json", "catalog_sha256": "f" * 64,
        "productions": rows,
    }


class FullCadenceSplitBuilderTest(unittest.TestCase):
    def test_only_sparse_chug_train_dev_are_removed(self):
        active = _active()
        catalog = {"sources": [{
            "id": row["production_id"],
            "production_id": row["production_id"],
            "source_kind": row["source_kind"],
        } for row in active["productions"]]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            path.write_text("{}", encoding="utf-8")
            with mock.patch.object(builder.combined, "_load", return_value=active), \
                    mock.patch.object(builder.combined, "_verified",
                                      side_effect=lambda *_args: Path(_args[-1])), \
                    mock.patch.object(builder.sources, "load_catalog",
                                      return_value=catalog):
                _, _, rows, manifests = builder._base_without_sparse_native(path)
        self.assertEqual({row["production_id"] for row in rows}, {
            "mono_training", "mono_development", "mono_test",
        })
        self.assertEqual(len(manifests), 3)

    def test_missing_retired_production_fails_closed(self):
        active = _active(native=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            path.write_text("{}", encoding="utf-8")
            with mock.patch.object(builder.combined, "_load", return_value=active), \
                    mock.patch.object(builder.combined, "_verified",
                                      return_value=Path("catalog.json")), \
                    mock.patch.object(builder.sources, "load_catalog",
                                      return_value={"sources": []}):
                with self.assertRaisesRegex(RuntimeError, "exactly the retired"):
                    builder._base_without_sparse_native(path)


if __name__ == "__main__":
    unittest.main()
