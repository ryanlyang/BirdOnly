from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixture_data import create_fixture, fixture_config
from setv.data.splits import SAFE_COLUMNS, build_splits, read_and_validate_metadata


class SplitTests(unittest.TestCase):
    def test_joint_group_split_is_deterministic_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, masks = create_fixture(root)
            config = fixture_config(dataset, masks, root / "output")
            metadata = read_and_validate_metadata(config)
            first = build_splits(metadata, config)
            second = build_splits(metadata, config)

            for name in ("candidate_train", "biased_val", "oracle_val", "test"):
                pd.testing.assert_frame_equal(
                    first.safe_manifests[name], second.safe_manifests[name]
                )
                self.assertEqual(
                    list(first.safe_manifests[name].columns), SAFE_COLUMNS
                )
                self.assertNotIn("place", first.safe_manifests[name].columns)
                self.assertNotIn("group", first.safe_manifests[name].columns)

            self.assertEqual(len(first.safe_manifests["candidate_train"]), 32)
            self.assertEqual(len(first.safe_manifests["biased_val"]), 8)
            self.assertEqual(
                first.summary["splits"]["biased_val"]["group_counts"],
                {
                    "y=0,place=0": 2,
                    "y=0,place=1": 2,
                    "y=1,place=0": 2,
                    "y=1,place=1": 2,
                },
            )
            self.assertEqual(
                sum(
                    first.summary["splits"]["biased_val"][
                        "group_proportions"
                    ].values()
                ),
                1.0,
            )
            self.assertEqual(
                set(first.protected_labels.columns),
                {"sample_id", "metadata_index", "split_name", "y", "place", "group"},
            )


if __name__ == "__main__":
    unittest.main()
