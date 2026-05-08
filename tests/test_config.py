import unittest

import _bootstrap  # noqa: F401
from dataset_config import DATASETS, PROJECT_ROOT, available_datasets, dataset_paths, get_dataset


class DatasetConfigTests(unittest.TestCase):
    def test_known_datasets_exist(self):
        self.assertIn("bgl", DATASETS)
        self.assertIn("openssh", DATASETS)

    def test_paths_are_project_root_anchored(self):
        for dataset in DATASETS:
            paths = dataset_paths(dataset)
            for key in ["raw", "output_dir", "parsed", "scores", "alerts"]:
                self.assertTrue(paths[key].is_absolute(), f"{dataset}:{key} is not absolute")
                self.assertTrue(
                    str(paths[key]).startswith(str(PROJECT_ROOT)),
                    f"{dataset}:{key} is not under project root",
                )

    def test_unknown_dataset_raises_key_error(self):
        with self.assertRaises(KeyError):
            get_dataset("does_not_exist")

    def test_available_datasets_shape(self):
        rows = available_datasets()
        names = {row["name"] for row in rows}

        self.assertEqual(names, {"bgl", "openssh"})
        for row in rows:
            self.assertIn(row["status"], {"ready", "not_ready"})
            self.assertIn("files", row)
            self.assertIn("parsed", row["files"])


if __name__ == "__main__":
    unittest.main()
