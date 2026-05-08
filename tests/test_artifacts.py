import json
import unittest

import pandas as pd

import _bootstrap  # noqa: F401
from dataset_config import DATASETS, dataset_paths, get_dataset


def artifacts_ready(dataset: str) -> bool:
    paths = dataset_paths(dataset)
    return paths["parsed"].exists() and paths["scores"].exists() and paths["metrics"].exists()


class ArtifactIntegrityTests(unittest.TestCase):
    def test_metrics_match_scores(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                if not artifacts_ready(dataset):
                    self.skipTest(f"{dataset} artifacts are not ready")

                paths = dataset_paths(dataset)
                scores = pd.read_parquet(paths["scores"], engine="pyarrow")
                metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))

                self.assertEqual(metrics["dataset"], dataset)
                self.assertEqual(metrics["total_windows"], len(scores))
                self.assertEqual(
                    metrics["predicted_anomalous_windows"],
                    int(scores["predicted"].sum()),
                )
                self.assertFalse(scores["anomaly_score"].isna().any())
                self.assertTrue(set(scores["predicted"].unique()).issubset({0, 1}))

    def test_feature_window_configuration(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                paths = dataset_paths(dataset)
                if not paths["features"].exists():
                    self.skipTest(f"{dataset} features are not ready")

                cfg = get_dataset(dataset)
                features = pd.read_parquet(
                    paths["features"],
                    columns=["window_mode", "window_size_sec", "step_size_sec"],
                    engine="pyarrow",
                )

                self.assertFalse(features.empty)
                self.assertEqual(set(features["window_mode"].unique()), {"sliding"})
                self.assertTrue((features["window_size_sec"] == int(cfg["window"])).all())
                self.assertTrue((features["step_size_sec"] == int(cfg["step"])).all())

    def test_alert_artifact_schemas(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                paths = dataset_paths(dataset)
                if not paths["alerts"].exists() or not paths["alerts_minilm"].exists():
                    self.skipTest(f"{dataset} alert artifacts are not ready")

                alerts = pd.read_parquet(paths["alerts"], engine="pyarrow")
                minilm = pd.read_parquet(paths["alerts_minilm"], engine="pyarrow")

                for column in ["window_start", "window_end", "severity", "top_template"]:
                    self.assertIn(column, alerts.columns)
                    self.assertIn(column, minilm.columns)

                # Old artifacts may not include this until alerts_minilm.py is rerun.
                if "top_level" in minilm.columns:
                    self.assertFalse(minilm["top_level"].isna().any())


if __name__ == "__main__":
    unittest.main()
