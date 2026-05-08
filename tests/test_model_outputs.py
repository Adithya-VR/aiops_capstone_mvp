import json
import unittest

import pandas as pd

import _bootstrap  # noqa: F401
from dataset_config import DATASETS, dataset_paths, get_dataset


def model_artifacts_ready(dataset: str) -> bool:
    paths = dataset_paths(dataset)
    return paths["scores"].exists() and paths["metrics"].exists()


class ModelOutputTests(unittest.TestCase):
    def test_scores_have_required_model_columns(self):
        required_columns = {
            "window_start",
            "window_end",
            "total_logs",
            "anomaly_count",
            "is_anomaly",
            "anomaly_score",
            "predicted",
            "window_mode",
            "window_size_sec",
            "step_size_sec",
        }

        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                if not model_artifacts_ready(dataset):
                    self.skipTest(f"{dataset} model artifacts are not ready")

                scores = pd.read_parquet(dataset_paths(dataset)["scores"], engine="pyarrow")

                self.assertTrue(required_columns.issubset(scores.columns))
                self.assertFalse(scores.empty)

    def test_scores_are_valid_and_predictions_are_binary(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                if not model_artifacts_ready(dataset):
                    self.skipTest(f"{dataset} model artifacts are not ready")

                scores = pd.read_parquet(dataset_paths(dataset)["scores"], engine="pyarrow")
                predicted_values = set(scores["predicted"].dropna().unique())

                self.assertTrue(predicted_values.issubset({0, 1}))
                self.assertFalse(scores["anomaly_score"].isna().any())
                self.assertFalse(scores["window_start"].isna().any())
                self.assertFalse(scores["window_end"].isna().any())
                self.assertTrue((scores["window_end"] > scores["window_start"]).all())
                self.assertGreater(scores["anomaly_score"].max(), scores["anomaly_score"].min())

                predicted_count = int(scores["predicted"].sum())
                self.assertGreater(predicted_count, 0)
                self.assertLess(predicted_count, len(scores))

    def test_metrics_match_model_outputs(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                if not model_artifacts_ready(dataset):
                    self.skipTest(f"{dataset} model artifacts are not ready")

                paths = dataset_paths(dataset)
                scores = pd.read_parquet(paths["scores"], engine="pyarrow")
                metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))

                self.assertEqual(metrics["dataset"], dataset)
                self.assertEqual(metrics["total_windows"], len(scores))
                self.assertEqual(
                    metrics["predicted_anomalous_windows"],
                    int(scores["predicted"].sum()),
                )
                self.assertEqual(
                    metrics["predicted_normal_windows"],
                    int((scores["predicted"] == 0).sum()),
                )
                self.assertAlmostEqual(
                    metrics["score_min"],
                    round(float(scores["anomaly_score"].min()), 4),
                    places=4,
                )
                self.assertAlmostEqual(
                    metrics["score_max"],
                    round(float(scores["anomaly_score"].max()), 4),
                    places=4,
                )

    def test_labeled_and_unlabeled_metric_modes_are_separate(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                if not model_artifacts_ready(dataset):
                    self.skipTest(f"{dataset} model artifacts are not ready")

                cfg = get_dataset(dataset)
                metrics = json.loads(
                    dataset_paths(dataset)["metrics"].read_text(encoding="utf-8")
                )

                self.assertEqual(
                    bool(metrics["ground_truth_available"]),
                    bool(cfg.get("has_labels", True)),
                )

                if cfg.get("has_labels", True):
                    self.assertIn("f1_anomaly", metrics)
                    self.assertIn("accuracy", metrics)
                    self.assertIn("confusion_matrix", metrics)
                else:
                    self.assertNotIn("f1_anomaly", metrics)
                    self.assertNotIn("accuracy", metrics)
                    self.assertNotIn("confusion_matrix", metrics)
                    self.assertEqual(cfg.get("evaluation_mode"), "unlabeled")

    def test_openssh_uses_unlabeled_prediction_policy(self):
        cfg = get_dataset("openssh")

        self.assertFalse(cfg["has_labels"])
        self.assertEqual(cfg["evaluation_mode"], "unlabeled")
        self.assertEqual(cfg["post_filter"], "predicted_only")
        self.assertGreater(float(cfg["contamination"]), 0)
        self.assertLess(float(cfg["contamination"]), 1)


if __name__ == "__main__":
    unittest.main()
