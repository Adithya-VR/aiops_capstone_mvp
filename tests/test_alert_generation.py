import unittest

import pandas as pd

from alert_generation import generate_alerts, severity_for_score


class AlertGenerationTests(unittest.TestCase):
    def test_severity_thresholds(self):
        self.assertEqual(severity_for_score(10.0, 7.0, 8.5, 9.5), "CRITICAL")
        self.assertEqual(severity_for_score(9.0, 7.0, 8.5, 9.5), "HIGH")
        self.assertEqual(severity_for_score(7.5, 7.0, 8.5, 9.5), "MEDIUM")
        self.assertEqual(severity_for_score(6.0, 7.0, 8.5, 9.5), "LOW")

    def test_generate_alerts_includes_top_level(self):
        scores = pd.DataFrame(
            {
                "window_start": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                "window_end": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                "anomaly_score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "predicted": [1] * 10,
                "anomaly_count": [1] * 10,
                "total_logs": [2] * 10,
            }
        )
        parsed = pd.DataFrame(
            {
                "timestamp": [1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
                "is_anomaly": [1] * 10,
                "template": [f"template-{i}" for i in range(10)],
                "level": ["INFO", "WARNING", "ERROR", "FATAL", "INFO"] * 2,
            }
        )

        alerts = generate_alerts(scores, parsed, has_labels=True)

        self.assertEqual(len(alerts), 10)
        self.assertIn("top_level", alerts.columns)
        self.assertIn("top_template", alerts.columns)
        self.assertIn("CRITICAL", set(alerts["severity"]))
        self.assertIn("HIGH", set(alerts["severity"]))
        self.assertIn("MEDIUM", set(alerts["severity"]))
        self.assertIn("LOW", set(alerts["severity"]))

    def test_labeled_dataset_prefers_anomalous_representative_logs(self):
        scores = pd.DataFrame(
            {
                "window_start": [0],
                "window_end": [10],
                "anomaly_score": [1.0],
                "predicted": [1],
                "anomaly_count": [1],
                "total_logs": [2],
            }
        )
        parsed = pd.DataFrame(
            {
                "timestamp": [1, 2],
                "is_anomaly": [0, 1],
                "template": ["normal-template", "anomaly-template"],
                "level": ["INFO", "FATAL"],
            }
        )

        alerts = generate_alerts(scores, parsed, has_labels=True)

        self.assertEqual(alerts.iloc[0]["top_template"], "anomaly-template")
        self.assertEqual(alerts.iloc[0]["top_level"], "FATAL")


if __name__ == "__main__":
    unittest.main()
