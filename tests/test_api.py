import unittest

from fastapi.testclient import TestClient

from api.main import app
from dataset_config import DATASETS, dataset_paths


def artifacts_ready(dataset: str) -> bool:
    paths = dataset_paths(dataset)
    required = [
        "parsed",
        "features",
        "scores",
        "metrics",
        "alerts",
        "alerts_minilm",
        "clustering_comparison",
    ]
    return all(paths[key].exists() for key in required)


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_and_dataset_listing(self):
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200)
        self.assertEqual(root.json()["status"], "ok")

        datasets = self.client.get("/datasets")
        self.assertEqual(datasets.status_code, 200)
        names = {row["name"] for row in datasets.json()["datasets"]}
        self.assertEqual(names, set(DATASETS))

    def test_unknown_dataset_returns_404(self):
        response = self.client.get("/datasets/does_not_exist/stats")
        self.assertEqual(response.status_code, 404)

    def test_missing_artifact_endpoint_uses_http_error(self):
        response = self.client.get("/datasets/does_not_exist/metrics")
        self.assertEqual(response.status_code, 404)

    def test_ready_dataset_endpoints(self):
        for dataset in DATASETS:
            with self.subTest(dataset=dataset):
                if not artifacts_ready(dataset):
                    self.skipTest(f"{dataset} artifacts are not ready")

                for endpoint in [
                    "/stats",
                    "/levels",
                    "/scores/timeline",
                    "/alerts/summary",
                    "/alerts/minilm/clusters?method=minilm",
                    "/clustering/comparison",
                    "/source-files",
                ]:
                    response = self.client.get(f"/datasets/{dataset}{endpoint}")
                    self.assertEqual(response.status_code, 200, endpoint)

    def test_log_search_and_cluster_filters_accept_user_input(self):
        if not artifacts_ready("bgl"):
            self.skipTest("bgl artifacts are not ready")

        logs = self.client.get(
            "/datasets/bgl/logs",
            params={"limit": 5, "search": "machine' OR 1=1 --"},
        )
        self.assertEqual(logs.status_code, 200)
        self.assertIn("data", logs.json())

        clusters = self.client.get(
            "/datasets/bgl/alerts/minilm/clusters",
            params={"method": "minilm", "severity": "LOW,MEDIUM"},
        )
        self.assertEqual(clusters.status_code, 200)
        self.assertIn("data", clusters.json())


if __name__ == "__main__":
    unittest.main()
