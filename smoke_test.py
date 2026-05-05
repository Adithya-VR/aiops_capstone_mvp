import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from dataset_config import DATASETS, dataset_paths


REQUIRED_ARTIFACTS = [
    "parsed",
    "features",
    "scores",
    "metrics",
    "alerts",
    "alerts_minilm",
    "clustering_comparison",
]

REQUIRED_ENDPOINTS = [
    "/stats",
    "/levels",
    "/levels/distribution",
    "/levels/predicted-window-distribution",
    "/templates/top?limit=5",
    "/scores/timeline",
    "/scores/histogram?bins=20",
    "/alerts?limit=5",
    "/alerts/summary",
    "/clusters",
    "/clustering/comparison",
    "/alerts/minilm/clusters?method=minilm",
]


def check(condition: bool, message: str, errors: list[str]) -> None:
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        errors.append(message)


def row_count(path: Path) -> int:
    return len(pd.read_parquet(path, engine="pyarrow"))


def validate_dataset(dataset: str, client: TestClient, errors: list[str]) -> None:
    print(f"\n=== {dataset.upper()} ===")
    paths = dataset_paths(dataset)

    for key in REQUIRED_ARTIFACTS:
        check(paths[key].exists(), f"artifact exists: {paths[key]}", errors)

    if paths["metrics"].exists():
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        check(metrics.get("dataset") == dataset, "metrics dataset matches", errors)

    if paths["parsed"].exists() and paths["scores"].exists():
        parsed_rows = row_count(paths["parsed"])
        scores_rows = row_count(paths["scores"])
        check(parsed_rows > 0, f"parsed rows > 0 ({parsed_rows:,})", errors)
        check(scores_rows > 0, f"score rows > 0 ({scores_rows:,})", errors)

    for endpoint in REQUIRED_ENDPOINTS:
        url = f"/datasets/{dataset}{endpoint}"
        response = client.get(url)
        check(response.status_code == 200, f"GET {url} -> {response.status_code}", errors)

    stats = client.get(f"/datasets/{dataset}/stats").json()
    check(stats["total_logs"] > 0, "stats total_logs > 0", errors)
    check(stats["total_windows"] > 0, "stats total_windows > 0", errors)
    check(
        bool(stats["has_labels"]) == bool(DATASETS[dataset]["has_labels"]),
        "stats has_labels matches config",
        errors,
    )

    alert_summary = client.get(f"/datasets/{dataset}/alerts/summary").json()
    check(alert_summary["total_alerts"] > 0, "alerts generated", errors)

    minilm_clusters = client.get(
        f"/datasets/{dataset}/alerts/minilm/clusters",
        params={"method": "minilm"},
    ).json()
    check(minilm_clusters["total_clusters"] > 0, "MiniLM clusters generated", errors)


def main() -> None:
    client = TestClient(app)
    errors: list[str] = []

    for dataset in ["bgl", "openssh"]:
        validate_dataset(dataset, client, errors)

    missing = client.get("/datasets/does_not_exist/stats")
    check(missing.status_code == 404, "unknown dataset returns HTTP 404", errors)

    print("\n=== SUMMARY ===")
    if errors:
        print(f"FAILED: {len(errors)} issue(s)")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
