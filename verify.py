import json
from pathlib import Path

import pandas as pd


SCORES = Path("output/scores.parquet")
PARSED = Path("output/parsed.parquet")
METRICS = Path("output/metrics.json")


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run pipeline.py first.")


def main() -> None:
    require(SCORES)
    require(PARSED)
    require(METRICS)

    scores = pd.read_parquet(SCORES, engine="pyarrow")
    parsed = pd.read_parquet(PARSED, engine="pyarrow")
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    cm = metrics["confusion_matrix"]

    errors = []

    total_from_cm = (
        cm["true_negative"]
        + cm["false_positive"]
        + cm["false_negative"]
        + cm["true_positive"]
    )
    if total_from_cm != len(scores):
        errors.append(
            f"confusion matrix total {total_from_cm} != "
            f"scores rows {len(scores)}"
        )

    if metrics["total_windows"] != len(scores):
        errors.append(
            f"metrics total_windows {metrics['total_windows']} != "
            f"scores rows {len(scores)}"
        )

    gt_anomalous = int(scores["is_anomaly"].sum())
    if metrics["anomalous_windows"] != gt_anomalous:
        errors.append(
            f"metrics anomalous_windows {metrics['anomalous_windows']} != "
            f"scores is_anomaly sum {gt_anomalous}"
        )

    predicted = set(scores["predicted"].dropna().unique())
    if not predicted.issubset({0, 1}):
        errors.append(f"predicted has unexpected values: {sorted(predicted)}")

    if scores["anomaly_score"].isna().any():
        errors.append("anomaly_score contains NaN values")

    if (scores["anomaly_score"] < 0).any():
        errors.append("anomaly_score contains negative values")

    missing_score_windows = scores[
        scores["window_start"].isna() | scores["window_end"].isna()
    ]
    if len(missing_score_windows) > 0:
        errors.append(f"{len(missing_score_windows)} score rows lack windows")

    if errors:
        print("Verification failed:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("All checks passed.")
    print(f"  Parsed log rows : {len(parsed):,}")
    print(f"  Total windows   : {len(scores):,}")
    print(f"  Anomalous truth : {gt_anomalous:,}")
    print(f"  Predicted alerts: {int(scores['predicted'].sum()):,}")
    print(f"  F1 anomaly      : {metrics['f1_anomaly']:.4f}")
    print(f"  Accuracy        : {metrics['accuracy']:.2%}")

    top5 = scores.nlargest(5, "anomaly_score")
    print("\nTop 5 most anomalous windows:")
    for _, row in top5.iterrows():
        print(
            f"\nWindow {int(row['window_start'])} | "
            f"Score: {row['anomaly_score']:.3f} | "
            f"Anomalous lines: {int(row['anomaly_count'])}"
        )

        logs = parsed[
            (parsed["timestamp"] >= row["window_start"])
            & (parsed["timestamp"] < row["window_end"])
            & (parsed["is_anomaly"] == 1)
        ].head(5)

        for _, log in logs.iterrows():
            print(f"  [{log['level']}] {log['node']}: {log['template']}")


if __name__ == "__main__":
    main()
