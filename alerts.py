import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from alert_generation import generate_alerts
from dataset_config import DEFAULT_DATASET, dataset_paths, get_dataset

TFIDF_EPS = 0.5
MIN_SAMPLES = 2


def assign_unique_noise_ids(labels):
    labels = labels.copy()
    noise_positions = np.where(labels == -1)[0]
    for noise_number, row_index in enumerate(noise_positions, start=1):
        labels[row_index] = -noise_number
    return labels


def cluster_labels(alert_df: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    templates = alert_df["top_template"].fillna("unknown").tolist()
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-zA-Z]+",
        max_features=500,
    )
    x_tfidf = normalize(vectorizer.fit_transform(templates).toarray())
    labels = DBSCAN(
        eps=TFIDF_EPS,
        min_samples=MIN_SAMPLES,
        metric="cosine",
    ).fit_predict(x_tfidf)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    return labels, n_clusters, n_noise


def add_cluster_labels(alert_df: pd.DataFrame) -> pd.DataFrame:
    cluster_labels_by_id = {}
    for cid in set(alert_df["cluster_id"]):
        if cid < 0:
            continue
        members = alert_df[alert_df["cluster_id"] == cid]
        cluster_labels_by_id[cid] = members["top_template"].value_counts().index[0][:60]

    alert_df["cluster_label"] = alert_df.apply(
        lambda row: (
            f"Unique: {str(row['top_template'])[:40]}"
            if row["cluster_id"] < 0
            else cluster_labels_by_id.get(row["cluster_id"], "Unknown")
        ),
        axis=1,
    )
    return alert_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    cfg = get_dataset(args.dataset)
    paths = dataset_paths(args.dataset)
    has_labels = bool(cfg.get("has_labels", True))

    print("Generating alerts from anomalous windows...")
    scores = pd.read_parquet(paths["scores"], engine="pyarrow")
    parsed = pd.read_parquet(paths["parsed"], engine="pyarrow")
    alert_df = generate_alerts(scores, parsed, has_labels)
    alert_df["cluster_id"] = -1

    print(f"  Alerts generated: {len(alert_df):,}")
    if alert_df.empty:
        alert_df["cluster_label"] = []
        alert_df.to_parquet(paths["alerts"], engine="pyarrow", index=False)
        print(f"\nAlerts saved -> {paths['alerts']}")
        return

    print(f"  CRITICAL: {(alert_df['severity'] == 'CRITICAL').sum()}")
    print(f"  HIGH    : {(alert_df['severity'] == 'HIGH').sum()}")
    print(f"  MEDIUM  : {(alert_df['severity'] == 'MEDIUM').sum()}")
    print(f"  LOW     : {(alert_df['severity'] == 'LOW').sum()}")

    print("\nClustering alerts by template similarity...")
    raw_labels, n_clusters, n_noise = cluster_labels(alert_df)
    alert_df["cluster_id"] = assign_unique_noise_ids(raw_labels)
    alert_df = add_cluster_labels(alert_df)

    print(f"  Total alerts  : {len(alert_df):,}")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Unclustered   : {n_noise} (unique alert types)")
    print(
        f"  Reduction     : {len(alert_df):,} alerts -> "
        f"{n_clusters + n_noise} distinct groups"
    )

    alert_df.to_parquet(paths["alerts"], engine="pyarrow", index=False)
    print(f"\nAlerts saved -> {paths['alerts']}")

    print("\n-- Top Alert Clusters --")
    summary = (
        alert_df.groupby(["cluster_id", "cluster_label"])
        .agg(
            count=("anomaly_score", "count"),
            avg_score=("anomaly_score", "mean"),
            max_score=("anomaly_score", "max"),
            critical_count=("severity", lambda x: (x == "CRITICAL").sum()),
        )
        .sort_values("count", ascending=False)
        .head(10)
    )
    print(summary.to_string())
    print("\nDone. Now run: streamlit run app.py")


if __name__ == "__main__":
    main()
