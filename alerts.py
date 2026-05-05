import pandas as pd
import numpy as np
import duckdb
import argparse
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from dataset_config import DEFAULT_DATASET, dataset_paths, get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=DEFAULT_DATASET)
args = parser.parse_args()
CFG = get_dataset(args.dataset)
PATHS = dataset_paths(args.dataset)
HAS_LABELS = bool(CFG.get("has_labels", True))

SCORES  = PATHS["scores"]
PARSED  = PATHS["parsed"]
ALERTS  = PATHS["alerts"]

print("Generating alerts from anomalous windows...")

scores = pd.read_parquet(SCORES, engine="pyarrow")
parsed = pd.read_parquet(PARSED, engine="pyarrow")

# ── Step 1: Generate alerts from anomalous windows ────────────────
# An alert = one anomalous window with its most common fatal template
# p95 = scores["anomaly_score"].quantile(0.95)
# p85 = scores["anomaly_score"].quantile(0.85)
# p70 = scores["anomaly_score"].quantile(0.70)

# Compute percentiles from anomalous windows only
# so severity levels distribute meaningfully across alerts
p95 = scores[scores["predicted"]==1]["anomaly_score"].quantile(0.95)
p85 = scores[scores["predicted"]==1]["anomaly_score"].quantile(0.85)
p70 = scores[scores["predicted"]==1]["anomaly_score"].quantile(0.70)

anomalous = scores[scores["predicted"] == 1].copy()
print(f"  Anomalous windows: {len(anomalous):,}")

alerts = []
for _, row in anomalous.iterrows():
    # Get log lines in this window
    window_logs = parsed[
        (parsed["timestamp"] >= row["window_start"]) &
        (parsed["timestamp"] <  row["window_end"])
    ]

    # Labeled datasets use the true anomalous lines as representatives.
    # Unlabeled datasets use all logs in the predicted alert window.
    if HAS_LABELS:
        representative_logs = window_logs[window_logs["is_anomaly"] == 1]
        if representative_logs.empty:
            representative_logs = window_logs
    else:
        representative_logs = window_logs

    top_template = (representative_logs["template"]
                    .value_counts().index[0]
                    if len(representative_logs) > 0 else "unknown")

    top_level = (representative_logs["level"]
                 .value_counts().index[0]
                 if len(representative_logs) > 0 else "INFO")

    # Assign severity based on score percentile
    # if row["anomaly_score"] >= p95:
    #     severity = "CRITICAL"
    # elif row["anomaly_score"] >= p85:
    #     severity = "HIGH"
    # else:
    #     severity = "MEDIUM"

    if row["anomaly_score"] >= p95:
        severity = "CRITICAL"      # top 5%  — most dangerous
    elif row["anomaly_score"] >= p85:
        severity = "HIGH"          # next 10% — serious
    elif row["anomaly_score"] >= p70:
        severity = "MEDIUM"        # next 15% — moderate
    else:
        severity = "LOW"           # bottom 70% of anomalies — minor

    alerts.append({
        "window_start":   row["window_start"],
        "window_end":     row["window_end"],
        "anomaly_score":  row["anomaly_score"],
        "anomaly_count":  row["anomaly_count"],
        "total_logs":     row["total_logs"],
        "severity":       severity,
        "top_template":   top_template,
        "top_level":      top_level,
        "cluster_id":     -1,   # filled in next step
    })

alert_df = pd.DataFrame(alerts)
print(f"  Alerts generated: {len(alert_df):,}")
print(f"  CRITICAL: {(alert_df['severity']=='CRITICAL').sum()}")
print(f"  HIGH    : {(alert_df['severity']=='HIGH').sum()}")
print(f"  MEDIUM  : {(alert_df['severity']=='MEDIUM').sum()}")
print(f"  LOW  : {(alert_df['severity']=='LOW').sum()}")
# ── Step 2: Cluster alerts by template similarity ─────────────────
# Convert templates to TF-IDF style vectors then cluster with DBSCAN
# No heavy models needed — just token overlap similarity
print("\nClustering alerts by template similarity...")

from sklearn.feature_extraction.text import TfidfVectorizer

templates = alert_df["top_template"].fillna("unknown").tolist()

# TF-IDF vectorize the template strings
vectorizer = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"[a-zA-Z]+",  # words only, ignore <*> tokens
    max_features=500
)
X = vectorizer.fit_transform(templates).toarray()
X = normalize(X)   # L2 normalize for cosine similarity

# DBSCAN — no need to specify number of clusters
# eps=0.4 means templates need 60%+ similarity to cluster together
clusterer = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
labels = clusterer.fit_predict(X)
unique_labels = labels.copy()
noise_positions = np.where(unique_labels == -1)[0]
for noise_number, row_index in enumerate(noise_positions, start=1):
    unique_labels[row_index] = -noise_number

alert_df["cluster_id"] = unique_labels

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = int((labels == -1).sum())

print(f"  Total alerts  : {len(alert_df):,}")
print(f"  Clusters found: {n_clusters}")
print(f"  Unclustered   : {n_noise} (unique alert types)")
print(f"  Reduction     : {len(alert_df):,} alerts -> "
      f"{n_clusters + n_noise} distinct groups")

# ── Step 3: Add cluster labels ─────────────────────────────────────
# For each cluster find the most representative template
cluster_labels = {}
for cid in set(unique_labels):
    if cid < 0:
        continue
    cluster_members = alert_df[alert_df["cluster_id"] == cid]
    # Most common template in this cluster = cluster name
    cluster_labels[cid] = (cluster_members["top_template"]
                           .value_counts().index[0][:60])

alert_df["cluster_label"] = alert_df.apply(
    lambda row: (
        f"Unique: {str(row['top_template'])[:40]}"
        if row["cluster_id"] < 0
        else cluster_labels.get(row["cluster_id"], "Unknown")
    ),
    axis=1,
)

alert_df.to_parquet(ALERTS, engine="pyarrow", index=False)
print(f"\nAlerts saved -> {ALERTS}")

# ── Step 4: Print cluster summary ─────────────────────────────────
print("\n-- Top Alert Clusters --")
summary = (alert_df
           .groupby(["cluster_id", "cluster_label"])
           .agg(
               count         = ("anomaly_score", "count"),
               avg_score     = ("anomaly_score", "mean"),
               max_score     = ("anomaly_score", "max"),
               critical_count= ("severity",
                                lambda x: (x=="CRITICAL").sum())
           )
           .sort_values("count", ascending=False)
           .head(10))
print(summary.to_string())
print("\nDone. Now run: streamlit run app.py")
