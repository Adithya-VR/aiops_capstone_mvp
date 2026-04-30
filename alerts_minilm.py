# alerts_minilm.py
# Runs MiniLM-based alert clustering alongside existing TF-IDF results
# For comparison purposes — does NOT replace alerts.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SCORES  = Path("output/scores.parquet")
PARSED  = Path("output/parsed.parquet")
OUT     = Path("output/alerts_minilm.parquet")

print("Loading data...")
scores = pd.read_parquet(SCORES, engine="pyarrow")
parsed = pd.read_parquet(PARSED, engine="pyarrow")

# ── Generate alerts (same logic as alerts.py) ──────────────────────
p95 = scores[scores["predicted"]==1]["anomaly_score"].quantile(0.95)
p85 = scores[scores["predicted"]==1]["anomaly_score"].quantile(0.85)
p70 = scores[scores["predicted"]==1]["anomaly_score"].quantile(0.70)

anomalous = scores[scores["predicted"] == 1].copy()
print(f"Anomalous windows: {len(anomalous):,}")

alerts = []
for _, row in anomalous.iterrows():
    window_logs = parsed[
        (parsed["timestamp"] >= row["window_start"]) &
        (parsed["timestamp"] <  row["window_end"])
    ]
    anom_logs = window_logs[window_logs["is_anomaly"] == 1]
    if anom_logs.empty:
        anom_logs = window_logs
    if anom_logs.empty:
        continue

    top_template = (anom_logs["template"]
                    .value_counts().index[0])

    if row["anomaly_score"] >= p95:
        severity = "CRITICAL"
    elif row["anomaly_score"] >= p85:
        severity = "HIGH"
    elif row["anomaly_score"] >= p70:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    alerts.append({
        "window_start":  row["window_start"],
        "window_end":    row["window_end"],
        "anomaly_score": row["anomaly_score"],
        "anomaly_count": row["anomaly_count"],
        "total_logs":    row["total_logs"],
        "severity":      severity,
        "top_template":  top_template,
    })

alert_df = pd.DataFrame(alerts)
print(f"Alerts generated: {len(alert_df):,}")
templates = alert_df["top_template"].fillna("unknown").tolist()

# ══════════════════════════════════════════════════════════════════
# METHOD 1: TF-IDF + DBSCAN (existing approach)
# ══════════════════════════════════════════════════════════════════
print("\n── Method 1: TF-IDF + DBSCAN ────────────────────")

vectorizer = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"[a-zA-Z]+",
    max_features=500
)
X_tfidf  = vectorizer.fit_transform(templates).toarray()
X_tfidf  = normalize(X_tfidf)

tfidf_labels = DBSCAN(
    eps=0.5, min_samples=2, metric="cosine"
).fit_predict(X_tfidf)

n_tfidf_clusters = len(set(tfidf_labels)) - \
                   (1 if -1 in tfidf_labels else 0)
n_tfidf_noise    = int((tfidf_labels == -1).sum())

print(f"  Clusters  : {n_tfidf_clusters}")
print(f"  Unique    : {n_tfidf_noise}")
print(f"  Groups    : {n_tfidf_clusters + n_tfidf_noise}")
print(f"  Reduction : {len(alert_df):,} → "
      f"{n_tfidf_clusters + n_tfidf_noise} groups "
      f"({(1-(n_tfidf_clusters+n_tfidf_noise)/len(alert_df)):.1%})")

# Silhouette score for TF-IDF
# if len(set(tfidf_labels)) > 1:
#     tfidf_sil = silhouette_score(
#         X_tfidf, tfidf_labels, metric="cosine"
#     )
#     print(f"  Silhouette: {tfidf_sil:.4f}")
# else:
#     tfidf_sil = 0
#     print(f"  Silhouette: N/A (only 1 cluster)")

# Silhouette score for TF-IDF — exclude noise points (label = -1)
tfidf_mask = tfidf_labels != -1
if tfidf_mask.sum() > 1 and len(set(tfidf_labels[tfidf_mask])) > 1:
    tfidf_sil = silhouette_score(
        X_tfidf[tfidf_mask], tfidf_labels[tfidf_mask],
        metric="cosine"
    )
    print(f"  Silhouette: {tfidf_sil:.4f} (noise points excluded)")
else:
    tfidf_sil = 0.0
    print(f"  Silhouette: N/A")

# ══════════════════════════════════════════════════════════════════
# METHOD 2: MiniLM + DBSCAN (new approach)
# ══════════════════════════════════════════════════════════════════
print("\n── Method 2: MiniLM + DBSCAN ────────────────────")
print("  Loading sentence-transformers/all-MiniLM-L6-v2...")
print("  (Downloads ~80MB on first run, cached after)")

from sentence_transformers import SentenceTransformer

encoder    = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)
embeddings = encoder.encode(
    templates,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

minilm_labels = DBSCAN(
    eps=0.4, min_samples=2, metric="cosine"
).fit_predict(embeddings)

n_minilm_clusters = len(set(minilm_labels)) - \
                    (1 if -1 in minilm_labels else 0)
n_minilm_noise    = int((minilm_labels == -1).sum())

print(f"\n  Clusters  : {n_minilm_clusters}")
print(f"  Unique    : {n_minilm_noise}")
print(f"  Groups    : {n_minilm_clusters + n_minilm_noise}")
print(f"  Reduction : {len(alert_df):,} → "
      f"{n_minilm_clusters + n_minilm_noise} groups "
      f"({(1-(n_minilm_clusters+n_minilm_noise)/len(alert_df)):.1%})")

# Silhouette score for MiniLM
# if len(set(minilm_labels)) > 1:
#     minilm_sil = silhouette_score(
#         embeddings, minilm_labels, metric="cosine"
#     )
#     print(f"  Silhouette: {minilm_sil:.4f}")
# else:
#     minilm_sil = 0
#     print(f"  Silhouette: N/A (only 1 cluster)")

# Silhouette score for MiniLM — exclude noise points (label = -1)
minilm_mask = minilm_labels != -1
if minilm_mask.sum() > 1 and len(set(minilm_labels[minilm_mask])) > 1:
    minilm_sil = silhouette_score(
        embeddings[minilm_mask], minilm_labels[minilm_mask],
        metric="cosine"
    )
    print(f"  Silhouette: {minilm_sil:.4f} (noise points excluded)")
else:
    minilm_sil = 0.0
    print(f"  Silhouette: N/A")

# ══════════════════════════════════════════════════════════════════
# COMPARISON
# ══════════════════════════════════════════════════════════════════
print("\n── Comparison Summary ───────────────────────────")
print(f"{'Metric':<25} {'TF-IDF':>10} {'MiniLM':>10} {'Winner':>10}")
print("-" * 55)

tfidf_groups  = n_tfidf_clusters  + n_tfidf_noise
minilm_groups = n_minilm_clusters + n_minilm_noise

print(f"{'Clusters found':<25} {n_tfidf_clusters:>10} "
      f"{n_minilm_clusters:>10}")
print(f"{'Unique alerts':<25} {n_tfidf_noise:>10} "
      f"{n_minilm_noise:>10}")
print(f"{'Total groups':<25} {tfidf_groups:>10} "
      f"{minilm_groups:>10}")
print(f"{'Noise reduction':<25} "
      f"{(1-tfidf_groups/len(alert_df)):.1%}".rjust(10+25) +
      f" {(1-minilm_groups/len(alert_df)):.1%}".rjust(10))
print(f"{'Silhouette score':<25} {tfidf_sil:>10.4f} "
      f"{minilm_sil:>10.4f} "
      f"{'MiniLM' if minilm_sil > tfidf_sil else 'TF-IDF':>10}")

# ── Save MiniLM results ────────────────────────────────────────────
alert_df["cluster_id_tfidf"]   = tfidf_labels
alert_df["cluster_id_minilm"]  = minilm_labels
alert_df.to_parquet(OUT, engine="pyarrow", index=False)
print(f"\nComparison results saved → {OUT}")
print("\nDecision: if MiniLM silhouette > TF-IDF silhouette,")
print("replace alerts.py clustering with MiniLM.")
print("Otherwise keep TF-IDF.")

# Save silhouette scores to file for dashboard
import json
from pathlib import Path

scores_out = {
    "tfidf_silhouette":  round(float(tfidf_sil), 4),
    "minilm_silhouette": round(float(minilm_sil), 4),
    "tfidf_clusters":    n_tfidf_clusters,
    "tfidf_unique":      n_tfidf_noise,
    "minilm_clusters":   n_minilm_clusters,
    "minilm_unique":     n_minilm_noise,
}
Path("output/clustering_comparison.json").write_text(
    json.dumps(scores_out, indent=2)
)
print(f"\nScores saved → output/clustering_comparison.json")