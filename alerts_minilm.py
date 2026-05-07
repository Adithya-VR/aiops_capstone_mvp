import argparse
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from alert_generation import generate_alerts
from dataset_config import DEFAULT_DATASET, dataset_paths, get_dataset

TFIDF_EPS = 0.5
MINILM_EPS = 0.4
MIN_SAMPLES = 2
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def assign_unique_noise_ids(labels):
    labels = labels.copy()
    noise_positions = np.where(labels == -1)[0]
    for noise_number, row_index in enumerate(noise_positions, start=1):
        labels[row_index] = -noise_number
    return labels


def silhouette_without_noise(matrix, labels, metric: str) -> float:
    mask = labels != -1
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        return float(silhouette_score(matrix[mask], labels[mask], metric=metric))
    return 0.0


def run_dbscan(matrix, eps: float, metric: str) -> tuple[np.ndarray, int, int, float]:
    labels = DBSCAN(
        eps=eps,
        min_samples=MIN_SAMPLES,
        metric=metric,
    ).fit_predict(matrix)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    sil = silhouette_without_noise(matrix, labels, metric)
    return labels, n_clusters, n_noise, sil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    cfg = get_dataset(args.dataset)
    paths = dataset_paths(args.dataset)
    has_labels = bool(cfg.get("has_labels", True))

    print("Loading data...")
    scores = pd.read_parquet(paths["scores"], engine="pyarrow")
    parsed = pd.read_parquet(paths["parsed"], engine="pyarrow")
    alert_df = generate_alerts(scores, parsed, has_labels)
    print(f"Alerts generated: {len(alert_df):,}")

    if alert_df.empty:
        alert_df["cluster_id_tfidf"] = []
        alert_df["cluster_id_minilm"] = []
        alert_df.to_parquet(paths["alerts_minilm"], engine="pyarrow", index=False)
        paths["clustering_comparison"].write_text(
            json.dumps(
                {
                    "tfidf_eps": TFIDF_EPS,
                    "minilm_eps": MINILM_EPS,
                    "min_samples": MIN_SAMPLES,
                    "tfidf_silhouette": 0.0,
                    "minilm_silhouette": 0.0,
                    "tfidf_clusters": 0,
                    "tfidf_unique": 0,
                    "minilm_clusters": 0,
                    "minilm_unique": 0,
                    "methodology": "DBSCAN with cosine distance; silhouette excludes noise points.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    templates = alert_df["top_template"].fillna("unknown").tolist()

    print("\n-- Method 1: TF-IDF + DBSCAN --")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-zA-Z]+",
        max_features=500,
    )
    x_tfidf = normalize(vectorizer.fit_transform(templates).toarray())
    tfidf_labels, n_tfidf_clusters, n_tfidf_noise, tfidf_sil = run_dbscan(
        x_tfidf,
        eps=TFIDF_EPS,
        metric="cosine",
    )
    print(f"  eps       : {TFIDF_EPS}")
    print(f"  Clusters  : {n_tfidf_clusters}")
    print(f"  Unique    : {n_tfidf_noise}")
    print(f"  Silhouette: {tfidf_sil:.4f} (noise points excluded)")

    print("\n-- Method 2: MiniLM + DBSCAN --")
    print(f"  Loading {MODEL_NAME}...")
    encoder = SentenceTransformer(MODEL_NAME)
    embeddings = encoder.encode(
        templates,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    minilm_labels, n_minilm_clusters, n_minilm_noise, minilm_sil = run_dbscan(
        embeddings,
        eps=MINILM_EPS,
        metric="cosine",
    )
    print(f"  eps       : {MINILM_EPS}")
    print(f"  Clusters  : {n_minilm_clusters}")
    print(f"  Unique    : {n_minilm_noise}")
    print(f"  Silhouette: {minilm_sil:.4f} (noise points excluded)")

    tfidf_groups = n_tfidf_clusters + n_tfidf_noise
    minilm_groups = n_minilm_clusters + n_minilm_noise

    print("\n-- Comparison Summary --")
    print(f"{'Metric':<25} {'TF-IDF':>10} {'MiniLM':>10}")
    print("-" * 48)
    print(f"{'Clusters found':<25} {n_tfidf_clusters:>10} {n_minilm_clusters:>10}")
    print(f"{'Unique alerts':<25} {n_tfidf_noise:>10} {n_minilm_noise:>10}")
    print(f"{'Total groups':<25} {tfidf_groups:>10} {minilm_groups:>10}")
    print(
        f"{'Noise reduction':<25} "
        f"{(1 - tfidf_groups / len(alert_df)):>10.1%} "
        f"{(1 - minilm_groups / len(alert_df)):>10.1%}"
    )
    print(f"{'Silhouette score':<25} {tfidf_sil:>10.4f} {minilm_sil:>10.4f}")

    alert_df["cluster_id_tfidf"] = assign_unique_noise_ids(tfidf_labels)
    alert_df["cluster_id_minilm"] = assign_unique_noise_ids(minilm_labels)
    alert_df.to_parquet(paths["alerts_minilm"], engine="pyarrow", index=False)
    print(f"\nComparison results saved -> {paths['alerts_minilm']}")

    scores_out = {
        "tfidf_eps": TFIDF_EPS,
        "minilm_eps": MINILM_EPS,
        "min_samples": MIN_SAMPLES,
        "tfidf_silhouette": round(float(tfidf_sil), 4),
        "minilm_silhouette": round(float(minilm_sil), 4),
        "tfidf_clusters": n_tfidf_clusters,
        "tfidf_unique": n_tfidf_noise,
        "minilm_clusters": n_minilm_clusters,
        "minilm_unique": n_minilm_noise,
        "methodology": (
            "DBSCAN with cosine distance. TF-IDF and MiniLM use separate eps "
            "values because their vector spaces have different distance scales; "
            "silhouette is computed after excluding DBSCAN noise points."
        ),
    }
    paths["clustering_comparison"].write_text(
        json.dumps(scores_out, indent=2),
        encoding="utf-8",
    )
    print(f"Scores saved -> {paths['clustering_comparison']}")


if __name__ == "__main__":
    main()
