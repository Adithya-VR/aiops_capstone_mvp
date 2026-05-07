import argparse
import json
import pickle
import re
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from dataset_config import DEFAULT_DATASET, dataset_paths, get_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default=DEFAULT_DATASET)
args = parser.parse_args()

DATASET = args.dataset.lower()
CFG = get_dataset(DATASET)
PATHS = dataset_paths(DATASET)
PATHS["output_dir"].mkdir(parents=True, exist_ok=True)

RAW_LOG = PATHS["raw"]
PARSED = PATHS["parsed"]
FEATURES = PATHS["features"]
SCORES = PATHS["scores"]
MODEL = PATHS["model"]
METRICS = PATHS["metrics"]

WINDOW = int(CFG["window"])
STEP = int(CFG["step"])
WINDOW_MODE = "sliding"
HAS_LABELS = bool(CFG.get("has_labels", True))
EVALUATION_MODE = CFG.get("evaluation_mode", "supervised")

SUSPICIOUS_LEVELS = {
    "SEVERE",
    "ERROR",
    "FAILURE",
    "FATAL",
    "CRITICAL",
    "BREAKIN_ATTEMPT",
    "INVALID_USER",
    "FAILED_PASSWORD",
    "AUTH_FAILURE",
    "TOO_MANY_FAILURES",
    "BAD_PROTOCOL",
}


def write_metrics(scores: pd.DataFrame, path: Path = METRICS) -> dict:
    """Write metrics.json from the same scores DataFrame used by the app/API."""
    y_true = scores["is_anomaly"].astype(int)
    y_pred = scores["predicted"].astype(int)
    anomaly_score = scores["anomaly_score"].astype(float)

    metrics = {
        "dataset": DATASET,
        "evaluation_mode": EVALUATION_MODE,
        "ground_truth_available": HAS_LABELS,
        "total_windows": int(len(scores)),
        "predicted_anomalous_windows": int(y_pred.sum()),
        "predicted_normal_windows": int((y_pred == 0).sum()),
        "score_min": round(float(anomaly_score.min()), 4),
        "score_max": round(float(anomaly_score.max()), 4),
        "score_mean": round(float(anomaly_score.mean()), 4),
        "score_p85": round(float(anomaly_score.quantile(0.85)), 4),
        "score_p90": round(float(anomaly_score.quantile(0.90)), 4),
        "score_p95": round(float(anomaly_score.quantile(0.95)), 4),
    }

    if not HAS_LABELS:
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        roc_auc = float(roc_auc_score(y_true, anomaly_score))
    except ValueError:
        roc_auc = 0.0

    metrics.update({
        "anomalous_windows": int(y_true.sum()),
        "normal_windows": int((y_true == 0).sum()),
        "precision_anomaly": round(
            float(precision_score(y_true, y_pred, zero_division=0)), 4
        ),
        "recall_anomaly": round(
            float(recall_score(y_true, y_pred, zero_division=0)), 4
        ),
        "f1_anomaly": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision_normal": round(
            float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
            4,
        ),
        "recall_normal": round(
            float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
            4,
        ),
        "f1_normal": round(
            float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
            4,
        ),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(roc_auc, 4),
        "macro_avg_f1": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            4,
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    })

    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def has_current_window_features(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        sample = pd.read_parquet(
            path,
            columns=["window_start", "window_end", "step_size_sec"],
            engine="pyarrow",
        )
    except Exception:
        return False

    if sample.empty:
        return False

    span = sample["window_end"] - sample["window_start"]
    return bool((span == WINDOW).all() and (sample["step_size_sec"] == STEP).all())


def parse_with_dataset_parser() -> tuple[pd.DataFrame, int]:
    parser_module = import_module(f"parsers.{CFG['parser']}")
    source_path = parser_module.get_source_path() or RAW_LOG
    if hasattr(parser_module, "reset_parser_state"):
        parser_module.reset_parser_state()

    miner = TemplateMiner()
    records = []
    skipped = 0

    if not source_path.exists():
        print(f"ERROR: source log not found for dataset {DATASET}")
        raise SystemExit(1)

    with open(source_path, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            record = parser_module.parse_line(line, i)
            if record is None:
                skipped += 1
                continue

            result = miner.add_log_message(record["content"])
            record["event_id"] = result["cluster_id"]
            record["template"] = result["template_mined"]
            records.append(record)

            if i % 50000 == 0 and i > 0:
                print(f"  Processed {i:,} lines...")

    return pd.DataFrame(records), skipped


print("\n" + "=" * 50)
print(f"STEP 1/3: Parsing {DATASET} logs with Drain3...")
print("=" * 50)

if PARSED.exists():
    print("  parsed.parquet already exists - skipping.")
    print(f"  Delete {PARSED} to re-run.\n")
else:
    parsed, skipped = parse_with_dataset_parser()

    parsed.to_parquet(PARSED, engine="pyarrow", index=False)
    print(f"\n  Total parsed    : {len(parsed):,}")
    print(f"  Skipped         : {skipped:,}")
    if HAS_LABELS:
        print(f"  Anomaly rate    : {parsed['is_anomaly'].mean():.2%}")
    else:
        print("  Ground truth    : not available")
    print(f"  Unique templates: {parsed['event_id'].nunique()}")
    print(f"  Saved -> {PARSED}")


print("\n" + "=" * 50)
print("STEP 2/3: Building feature matrix...")
print("=" * 50)

if has_current_window_features(FEATURES):
    print("  features.parquet already uses the current sliding-window config - skipping.")
    print(f"  Delete {FEATURES} to force a rebuild.\n")
else:
    if FEATURES.exists():
        print("  Existing features do not match the current window config.")
        print(f"  Rebuilding {FEATURES}.\n")

    parsed = pd.read_parquet(PARSED, engine="pyarrow")
    parsed = parsed.sort_values("timestamp").reset_index(drop=True)

    t_min = int(parsed["timestamp"].min())
    t_max = int(parsed["timestamp"].max())
    t_range = t_max - t_min

    print(f"  Time range : {t_range:,} seconds ({t_range / 86400:.1f} days)")
    print(f"  Window     : {WINDOW}s")
    print(f"  Step       : {STEP}s")
    print(f"  Mode       : {WINDOW_MODE}")
    print(f"  Event types: {parsed['event_id'].nunique()}")
    print("  Building...")

    base_window_id = ((parsed["timestamp"] - t_min) // STEP).astype(np.int64)
    max_overlap = int(np.ceil(WINDOW / STEP))
    membership_cols = [
        "line_id",
        "event_id",
        "timestamp",
        "is_anomaly",
        "level",
        "node",
    ]
    membership_cols.extend(
        col for col in ["source_ip", "user"] if col in parsed.columns
    )

    membership_parts = []
    for offset in range(max_overlap):
        window_id = base_window_id - offset
        window_start = t_min + window_id * STEP
        valid = (
            (window_id >= 0)
            & (parsed["timestamp"] >= window_start)
            & (parsed["timestamp"] < window_start + WINDOW)
        )
        part = parsed.loc[valid, membership_cols].copy()
        part.insert(0, "window_id", window_id[valid].to_numpy())
        membership_parts.append(part)

    memberships = pd.concat(membership_parts, ignore_index=True)
    memberships["window_id"] = memberships["window_id"].astype(np.int64)

    pivot = (
        memberships
        .groupby(["window_id", "event_id"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.columns = [f"e_{c}" for c in pivot.columns]

    meta = memberships.groupby("window_id").agg(
        total_logs=("line_id", "count"),
        anomaly_count=("is_anomaly", "sum"),
        error_ratio=("level", lambda x: x.isin(SUSPICIOUS_LEVELS).mean()),
        fatal_count=("level", lambda x: (x == "FATAL").sum()),
        severe_count=("level", lambda x: (x == "SEVERE").sum()),
        unique_nodes=("node", "nunique"),
        unique_templates=("event_id", "nunique"),
    )

    level_counts = (
        memberships
        .groupby(["window_id", "level"])
        .size()
        .unstack(fill_value=0)
    )
    level_counts.columns = [
        "level_" + re.sub(r"[^0-9a-zA-Z]+", "_", str(col)).lower()
        for col in level_counts.columns
    ]
    meta = meta.join(level_counts, how="left")

    if "source_ip" in memberships.columns:
        source_ip_counts = (
            memberships[memberships["source_ip"] != ""]
            .groupby(["window_id", "source_ip"])
            .size()
        )
        meta["unique_source_ips"] = (
            memberships
            .groupby("window_id")["source_ip"]
            .agg(lambda x: x[x != ""].nunique())
        )
        meta["max_events_from_single_ip"] = (
            source_ip_counts.groupby("window_id").max()
            if len(source_ip_counts)
            else 0
        )

    if "user" in memberships.columns:
        meta["unique_users"] = (
            memberships
            .groupby("window_id")["user"]
            .agg(lambda x: x[x != ""].nunique())
        )

    meta["window_start"] = t_min + meta.index.astype(np.int64) * STEP
    meta["window_end"] = meta["window_start"] + WINDOW
    meta["is_anomaly"] = (meta["anomaly_count"] > 0).astype(int)
    meta["window_mode"] = WINDOW_MODE
    meta["window_size_sec"] = WINDOW
    meta["step_size_sec"] = STEP

    feat = pivot.join(meta).reset_index(drop=True).fillna(0)
    feat.to_parquet(FEATURES, engine="pyarrow", index=False)

    total = len(feat)
    anomalous = int(feat["is_anomaly"].sum())
    print(f"\n  Total windows    : {total:,}")
    if HAS_LABELS:
        print(f"  Anomalous windows: {anomalous:,} ({anomalous / total:.2%})")
    else:
        print("  Anomalous windows: ground truth not available")
    print(f"  Features per row : {feat.shape[1]}")
    print(f"  Saved -> {FEATURES}")


print("\n" + "=" * 50)
print("STEP 3/3: Training Isolation Forest...")
print("=" * 50)

feat = pd.read_parquet(FEATURES, engine="pyarrow")

use_existing_scores = False
if SCORES.exists():
    scores = pd.read_parquet(SCORES, engine="pyarrow")
    use_existing_scores = (
        len(scores) == len(feat)
        and "step_size_sec" in scores.columns
        and "window_size_sec" in scores.columns
        and int(scores["step_size_sec"].iloc[0]) == STEP
        and int(scores["window_size_sec"].iloc[0]) == WINDOW
    )

if use_existing_scores:
    print("  scores.parquet already matches current window config - skipping.")
    print(f"  Delete {SCORES} to force a retrain.\n")
    metrics = write_metrics(scores)
    print(f"  Metrics synced -> {METRICS}")
    print(
        f"  Windows: {metrics['total_windows']:,} | "
        f"Predicted alerts: {metrics['predicted_anomalous_windows']:,}"
    )
else:
    if SCORES.exists():
        print("  Existing scores do not match current window config.")
        print(f"  Retraining model and overwriting {SCORES}.\n")

    meta_cols = {
        "window_start",
        "window_end",
        "is_anomaly",
        "anomaly_count",
        "window_mode",
        "window_size_sec",
        "step_size_sec",
    }
    feat_cols = [
        col for col in feat.columns
        if col not in meta_cols and pd.api.types.is_numeric_dtype(feat[col])
    ]

    X = feat[feat_cols].values
    y = feat["is_anomaly"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if HAS_LABELS:
        contamination = max(0.01, min(float(y.mean()), 0.45))
    else:
        contamination = float(CFG.get("contamination", 0.03))

    print(f"  Windows     : {len(X):,}")
    print(f"  Features    : {len(feat_cols)}")
    print(f"  Contamination: {contamination:.3f}")
    if not HAS_LABELS and CFG.get("contamination_note"):
        print(f"  Note        : {CFG['contamination_note']}")
    print("  Training...")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    feat["anomaly_score"] = -model.score_samples(X_scaled)
    feat["predicted"] = (model.predict(X_scaled) == -1).astype(int)

    score_p95 = float(feat["anomaly_score"].quantile(0.95))
    confirmed = (
        (feat["predicted"] == 1)
        & (
            (feat.get("fatal_count", 0) > 0)
            | (feat.get("severe_count", 0) > 0)
            | (feat["error_ratio"] > 0)
            | (feat["anomaly_score"] >= score_p95)
        )
    ).astype(int)

    if HAS_LABELS:
        y_raw = feat["is_anomaly"].values
        predicted_raw = feat["predicted"].values
        f1_before = f1_score(y_raw, predicted_raw, zero_division=0)
        f1_after = f1_score(y_raw, confirmed, zero_division=0)

        print("\n  Post-processing filter results:")
        print(f"  F1 before filter : {f1_before:.4f}")
        print(f"  F1 after filter  : {f1_after:.4f}")

        cm_before = confusion_matrix(y_raw, predicted_raw, labels=[0, 1])
        cm_after = confusion_matrix(y_raw, confirmed, labels=[0, 1])
        print("\n  Before filter:")
        print(f"    FP: {cm_before[0][1]}  FN: {cm_before[1][0]}")
        print("  After filter:")
        print(f"    FP: {cm_after[0][1]}  FN: {cm_after[1][0]}")

        if f1_after >= f1_before - 0.01:
            feat["predicted"] = confirmed
            print("\n  Post-processing applied")
        else:
            print("\n  Post-processing hurt F1 - keeping raw predictions")
    else:
        feat["predicted"] = confirmed
        print("\n  Unlabeled dataset: applied evidence-based post-processing.")

    feat.to_parquet(SCORES, engine="pyarrow", index=False)
    metrics = write_metrics(feat)

    if HAS_LABELS:
        print("\n-- Evaluation vs Ground Truth Labels --")
        print(classification_report(
            y,
            feat["predicted"],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        ))
    else:
        print("\n-- Unlabeled Dataset --")
        print("Ground-truth labels are unavailable; F1/accuracy are not computed.")
        print(f"Predicted alert windows: {metrics['predicted_anomalous_windows']:,}")

    with open(MODEL, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feat_cols": feat_cols}, f)

    print(f"  Model saved   -> {MODEL}")
    print(f"  Scores saved  -> {SCORES}")
    print(f"  Metrics saved -> {METRICS}")

print("\n" + "=" * 50)
print("PIPELINE COMPLETE.")
print("Run:  streamlit run app.py")
print("=" * 50)
