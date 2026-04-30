import re
import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path
from drain3 import TemplateMiner
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Paths ──────────────────────────────────────────────────────────
Path("output").mkdir(exist_ok=True)
BGL_LOG  = Path("data/BGL.log")
PARSED   = Path("output/parsed.parquet")
FEATURES = Path("output/features.parquet")
SCORES   = Path("output/scores.parquet")
MODEL    = Path("output/model.pkl")
METRICS  = Path("output/metrics.json")

WINDOW = 3600
STEP = 1800
WINDOW_MODE = "sliding"


def write_metrics(scores: pd.DataFrame, path: Path = METRICS) -> dict:
    """Write metrics.json from the same scores DataFrame used by the app/API."""
    y_true = scores["is_anomaly"].astype(int)
    y_pred = scores["predicted"].astype(int)
    anomaly_score = scores["anomaly_score"].astype(float)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    try:
        roc_auc = float(roc_auc_score(y_true, anomaly_score))
    except ValueError:
        roc_auc = 0.0

    metrics = {
        "dataset": "bgl",
        "total_windows": int(len(scores)),
        "anomalous_windows": int(y_true.sum()),
        "normal_windows": int((y_true == 0).sum()),
        "predicted_anomalous_windows": int(y_pred.sum()),
        "precision_anomaly": round(
            float(precision_score(y_true, y_pred, zero_division=0)), 4
        ),
        "recall_anomaly": round(
            float(recall_score(y_true, y_pred, zero_division=0)), 4
        ),
        "f1_anomaly": round(
            float(f1_score(y_true, y_pred, zero_division=0)), 4
        ),
        "precision_normal": round(
            float(precision_score(
                y_true, y_pred, pos_label=0, zero_division=0
            )),
            4,
        ),
        "recall_normal": round(
            float(recall_score(
                y_true, y_pred, pos_label=0, zero_division=0
            )),
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
        "score_min": round(float(anomaly_score.min()), 4),
        "score_max": round(float(anomaly_score.max()), 4),
        "score_mean": round(float(anomaly_score.mean()), 4),
        "score_p85": round(float(anomaly_score.quantile(0.85)), 4),
        "score_p90": round(float(anomaly_score.quantile(0.90)), 4),
        "score_p95": round(float(anomaly_score.quantile(0.95)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def has_sliding_window_features(path: Path) -> bool:
    """Return True when features match the current sliding-window config."""
    if not path.exists():
        return False

    try:
        sample = pd.read_parquet(
            path,
            columns=["window_start", "window_end"],
            engine="pyarrow",
        )
    except Exception:
        return False

    if sample.empty:
        return False

    span = sample["window_end"] - sample["window_start"]
    return bool((span == WINDOW).all())

if not BGL_LOG.exists():
    print("ERROR: data/BGL.log not found.")
    exit(1)

# ══════════════════════════════════════════════════════════════════
# STEP 1 — PARSE
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("STEP 1/3: Parsing BGL.log with Drain3...")
print("="*50)

if PARSED.exists():
    print("  parsed.parquet already exists — skipping.")
    print("  Delete output/parsed.parquet to re-run.\n")
else:
    # BGL_RE = re.compile(
    #     r'^(?P<label>\S+)\s+'
    #     r'(?P<timestamp>\d+)\s+'
    #     r'(?P<date>\S+)\s+'
    #     r'(?P<node>\S+)\s+'
    #     r'(?P<time>\S+)\s+'
    #     r'(?P<node2>\S+)\s+'
    #     r'(?P<type>\S+)\s+'
    #     r'(?P<component>\S+)\s+'
    #     r'(?P<level>\S+)\s+'
    #     r'(?P<content>.+)$'
    # )

    # Replace the single BGL_RE with two patterns:

# Format 1: Full format with node (most lines)
# LABEL TIMESTAMP DATE NODE TIME NODE2 TYPE COMPONENT LEVEL CONTENT
    BGL_RE_FULL = re.compile(
        r'^(?P<label>\S+)\s+'
        r'(?P<timestamp>\d+)\s+'
        r'(?P<date>\S+)\s+'
        r'(?P<node>\S+)\s+'
        r'(?P<time>\S+)\s+'
        r'(?P<node2>\S+)\s+'
        r'(?P<type>\S+)\s+'
        r'(?P<component>\S+)\s+'
        r'(?P<level>INFO|WARN|WARNING|ERROR|FATAL|SEVERE|FAILURE|CRITICAL)\s+'
        r'(?P<content>.+)$'
    )

# Format 2: Short format where node = "-" (no node2 field)
# LABEL TIMESTAMP DATE - TIME TYPE COMPONENT LEVEL CONTENT
    BGL_RE_SHORT = re.compile(
        r'^(?P<label>\S+)\s+'
        r'(?P<timestamp>\d+)\s+'
        r'(?P<date>\S+)\s+'
        r'-\s+'                      # node is literally "-"
        r'(?P<time>\S+)\s+'
        r'(?P<type>\S+)\s+'
        r'(?P<component>\S+)\s+'
        r'(?P<level>INFO|WARN|WARNING|ERROR|FATAL|SEVERE|FAILURE|CRITICAL)\s+'
        r'(?P<content>.+)$'
    )

    miner   = TemplateMiner()
    records = []
    skipped = 0

    with open(BGL_LOG, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # Try full format first, then short format
            m = BGL_RE_FULL.match(line)
            fmt = "full"
            if not m:
                m = BGL_RE_SHORT.match(line)
                fmt = "short"

            if not m:
                skipped += 1
                continue

            d = m.groupdict()

            # For short format, node2 doesn't exist — use "-"
            node = d.get("node", "-")
            if node == "-" or fmt == "short":
                node = "SYSTEM"   # cleaner label for dashboard

            r = miner.add_log_message(d["content"])
            records.append({
                "line_id":    i,
                "is_anomaly": int(d["label"] != "-"),
                "timestamp":  int(d["timestamp"]),
                "date":       d["date"],
                "node":       node,
                "level":      d["level"],
                "component":  d["component"],
                "content":    d["content"],
                "event_id":   r["cluster_id"],
                "template":   r["template_mined"],
            })

            if i % 50000 == 0 and i > 0:
                print(f"  Processed {i:,} lines...")
    parsed = pd.DataFrame(records)
    # parsed.to_parquet(PARSED, engine="pyarrow", index=False)

 # ── Normalize log levels ───────────────────────────────────────
    # BGL valid levels — anything else is a parsing artifact
    # VALID_LEVELS = {"INFO", "WARN", "WARNING", "ERROR",
    #                 "FATAL", "SEVERE", "FAILURE", "CRITICAL"}

    # parsed["level"] = parsed["level"].apply(
    #     lambda x: x if x in VALID_LEVELS else "OTHER"
    # )

    parsed.to_parquet(PARSED, engine="pyarrow", index=False)
    print(f"\n  Total parsed    : {len(parsed):,}")
    print(f"  Skipped         : {skipped:,}")
    print(f"  Anomaly rate    : {parsed['is_anomaly'].mean():.2%}")
    print(f"  Unique templates: {parsed['event_id'].nunique()}")
    print(f"  Saved -> {PARSED}")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("STEP 2/3: Building feature matrix...")
print("="*50)

if has_sliding_window_features(FEATURES):
    print("  features.parquet already uses sliding windows - skipping.")
    print("  Delete output/features.parquet to force a rebuild.\n")
else:
    if FEATURES.exists():
        print("  Existing features are not sliding-window features.")
        print("  Rebuilding output/features.parquet with current config.\n")

    parsed = pd.read_parquet(PARSED, engine="pyarrow")
    parsed = parsed.sort_values("timestamp").reset_index(drop=True)

    t_min   = parsed["timestamp"].min()
    t_max   = parsed["timestamp"].max()
    t_range = t_max - t_min

    print(f"  Time range : {t_range:,} seconds ({t_range/86400:.1f} days)")
    print(f"  Window     : {WINDOW}s (1 hour)")
    print(f"  Step       : {STEP}s (30 min)")
    print(f"  Mode       : {WINDOW_MODE}")
    print(f"  Event types: {parsed['event_id'].nunique()}")
    print(f"  Building... (3-5 minutes)")

    base_window_id = (
        (parsed["timestamp"] - t_min) // STEP
    ).astype(np.int64)

    membership_parts = []
    max_overlap = int(np.ceil(WINDOW / STEP))
    membership_cols = [
        "line_id",
        "event_id",
        "timestamp",
        "is_anomaly",
        "level",
        "node",
    ]

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

    # Count each event type per sliding window - vectorized, no row loop
    pivot = (
        memberships
        .groupby(["window_id", "event_id"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.columns = [f"e_{c}" for c in pivot.columns]

    # Metadata per sliding window
    meta = memberships.groupby("window_id").agg(
        total_logs       = ("line_id",    "count"),
        anomaly_count    = ("is_anomaly", "sum"),
        error_ratio      = ("level",
                            lambda x: x.isin(["SEVERE", "ERROR", "FAILURE", "FATAL"]).mean()),
        fatal_count      = ("level",                        # NEW
                            lambda x: (x == "FATAL").sum()),
        severe_count     = ("level",                        # NEW
                            lambda x: (x == "SEVERE").sum()),
        unique_nodes     = ("node",       "nunique"),
        unique_templates = ("event_id",   "nunique"),
    )
    meta["window_start"] = t_min + meta.index.astype(np.int64) * STEP
    meta["window_end"] = meta["window_start"] + WINDOW
    meta["is_anomaly"] = (meta["anomaly_count"] > 0).astype(int)
    meta["window_mode"] = WINDOW_MODE
    meta["window_size_sec"] = WINDOW
    meta["step_size_sec"] = STEP

    # Combine
    feat = pivot.join(meta).reset_index(drop=True).fillna(0)
    feat.to_parquet(FEATURES, engine="pyarrow", index=False)

    total     = len(feat)
    anomalous = int(feat["is_anomaly"].sum())
    print(f"\n  Total windows    : {total:,}")
    print(f"  Anomalous windows: {anomalous:,} ({anomalous/total:.2%})")
    print(f"  Features per row : {feat.shape[1]}")
    print(f"  Saved -> {FEATURES}")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — MODEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("STEP 3/3: Training Isolation Forest...")
print("="*50)

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
    print("  Delete output/scores.parquet to force a retrain.\n")
    metrics = write_metrics(scores)
    print(f"  Metrics synced -> {METRICS}")
    print(
        f"  Windows: {metrics['total_windows']:,} | "
        f"F1 anomaly: {metrics['f1_anomaly']:.4f}"
    )
else:
    if SCORES.exists():
        print("  Existing scores do not match current window config.")
        print("  Retraining model and overwriting output/scores.parquet.\n")

    META_COLS = [
        "window_start",
        "window_end",
        "is_anomaly",
        "anomaly_count",
        "window_mode",
        "window_size_sec",
        "step_size_sec",
    ]
    FEAT_COLS = [
        c for c in feat.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(feat[c])
    ]

    X = feat[FEAT_COLS].values
    y = feat["is_anomaly"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = max(0.01, min(float(y.mean()), 0.45))
    print(f"  Windows     : {len(X):,}")
    print(f"  Features    : {len(FEAT_COLS)}")
    print(f"  Anomaly rate: {contamination:.3f}")
    print(f"  Training... (1-2 minutes)")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=1
    )
    model.fit(X_scaled)

    feat["anomaly_score"] = -model.score_samples(X_scaled)
    feat["predicted"]     = (
        model.predict(X_scaled) == -1
    ).astype(int)

    # ── Fix C: Post-processing filter ─────────────────────────────
    # A predicted anomaly must have at least one error-level log line
    # to be confirmed. This directly reduces false positives by
    # filtering windows that look unusual statistically but have
    # no actual error evidence in the logs.
    y_raw        = feat["is_anomaly"].values
    predicted_raw = feat["predicted"].values

    # CORRECT — check both fatal_count AND severe_count. Now replacing this with below code to 
    # confirmed = (
    # (feat["predicted"] == 1) &
    # (
    #     (feat["fatal_count"] > 0) |   # FATAL lines present
    #     (feat["severe_count"] > 0) |  # SEVERE lines present
    #     (feat["error_ratio"] > 0)     # any error ratio
    # )
    # ).astype(int)
    # Replacing the above, with the below code

    # Improved (also passes high-confidence predictions through):
    score_p95 = float(pd.Series(
    feat["anomaly_score"]
    ).quantile(0.95))

    confirmed = (
    (feat["predicted"] == 1) &
    (
        (feat["fatal_count"] > 0) |
        (feat["severe_count"] > 0) |
        (feat["error_ratio"] > 0) |
        # High confidence predictions pass regardless of error lines
        # These are likely real anomalies the model is very sure about
        (feat["anomaly_score"] >= score_p95)
    )
    ).astype(int)

    f1_before = f1_score(y_raw, predicted_raw, zero_division=0)
    f1_after  = f1_score(y_raw, confirmed,      zero_division=0)

    print(f"\n  Post-processing filter results:")
    print(f"  F1 before filter : {f1_before:.4f}")
    print(f"  F1 after filter  : {f1_after:.4f}")

    cm_before = confusion_matrix(y_raw, predicted_raw)
    cm_after  = confusion_matrix(y_raw, confirmed)

    print(f"\n  Before filter:")
    print(f"    FP: {cm_before[0][1]}  FN: {cm_before[1][0]}")
    print(f"  After filter:")
    print(f"    FP: {cm_after[0][1]}  FN: {cm_after[1][0]}")

    # Use filtered version if it improves or matches F1
    if f1_after >= f1_before - 0.01:
        feat["predicted"] = confirmed
        print(f"\n  Post-processing applied")
    else:
        print(f"\n  Post-processing hurt F1 - keeping raw predictions")
    feat.to_parquet(SCORES, engine="pyarrow", index=False)
    metrics = write_metrics(feat)

    print("\n-- Evaluation vs Ground Truth Labels --")
    print(classification_report(
        y, feat["predicted"],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

    pickle.dump(
        {"model": model, "scaler": scaler, "feat_cols": FEAT_COLS},
        open(MODEL, "wb")
    )

    print(f"  Model saved  -> {MODEL}")
    print(f"  Scores saved -> {SCORES}")
    print(f"  Metrics saved -> {METRICS}")

print("\n" + "="*50)
print("PIPELINE COMPLETE.")
print("Run:  streamlit run app.py")
print("="*50)
