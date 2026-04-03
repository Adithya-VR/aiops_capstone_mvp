import re
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from drain3 import TemplateMiner
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ── Paths ──────────────────────────────────────────────────────────
Path("output").mkdir(exist_ok=True)
BGL_LOG  = Path("data/BGL.log")
PARSED   = Path("output/parsed.parquet")
FEATURES = Path("output/features.parquet")
SCORES   = Path("output/scores.parquet")
MODEL    = Path("output/model.pkl")

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
    BGL_RE = re.compile(
        r'^(?P<label>\S+)\s+'
        r'(?P<timestamp>\d+)\s+'
        r'(?P<date>\S+)\s+'
        r'(?P<node>\S+)\s+'
        r'(?P<time>\S+)\s+'
        r'(?P<node2>\S+)\s+'
        r'(?P<type>\S+)\s+'
        r'(?P<component>\S+)\s+'
        r'(?P<level>\S+)\s+'
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
            m = BGL_RE.match(line)
            if not m:
                skipped += 1
                continue
            d = m.groupdict()
            r = miner.add_log_message(d["content"])
            records.append({
                "line_id":    i,
                "is_anomaly": int(d["label"] != "-"),
                "timestamp":  int(d["timestamp"]),
                "date":       d["date"],
                "node":       d["node"],
                "level":      d["level"],
                "component":  d["component"],
                "content":    d["content"],
                "event_id":   r["cluster_id"],
                "template":   r["template_mined"],
            })
            if i % 50000 == 0 and i > 0:
                print(f"  Processed {i:,} lines...")

    parsed = pd.DataFrame(records)
    parsed.to_parquet(PARSED, engine="pyarrow", index=False)
    print(f"\n  Total parsed    : {len(parsed):,}")
    print(f"  Skipped         : {skipped:,}")
    print(f"  Anomaly rate    : {parsed['is_anomaly'].mean():.2%}")
    print(f"  Unique templates: {parsed['event_id'].nunique()}")
    print(f"  Saved → {PARSED}")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — FEATURES
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("STEP 2/3: Building feature matrix...")
print("="*50)

if FEATURES.exists():
    print("  features.parquet already exists — skipping.")
    print("  Delete output/features.parquet to re-run.\n")
else:
    parsed = pd.read_parquet(PARSED, engine="pyarrow")
    parsed = parsed.sort_values("timestamp").reset_index(drop=True)

    t_min   = parsed["timestamp"].min()
    t_max   = parsed["timestamp"].max()
    t_range = t_max - t_min

    # 1-hour windows, 30-min steps
    # Gives ~10,000 windows for the full dataset
    WINDOW  = 3600
    STEP    = 1800

    print(f"  Time range : {t_range:,} seconds ({t_range/86400:.1f} days)")
    print(f"  Window     : {WINDOW}s (1 hour)")
    print(f"  Step       : {STEP}s (30 min)")
    print(f"  Event types: {parsed['event_id'].nunique()}")
    print(f"  Building... (3-5 minutes)")

    # Assign window bucket to every log line
    parsed["window_id"] = (
        (parsed["timestamp"] - t_min) // WINDOW
    ).astype(int)

    # Count each event type per window — vectorized, no loop
    pivot = (
        parsed
        .groupby(["window_id", "event_id"])
        .size()
        .unstack(fill_value=0)
    )
    pivot.columns = [f"e_{c}" for c in pivot.columns]

    # Metadata per window
    meta = parsed.groupby("window_id").agg(
        window_start     = ("timestamp",  "min"),
        window_end       = ("timestamp",  "max"),
        total_logs       = ("line_id",    "count"),
        anomaly_count    = ("is_anomaly", "sum"),
        error_ratio      = ("level",
                            lambda x: (x == "SEVERE").mean()),
        unique_nodes     = ("node",       "nunique"),
        unique_templates = ("event_id",   "nunique"),
    )
    meta["is_anomaly"] = (meta["anomaly_count"] > 0).astype(int)

    # Combine
    feat = pivot.join(meta).reset_index(drop=True).fillna(0)
    feat.to_parquet(FEATURES, engine="pyarrow", index=False)

    total     = len(feat)
    anomalous = int(feat["is_anomaly"].sum())
    print(f"\n  Total windows    : {total:,}")
    print(f"  Anomalous windows: {anomalous:,} ({anomalous/total:.2%})")
    print(f"  Features per row : {feat.shape[1]}")
    print(f"  Saved → {FEATURES}")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — MODEL
# ══════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("STEP 3/3: Training Isolation Forest...")
print("="*50)

if SCORES.exists():
    print("  scores.parquet already exists — skipping.")
    print("  Delete output/scores.parquet to re-run.\n")
else:
    feat = pd.read_parquet(FEATURES, engine="pyarrow")

    META_COLS = ["window_start", "window_end",
                 "is_anomaly",   "anomaly_count"]
    FEAT_COLS = [c for c in feat.columns if c not in META_COLS]

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
        n_jobs=-1
    )
    model.fit(X_scaled)

    feat["anomaly_score"] = -model.score_samples(X_scaled)
    feat["predicted"]     = (
        model.predict(X_scaled) == -1
    ).astype(int)

    feat.to_parquet(SCORES, engine="pyarrow", index=False)

    print("\n── Evaluation vs Ground Truth Labels ──")
    print(classification_report(
        y, feat["predicted"],
        target_names=["Normal", "Anomaly"],
        zero_division=0
    ))

    pickle.dump(
        {"model": model, "scaler": scaler, "feat_cols": FEAT_COLS},
        open(MODEL, "wb")
    )

    print(f"  Model saved  → {MODEL}")
    print(f"  Scores saved → {SCORES}")

print("\n" + "="*50)
print("PIPELINE COMPLETE.")
print("Run:  streamlit run app.py")
print("="*50)