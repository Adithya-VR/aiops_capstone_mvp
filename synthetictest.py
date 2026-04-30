# synthetic_test.py
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Load model
bundle   = pickle.load(open("output/model.pkl", "rb"))
model    = bundle["model"]
scaler   = bundle["scaler"]
feat_cols = bundle["feat_cols"]

# Load feature matrix
feat = pd.read_parquet("output/features.parquet", engine="pyarrow")

# ── Test 1: A perfectly normal window should score LOW ─────────────
# Take the median of all normal windows = as normal as it gets
normal_windows = feat[feat["is_anomaly"] == 0][feat_cols]
median_normal  = normal_windows.median().values.reshape(1, -1)
score_normal   = -model.score_samples(scaler.transform(median_normal))[0]
print(f"Median normal window score: {score_normal:.4f}")
assert score_normal < 0.35, f"Normal window scored too high: {score_normal}"
print("  ✓ Normal window correctly scores low")

# ── Test 2: A synthetic anomaly should score HIGH ──────────────────
# Create a window where every feature is 3 standard deviations
# above its mean — clearly abnormal behavior
X_all    = feat[feat_cols].values
X_mean   = X_all.mean(axis=0)
X_std    = X_all.std(axis=0) + 1e-8  # avoid division by zero

# Extreme anomaly: mean + 3 * std for every feature
synthetic_anomaly = (X_mean + 3 * X_std).reshape(1, -1)
score_anomaly     = -model.score_samples(
    scaler.transform(synthetic_anomaly)
)[0]
print(f"\nSynthetic extreme anomaly score: {score_anomaly:.4f}")
assert score_anomaly > 0.40, \
    f"Synthetic anomaly scored too low: {score_anomaly}"
print("  ✓ Synthetic extreme anomaly correctly scores high")

# ── Test 3: Real anomalous windows should score higher than normals─
actual_anomalies = feat[feat["is_anomaly"] == 1][feat_cols].values
actual_normals   = feat[feat["is_anomaly"] == 0][feat_cols].values

scores_anomaly = -model.score_samples(scaler.transform(actual_anomalies))
scores_normal  = -model.score_samples(scaler.transform(actual_normals))

mean_anom   = scores_anomaly.mean()
mean_normal = scores_normal.mean()

print(f"\nMean score — actual anomalies: {mean_anom:.4f}")
print(f"Mean score — actual normals  : {mean_normal:.4f}")
assert mean_anom > mean_normal, \
    "Anomalous windows do not score higher than normal ones on average"
print("  ✓ Anomalies score higher than normals on average")

print("\nAll synthetic tests passed. Model is working correctly.")