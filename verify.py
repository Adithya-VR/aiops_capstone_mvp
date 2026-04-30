# Run this as a quick check script: verify.py
import pandas as pd
import json
from pathlib import Path

scores  = pd.read_parquet("output/scores.parquet", engine="pyarrow")
metrics = json.loads(Path("output/metrics.json").read_text())
cm      = metrics["confusion_matrix"]

# Check 1: totals must add up
total_from_cm = (cm["true_negative"] + cm["false_positive"] +
                 cm["false_negative"] + cm["true_positive"])
assert total_from_cm == metrics["total_windows"], \
    f"CM total {total_from_cm} != windows {metrics['total_windows']}"

# Check 2: precision = TP / (TP + FP)
manual_precision = cm["true_positive"] / (
    cm["true_positive"] + cm["false_positive"]
)
assert abs(manual_precision - metrics["precision_anomaly"]) < 0.01, \
    f"Precision mismatch: {manual_precision:.3f} vs {metrics['precision_anomaly']}"

# Check 3: anomalous windows match ground truth count
gt_anomalous = scores["is_anomaly"].sum()
assert gt_anomalous == metrics["anomalous_windows"], \
    f"Anomaly count mismatch: {gt_anomalous} vs {metrics['anomalous_windows']}"

# Check 4: scores are in expected range
assert scores["anomaly_score"].min() >= 0, "Negative anomaly scores found"
assert scores["anomaly_score"].max() <= 2, "Anomaly scores unusually high"

# Check 5: predicted column only contains 0 and 1
assert set(scores["predicted"].unique()).issubset({0, 1}), \
    "Predicted column has unexpected values"

print("All checks passed.")
print(f"  Total windows : {metrics['total_windows']:,}")
print(f"  True Positives: {cm['true_positive']}")
print(f"  True Negatives: {cm['true_negative']}")
print(f"  False Positives: {cm['false_positive']}")
print(f"  False Negatives: {cm['false_negative']}")
print(f"  F1 (Anomaly)  : {metrics['f1_anomaly']}")
print(f"  Accuracy      : {metrics['accuracy']:.2%}")

# Add to verify.py or run interactively
import pandas as pd

scores = pd.read_parquet("output/scores.parquet", engine="pyarrow")
parsed = pd.read_parquet("output/parsed.parquet", engine="pyarrow")

top5 = scores.nlargest(5, "anomaly_score")
print("\nTop 5 most anomalous windows:")
for _, row in top5.iterrows():
    print(f"\nWindow {int(row['window_start'])} | "
          f"Score: {row['anomaly_score']:.3f} | "
          f"Anomalous lines: {int(row['anomaly_count'])}")
    
    # Get actual log lines
    logs = parsed[
        (parsed["timestamp"] >= row["window_start"]) &
        (parsed["timestamp"] <  row["window_end"])  &
        (parsed["is_anomaly"] == 1)
    ].head(5)
    
    for _, log in logs.iterrows():
        print(f"  [{log['level']}] {log['node']}: {log['template']}")