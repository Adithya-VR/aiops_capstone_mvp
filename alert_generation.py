import pandas as pd


def severity_for_score(score: float, p70: float, p85: float, p95: float) -> str:
    if score >= p95:
        return "CRITICAL"
    if score >= p85:
        return "HIGH"
    if score >= p70:
        return "MEDIUM"
    return "LOW"


def generate_alerts(scores: pd.DataFrame, parsed: pd.DataFrame, has_labels: bool) -> pd.DataFrame:
    predicted = scores[scores["predicted"] == 1].copy()
    if predicted.empty:
        return pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "anomaly_score",
                "anomaly_count",
                "total_logs",
                "severity",
                "top_template",
                "top_level",
            ]
        )

    p95 = predicted["anomaly_score"].quantile(0.95)
    p85 = predicted["anomaly_score"].quantile(0.85)
    p70 = predicted["anomaly_score"].quantile(0.70)

    alerts = []
    for _, row in predicted.iterrows():
        window_logs = parsed[
            (parsed["timestamp"] >= row["window_start"])
            & (parsed["timestamp"] < row["window_end"])
        ]

        if has_labels:
            representative_logs = window_logs[window_logs["is_anomaly"] == 1]
            if representative_logs.empty:
                representative_logs = window_logs
        else:
            representative_logs = window_logs

        if representative_logs.empty:
            continue

        top_template = representative_logs["template"].value_counts().index[0]
        top_level = representative_logs["level"].value_counts().index[0]

        alerts.append(
            {
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "anomaly_score": row["anomaly_score"],
                "anomaly_count": row["anomaly_count"],
                "total_logs": row["total_logs"],
                "severity": severity_for_score(
                    row["anomaly_score"], p70, p85, p95
                ),
                "top_template": top_template,
                "top_level": top_level,
            }
        )

    return pd.DataFrame(alerts)
