# api/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import duckdb
import pandas as pd
import json
from pathlib import Path
from typing import Optional

app = FastAPI(
    title="AIOps — BGL Anomaly Detection API",
    description=(
        "REST API for the AIOps capstone pipeline. "
        "Provides access to log data, anomaly detection results, "
        "alerts, clustering, and model evaluation metrics."
    ),
    version="1.0.0"
)

# Allow Streamlit to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Helper ─────────────────────────────────────────────────────────
def query(sql: str) -> list:
    """Execute SQL on Parquet files and return as list of dicts."""
    con = duckdb.connect()
    try:
        result = con.execute(sql).df()
        return result.to_dict(orient="records")
    finally:
        con.close()


# ══════════════════════════════════════════════════════════════════
# ROOT
# ══════════════════════════════════════════════════════════════════
@app.get("/", tags=["Health"])
def root():
    return {
        "status":  "ok",
        "service": "AIOps BGL Anomaly Detection API",
        "version": "1.0.0",
        "docs":    "/docs"
    }


# ══════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════
@app.get("/stats", tags=["Overview"])
def get_stats():
    """
    High-level system statistics.
    Returns total log lines, unique templates,
    anomalous windows, and anomaly rate.
    """
    parsed_path = "output/parsed.parquet"
    scores_path = "output/scores.parquet"

    if not Path(parsed_path).exists():
        return {"error": "Run pipeline.py first"}

    result = query(f"""
        SELECT
            COUNT(*)                        AS total_logs,
            SUM(is_anomaly)                 AS anomalous_lines,
            ROUND(AVG(is_anomaly) * 100, 2) AS anomaly_rate_pct,
            COUNT(DISTINCT event_id)        AS unique_templates
        FROM '{parsed_path}'
    """)

    scores_result = query(f"""
        SELECT
            COUNT(*)            AS total_windows,
            SUM(predicted)      AS anomalous_windows,
            ROUND(MIN(anomaly_score), 4) AS score_min,
            ROUND(MAX(anomaly_score), 4) AS score_max,
            ROUND(AVG(anomaly_score), 4) AS score_mean
        FROM '{scores_path}'
    """)

    return {**result[0], **scores_result[0]}


# ══════════════════════════════════════════════════════════════════
# LOGS
# ══════════════════════════════════════════════════════════════════
@app.get("/logs", tags=["Logs"])
def get_logs(
    level:        Optional[str]  = Query(None,
                      description="Filter by log level e.g. FATAL"),
    anomaly_only: bool           = Query(False,
                      description="Return only anomalous lines"),
    search:       Optional[str]  = Query(None,
                      description="Search in content field"),
    limit:        int            = Query(100, ge=1, le=1000,
                      description="Max rows to return"),
    offset:       int            = Query(0,   ge=0,
                      description="Rows to skip for pagination")
):
    """
    Paginated log lines with optional filters.
    Queries DuckDB directly on Parquet — no full load.
    """
    where = ["1=1"]
    if level:        where.append(f"level = '{level}'")
    if anomaly_only: where.append("is_anomaly = 1")
    if search:
        safe = search.replace("'", "''")
        where.append(f"content ILIKE '%{safe}%'")

    where_sql = " AND ".join(where)

    rows = query(f"""
        SELECT line_id, date, node, level,
               is_anomaly, template, content
        FROM 'output/parsed.parquet'
        WHERE {where_sql}
        ORDER BY line_id
        LIMIT {limit} OFFSET {offset}
    """)

    total = query(f"""
        SELECT COUNT(*) AS total
        FROM 'output/parsed.parquet'
        WHERE {where_sql}
    """)[0]["total"]

    return {
        "total":  total,
        "limit":  limit,
        "offset": offset,
        "data":   rows
    }


# ══════════════════════════════════════════════════════════════════
# ANOMALIES
# ══════════════════════════════════════════════════════════════════
@app.get("/anomalies", tags=["Anomalies"])
def get_anomalies(
    min_score: float = Query(0.0,
                   description="Minimum anomaly score"),
    limit:     int   = Query(50, ge=1, le=500,
                   description="Max windows to return"),
    offset:    int   = Query(0,  ge=0)
):
    """
    Anomalous windows ordered by score descending.
    Returns window timestamps, scores, and log counts.
    """
    rows = query(f"""
        SELECT
            window_start,
            window_end,
            ROUND(anomaly_score, 4) AS anomaly_score,
            predicted,
            is_anomaly              AS ground_truth,
            total_logs,
            anomaly_count
        FROM 'output/scores.parquet'
        WHERE predicted = 1
          AND anomaly_score >= {min_score}
        ORDER BY anomaly_score DESC
        LIMIT {limit} OFFSET {offset}
    """)
    return {"total": len(rows), "data": rows}


# ══════════════════════════════════════════════════════════════════
# ALERTS
# ══════════════════════════════════════════════════════════════════
@app.get("/alerts", tags=["Alerts"])
def get_alerts(
    severity:  Optional[str] = Query(None,
                   description="CRITICAL, HIGH, MEDIUM, or LOW"),
    min_score: float         = Query(0.0),
    limit:     int           = Query(50, ge=1, le=500),
    offset:    int           = Query(0,  ge=0)
):
    """
    Generated alerts with severity levels and cluster assignments.
    """
    alerts_path = "output/alerts.parquet"
    if not Path(alerts_path).exists():
        return {"error": "Run alerts.py first"}

    where = [f"anomaly_score >= {min_score}"]
    if severity:
        where.append(f"severity = '{severity.upper()}'")

    where_sql = " AND ".join(where)

    rows = query(f"""
        SELECT
            window_start,
            window_end,
            ROUND(anomaly_score, 4) AS anomaly_score,
            severity,
            top_template,
            cluster_id,
            cluster_label,
            anomaly_count,
            total_logs
        FROM '{alerts_path}'
        WHERE {where_sql}
        ORDER BY anomaly_score DESC
        LIMIT {limit} OFFSET {offset}
    """)

    total = query(f"""
        SELECT COUNT(*) AS total
        FROM '{alerts_path}'
        WHERE {where_sql}
    """)[0]["total"]

    return {"total": total, "data": rows}


# ══════════════════════════════════════════════════════════════════
# ALERTS SUMMARY
# ══════════════════════════════════════════════════════════════════
@app.get("/alerts/summary", tags=["Alerts"])
def get_alert_summary():
    """
    Alert count breakdown by severity and clustering stats.
    """
    alerts_path = "output/alerts.parquet"
    if not Path(alerts_path).exists():
        return {"error": "Run alerts.py first"}

    sev = query(f"""
        SELECT
            COUNT(*)  AS total_alerts,
            SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END)
                      AS critical,
            SUM(CASE WHEN severity = 'HIGH'     THEN 1 ELSE 0 END)
                      AS high,
            SUM(CASE WHEN severity = 'MEDIUM'   THEN 1 ELSE 0 END)
                      AS medium,
            SUM(CASE WHEN severity = 'LOW'      THEN 1 ELSE 0 END)
                      AS low,
            COUNT(DISTINCT CASE WHEN cluster_id >= 0
                  THEN cluster_id END)          AS clusters,
            SUM(CASE WHEN cluster_id = -1
                  THEN 1 ELSE 0 END)            AS unique_alerts
        FROM '{alerts_path}'
    """)[0]

    total     = sev["total_alerts"]
    groups    = sev["clusters"] + sev["unique_alerts"]
    reduction = round((1 - groups / total) * 100, 1) if total > 0 else 0

    return {
        **sev,
        "distinct_groups":     groups,
        "noise_reduction_pct": reduction
    }


# ══════════════════════════════════════════════════════════════════
# CLUSTERS
# ══════════════════════════════════════════════════════════════════
@app.get("/clusters", tags=["Alerts"])
def get_clusters():
    """
    Alert cluster summary — size, max score, severity breakdown.
    """
    alerts_path = "output/alerts.parquet"
    if not Path(alerts_path).exists():
        return {"error": "Run alerts.py first"}

    rows = query(f"""
        SELECT
            cluster_id,
            cluster_label,
            COUNT(*)                        AS alert_count,
            ROUND(MAX(anomaly_score), 4)    AS max_score,
            ROUND(AVG(anomaly_score), 4)    AS avg_score,
            SUM(CASE WHEN severity = 'CRITICAL'
                THEN 1 ELSE 0 END)          AS critical_count
        FROM '{alerts_path}'
        GROUP BY cluster_id, cluster_label
        ORDER BY alert_count DESC
    """)
    return {"total_clusters": len(rows), "data": rows}


# ══════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════
@app.get("/metrics", tags=["Evaluation"])
def get_metrics():
    """
    Model evaluation metrics — F1, precision, recall,
    accuracy, confusion matrix.
    """
    path = Path("output/metrics.json")
    if not path.exists():
        return {"error": "Run pipeline.py to generate metrics.json"}
    return json.loads(path.read_text())


# ══════════════════════════════════════════════════════════════════
# CLUSTERING COMPARISON
# ══════════════════════════════════════════════════════════════════
@app.get("/clustering/comparison", tags=["Evaluation"])
def get_clustering_comparison():
    """
    TF-IDF vs MiniLM clustering comparison —
    silhouette scores, cluster counts, noise reduction.
    """
    path = Path("output/clustering_comparison.json")
    if not path.exists():
        return {
            "error": "Run alerts_minilm.py to generate comparison"
        }
    data = json.loads(path.read_text())
    return {
        "tfidf": {
            "method":          "TF-IDF + DBSCAN",
            "eps":             0.5,
            "clusters":        data.get("tfidf_clusters"),
            "unique":          data.get("tfidf_unique"),
            "silhouette":      data.get("tfidf_silhouette"),
            "noise_reduction": round(
                (1 - (
                    data.get("tfidf_clusters", 0) +
                    data.get("tfidf_unique", 0)
                ) / 535) * 100, 1
            )
        },
        "minilm": {
            "method":          "MiniLM + DBSCAN",
            "eps":             0.4,
            "clusters":        data.get("minilm_clusters"),
            "unique":          data.get("minilm_unique"),
            "silhouette":      data.get("minilm_silhouette"),
            "noise_reduction": round(
                (1 - (
                    data.get("minilm_clusters", 0) +
                    data.get("minilm_unique", 0)
                ) / 535) * 100, 1
            )
        }
    }


# ══════════════════════════════════════════════════════════════════
# WINDOW LOGS
# ══════════════════════════════════════════════════════════════════
@app.get("/logs/window", tags=["Logs"])
def get_window_logs(
    window_start: int = Query(...,
                      description="Window start Unix timestamp"),
    window_end:   int = Query(...,
                      description="Window end Unix timestamp"),
    limit:        int = Query(50, ge=1, le=200)
):
    """
    Log lines for a specific time window.
    Used by dashboard to show lines inside alert expanders.
    """
    rows = query(f"""
        SELECT node, level, is_anomaly,
               template, content, timestamp
        FROM 'output/parsed.parquet'
        WHERE timestamp >= {window_start}
          AND timestamp <  {window_end}
        ORDER BY timestamp
        LIMIT {limit}
    """)
    return {"window_start": window_start,
            "window_end":   window_end,
            "count":        len(rows),
            "data":         rows}


# ══════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════
@app.get("/datasets", tags=["Overview"])
def list_datasets():
    """
    List available processed datasets.
    """
    output_dir = Path("output")
    datasets   = []

    if (output_dir / "parsed.parquet").exists():
        datasets.append({
            "name":        "bgl",
            "description": "BlueGene/L supercomputer logs",
            "status":      "ready",
            "files": {
                "parsed":   str(output_dir / "parsed.parquet"),
                "scores":   str(output_dir / "scores.parquet"),
                "alerts":   str(output_dir / "alerts.parquet"),
                "metrics":  str(output_dir / "metrics.json")
            }
        })

    return {"datasets": datasets, "count": len(datasets)}