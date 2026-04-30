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


def scalar(sql: str):
    rows = query(sql)
    if not rows:
        return None
    return next(iter(rows[0].values()))


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def csv_values(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def representative_log(window_start, window_end, template=None):
    """Most common concrete log line for a window/template."""
    template_filter = ""
    if template:
        template_filter = f"AND template = {sql_string(str(template))}"

    rows = query(f"""
        SELECT content, level, COUNT(*) AS count
        FROM 'output/parsed.parquet'
        WHERE timestamp >= {int(window_start)}
          AND timestamp <  {int(window_end)}
          {template_filter}
        GROUP BY content, level
        ORDER BY count DESC
        LIMIT 1
    """)

    if not rows and template:
        rows = query(f"""
            SELECT content, level, COUNT(*) AS count
            FROM 'output/parsed.parquet'
            WHERE timestamp >= {int(window_start)}
              AND timestamp <  {int(window_end)}
              AND level IN ('FATAL', 'SEVERE', 'ERROR', 'FAILURE')
            GROUP BY content, level
            ORDER BY count DESC
            LIMIT 1
        """)

    if not rows:
        rows = query(f"""
            SELECT content, level, COUNT(*) AS count
            FROM 'output/parsed.parquet'
            WHERE timestamp >= {int(window_start)}
              AND timestamp <  {int(window_end)}
            GROUP BY content, level
            ORDER BY count DESC
            LIMIT 1
        """)

    if not rows:
        return {"content": str(template or "unknown"), "level": "UNKNOWN"}

    return {"content": rows[0]["content"], "level": rows[0]["level"]}


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
    normal_only:  bool           = Query(False,
                      description="Return only normal lines"),
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
    levels = csv_values(level)
    if levels:
        where.append(
            "level IN (" + ", ".join(sql_string(v) for v in levels) + ")"
        )
    if anomaly_only: where.append("is_anomaly = 1")
    if normal_only:  where.append("is_anomaly = 0")
    if search:
        safe = search.replace("'", "''")
        where.append(f"content ILIKE '%{safe}%'")

    where_sql = " AND ".join(where)

    rows = query(f"""
        SELECT line_id, timestamp, date, node, level,
               is_anomaly, template, content
        FROM 'output/parsed.parquet'
        WHERE {where_sql}
        ORDER BY line_id
        LIMIT {int(limit)} OFFSET {int(offset)}
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


@app.get("/levels", tags=["Overview"])
def get_levels():
    """Distinct log levels available in parsed logs."""
    rows = query("""
        SELECT DISTINCT level
        FROM 'output/parsed.parquet'
        ORDER BY level
    """)
    return {"data": [r["level"] for r in rows]}


@app.get("/levels/distribution", tags=["Overview"])
def get_level_distribution():
    """Log count by level for dashboard pie charts."""
    rows = query("""
        SELECT level, COUNT(*) AS count
        FROM 'output/parsed.parquet'
        GROUP BY level
        ORDER BY count DESC
    """)
    return {"data": rows}


@app.get("/levels/anomaly-distribution", tags=["Overview"])
def get_level_anomaly_distribution():
    """Ground-truth normal/anomalous line counts by log level."""
    rows = query("""
        SELECT
            level,
            SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) AS anomalous,
            SUM(CASE WHEN is_anomaly = 0 THEN 1 ELSE 0 END) AS normal,
            COUNT(*) AS total,
            ROUND(AVG(is_anomaly) * 100, 4) AS anomaly_rate_pct
        FROM 'output/parsed.parquet'
        GROUP BY level
        ORDER BY anomalous DESC, total DESC
    """)
    return {"data": rows}


@app.get("/templates/top", tags=["Overview"])
def get_top_templates(limit: int = Query(15, ge=1, le=100)):
    """Most common mined log templates."""
    rows = query(f"""
        SELECT event_id, template, COUNT(*) AS count
        FROM 'output/parsed.parquet'
        GROUP BY event_id, template
        ORDER BY count DESC
        LIMIT {int(limit)}
    """)
    return {"data": rows}


@app.get("/scores", tags=["Anomalies"])
def get_scores(
    limit: int = Query(10000, ge=1, le=20000),
    offset: int = Query(0, ge=0),
):
    """Paginated anomaly score windows."""
    rows = query(f"""
        SELECT
            window_start,
            window_end,
            total_logs,
            anomaly_count,
            is_anomaly,
            ROUND(anomaly_score, 6) AS anomaly_score,
            predicted
        FROM 'output/scores.parquet'
        ORDER BY window_start
        LIMIT {int(limit)} OFFSET {int(offset)}
    """)
    total = scalar("SELECT COUNT(*) AS total FROM 'output/scores.parquet'")
    return {"total": total, "limit": limit, "offset": offset, "data": rows}


@app.get("/scores/timeline", tags=["Anomalies"])
def get_scores_timeline():
    """All scored windows, ordered for timeline charts."""
    rows = query("""
        SELECT
            window_start,
            window_end,
            total_logs,
            anomaly_count,
            is_anomaly,
            ROUND(anomaly_score, 6) AS anomaly_score,
            predicted
        FROM 'output/scores.parquet'
        ORDER BY window_start
    """)
    return {"data": rows}


@app.get("/scores/histogram", tags=["Anomalies"])
def get_score_histogram(bins: int = Query(80, ge=5, le=200)):
    """Histogram-ready anomaly score distribution."""
    rows = query(f"""
        WITH bounds AS (
            SELECT MIN(anomaly_score) AS min_score,
                   MAX(anomaly_score) AS max_score
            FROM 'output/scores.parquet'
        ),
        binned AS (
            SELECT
                CASE
                    WHEN max_score = min_score THEN 0
                    ELSE CAST(FLOOR(
                        (anomaly_score - min_score)
                        / NULLIF(max_score - min_score, 0)
                        * {int(bins)}
                    ) AS INTEGER)
                END AS bin_id,
                min_score,
                max_score
            FROM 'output/scores.parquet', bounds
        )
        SELECT
            bin_id,
            COUNT(*) AS count,
            ROUND(MIN(min_score + (max_score - min_score)
                * bin_id / {int(bins)}), 6) AS bin_start,
            ROUND(MIN(min_score + (max_score - min_score)
                * (bin_id + 1) / {int(bins)}), 6) AS bin_end
        FROM binned
        GROUP BY bin_id
        ORDER BY bin_id
    """)
    return {"data": rows}


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
    limit:     int           = Query(50, ge=1, le=5000),
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
        LIMIT {int(limit)} OFFSET {int(offset)}
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


@app.get("/alerts/minilm", tags=["Alerts"])
def get_minilm_alerts(limit: int = Query(5000, ge=1, le=10000)):
    """Alerts with both TF-IDF and MiniLM cluster assignments."""
    path = Path("output/alerts_minilm.parquet")
    if not path.exists():
        return {"error": "Run alerts_minilm.py first"}

    rows = query(f"""
        SELECT *
        FROM '{path}'
        ORDER BY anomaly_score DESC
        LIMIT {int(limit)}
    """)
    return {"total": len(rows), "data": rows}


@app.get("/alerts/minilm/clusters", tags=["Alerts"])
def get_minilm_clusters(
    method: str = Query("tfidf", pattern="^(tfidf|minilm)$")
):
    """Cluster summary for TF-IDF or MiniLM comparison, with content labels."""
    path = Path("output/alerts_minilm.parquet")
    if not path.exists():
        return {"error": "Run alerts_minilm.py first"}

    cid_col = "cluster_id_tfidf" if method == "tfidf" else "cluster_id_minilm"
    rows = query(f"""
        WITH cluster_stats AS (
            SELECT
                {cid_col} AS cluster_id,
                COUNT(*) AS alert_count,
                ROUND(MAX(anomaly_score), 4) AS max_score,
                SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END)
                    AS critical_count
            FROM '{path}'
            GROUP BY {cid_col}
        ),
        top_alert AS (
            SELECT *
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY {cid_col}
                        ORDER BY anomaly_score DESC
                    ) AS rn
                FROM '{path}'
            )
            WHERE rn = 1
        ),
        representative AS (
            SELECT *
            FROM (
                SELECT
                    a.{cid_col} AS cluster_id,
                    p.content AS representative_content,
                    p.level AS representative_level,
                    COUNT(*) AS content_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY a.{cid_col}
                        ORDER BY COUNT(*) DESC
                    ) AS rn
                FROM top_alert a
                JOIN 'output/parsed.parquet' p
                  ON p.timestamp >= a.window_start
                 AND p.timestamp <  a.window_end
                 AND p.template = a.top_template
                GROUP BY a.{cid_col}, p.content, p.level
            )
            WHERE rn = 1
        )
        SELECT
            s.*,
            COALESCE(r.representative_content, 'unknown')
                AS representative_content,
            COALESCE(r.representative_level, 'UNKNOWN')
                AS representative_level
        FROM cluster_stats s
        LEFT JOIN representative r USING (cluster_id)
        ORDER BY s.alert_count DESC
    """)
    return {"total_clusters": len(rows), "data": rows}


@app.get("/alerts/minilm/clusters/{cluster_id}", tags=["Alerts"])
def get_minilm_cluster_alerts(
    cluster_id: int,
    method: str = Query("tfidf", pattern="^(tfidf|minilm)$"),
):
    """Comparison cluster alert rows, enriched with concrete log content."""
    path = Path("output/alerts_minilm.parquet")
    if not path.exists():
        return {"error": "Run alerts_minilm.py first"}

    cid_col = "cluster_id_tfidf" if method == "tfidf" else "cluster_id_minilm"
    rows = query(f"""
        WITH selected AS (
            SELECT
                ROW_NUMBER() OVER (ORDER BY anomaly_score DESC) AS alert_id,
                window_start,
                window_end,
                ROUND(anomaly_score, 6) AS anomaly_score,
                severity,
                top_template,
                anomaly_count,
                total_logs
            FROM '{path}'
            WHERE {cid_col} = {int(cluster_id)}
        ),
        representative AS (
            SELECT *
            FROM (
                SELECT
                    a.alert_id,
                    p.content AS representative_content,
                    p.level AS representative_level,
                    COUNT(*) AS content_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY a.alert_id
                        ORDER BY COUNT(*) DESC
                    ) AS rn
                FROM selected a
                JOIN 'output/parsed.parquet' p
                  ON p.timestamp >= a.window_start
                 AND p.timestamp <  a.window_end
                 AND p.template = a.top_template
                GROUP BY a.alert_id, p.content, p.level
            )
            WHERE rn = 1
        )
        SELECT
            a.window_start,
            a.window_end,
            a.anomaly_score,
            a.severity,
            a.top_template,
            a.anomaly_count,
            a.total_logs,
            COALESCE(r.representative_content, a.top_template)
                AS representative_content,
            COALESCE(r.representative_level, 'UNKNOWN')
                AS representative_level
        FROM selected a
        LEFT JOIN representative r USING (alert_id)
        ORDER BY a.anomaly_score DESC
    """)
    return {"total": len(rows), "data": rows}


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
        WITH cluster_stats AS (
            SELECT
                cluster_id,
                cluster_label,
                COUNT(*) AS alert_count,
                ROUND(MAX(anomaly_score), 4) AS max_score,
                ROUND(AVG(anomaly_score), 4) AS avg_score,
                SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END)
                    AS critical_count
            FROM '{alerts_path}'
            GROUP BY cluster_id, cluster_label
        ),
        top_alert AS (
            SELECT *
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY cluster_id
                        ORDER BY anomaly_score DESC
                    ) AS rn
                FROM '{alerts_path}'
            )
            WHERE rn = 1
        ),
        representative AS (
            SELECT *
            FROM (
                SELECT
                    a.cluster_id,
                    p.content AS representative_content,
                    p.level AS representative_level,
                    COUNT(*) AS content_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY a.cluster_id
                        ORDER BY COUNT(*) DESC
                    ) AS rn
                FROM top_alert a
                JOIN 'output/parsed.parquet' p
                  ON p.timestamp >= a.window_start
                 AND p.timestamp <  a.window_end
                 AND p.is_anomaly = 1
                GROUP BY a.cluster_id, p.content, p.level
            )
            WHERE rn = 1
        )
        SELECT
            s.*,
            COALESCE(r.representative_content, s.cluster_label)
                AS representative_content,
            COALESCE(r.representative_level, 'UNKNOWN')
                AS representative_level
        FROM cluster_stats s
        LEFT JOIN representative r USING (cluster_id)
        ORDER BY s.alert_count DESC
    """)
    return {"total_clusters": len(rows), "data": rows}


@app.get("/clusters/{cluster_id}/alerts", tags=["Alerts"])
def get_cluster_alerts(cluster_id: int):
    """Alert rows for one cluster, enriched with representative log content."""
    alerts_path = "output/alerts.parquet"
    if not Path(alerts_path).exists():
        return {"error": "Run alerts.py first"}

    rows = query(f"""
        WITH selected AS (
            SELECT
                ROW_NUMBER() OVER (ORDER BY anomaly_score DESC) AS alert_id,
                window_start,
                window_end,
                ROUND(anomaly_score, 6) AS anomaly_score,
                severity,
                top_template,
                anomaly_count,
                total_logs
            FROM '{alerts_path}'
            WHERE cluster_id = {int(cluster_id)}
        ),
        representative AS (
            SELECT *
            FROM (
                SELECT
                    a.alert_id,
                    p.content AS representative_content,
                    p.level AS representative_level,
                    COUNT(*) AS content_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY a.alert_id
                        ORDER BY COUNT(*) DESC
                    ) AS rn
                FROM selected a
                JOIN 'output/parsed.parquet' p
                  ON p.timestamp >= a.window_start
                 AND p.timestamp <  a.window_end
                 AND p.template = a.top_template
                GROUP BY a.alert_id, p.content, p.level
            )
            WHERE rn = 1
        )
        SELECT
            a.window_start,
            a.window_end,
            a.anomaly_score,
            a.severity,
            a.top_template,
            a.anomaly_count,
            a.total_logs,
            COALESCE(r.representative_content, a.top_template)
                AS representative_content,
            COALESCE(r.representative_level, 'UNKNOWN')
                AS representative_level
        FROM selected a
        LEFT JOIN representative r USING (alert_id)
        ORDER BY a.anomaly_score DESC
    """)
    return {"total": len(rows), "data": rows}


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
    total_alerts = int(
        scalar("SELECT COUNT(*) AS total FROM 'output/alerts.parquet'") or 0
    )
    denom = max(total_alerts, 1)
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
                ) / denom) * 100, 1
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
                ) / denom) * 100, 1
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
