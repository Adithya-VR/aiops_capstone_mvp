from pathlib import Path
from typing import Optional
import json

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from dataset_config import (
    DEFAULT_DATASET,
    available_datasets,
    dataset_paths,
    get_dataset,
)


app = FastAPI(
    title="AIOps Dataset API",
    description=(
        "REST API for AIOps processed datasets. Provides dashboard-ready "
        "access to logs, anomaly scores, alerts, clusters, and metrics."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def query(sql: str) -> list:
    con = duckdb.connect()
    try:
        return con.execute(sql).df().to_dict(orient="records")
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


def paths_for(dataset: str) -> dict:
    try:
        paths = dataset_paths(dataset)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset}")
    return paths


def parquet(paths: dict, key: str) -> str:
    return str(paths[key]).replace("\\", "/")


def require_file(paths: dict, key: str, message: str) -> Optional[dict]:
    if not paths[key].exists():
        raise HTTPException(status_code=404, detail=message)
    return None


def representative_log(paths: dict, window_start, window_end, template=None):
    parsed = parquet(paths, "parsed")
    template_filter = ""
    if template:
        template_filter = f"AND template = {sql_string(str(template))}"

    rows = query(f"""
        SELECT content, level, COUNT(*) AS count
        FROM '{parsed}'
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
            FROM '{parsed}'
            WHERE timestamp >= {int(window_start)}
              AND timestamp <  {int(window_end)}
              AND level IN ('FATAL', 'SEVERE', 'ERROR', 'FAILURE')
            GROUP BY content, level
            ORDER BY count DESC
            LIMIT 1
        """)

    if not rows:
        return {"content": str(template or "unknown"), "level": "UNKNOWN"}
    return {"content": rows[0]["content"], "level": rows[0]["level"]}


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "AIOps Dataset API",
        "version": "1.1.0",
        "default_dataset": DEFAULT_DATASET,
        "docs": "/docs",
    }


@app.get("/datasets", tags=["Overview"])
def list_datasets():
    datasets = available_datasets()
    return {"datasets": datasets, "count": len(datasets)}


@app.get("/stats", tags=["Overview"])
@app.get("/datasets/{dataset}/stats", tags=["Overview"])
def get_stats(dataset: str = DEFAULT_DATASET):
    paths = paths_for(dataset)
    if err := require_file(paths, "parsed", "Run pipeline.py first"):
        return err
    if err := require_file(paths, "scores", "Run pipeline.py first"):
        return err

    parsed = parquet(paths, "parsed")
    scores = parquet(paths, "scores")

    result = query(f"""
        SELECT
            COUNT(*) AS total_logs,
            SUM(is_anomaly) AS anomalous_lines,
            ROUND(AVG(is_anomaly) * 100, 2) AS anomaly_rate_pct,
            COUNT(DISTINCT event_id) AS unique_templates
        FROM '{parsed}'
    """)[0]

    score_result = query(f"""
        SELECT
            COUNT(*) AS total_windows,
            SUM(predicted) AS anomalous_windows,
            ROUND(MIN(anomaly_score), 4) AS score_min,
            ROUND(MAX(anomaly_score), 4) AS score_max,
            ROUND(AVG(anomaly_score), 4) AS score_mean
        FROM '{scores}'
    """)[0]

    cfg = get_dataset(dataset)
    return {
        "dataset": dataset,
        "display_name": cfg["display_name"],
        "has_labels": cfg.get("has_labels", True),
        "evaluation_mode": cfg.get("evaluation_mode", "supervised"),
        **result,
        **score_result,
    }


@app.get("/logs", tags=["Logs"])
@app.get("/datasets/{dataset}/logs", tags=["Logs"])
def get_logs(
    dataset: str = DEFAULT_DATASET,
    level: Optional[str] = Query(None),
    anomaly_only: bool = Query(False),
    normal_only: bool = Query(False),
    predicted_only: bool = Query(False),
    non_predicted_only: bool = Query(False),
    search: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    paths = paths_for(dataset)
    if err := require_file(paths, "parsed", "Run pipeline.py first"):
        return err
    if predicted_only or non_predicted_only:
        if err := require_file(paths, "scores", "Run pipeline.py first"):
            return err
    parsed = parquet(paths, "parsed")
    scores = parquet(paths, "scores")

    where = ["1=1"]
    levels = csv_values(level)
    if levels:
        where.append(
            "level IN (" + ", ".join(sql_string(v) for v in levels) + ")"
        )
    if anomaly_only:
        where.append("is_anomaly = 1")
    if normal_only:
        where.append("is_anomaly = 0")
    if predicted_only:
        where.append(f"""
            EXISTS (
                SELECT 1
                FROM '{scores}' s
                WHERE s.predicted = 1
                  AND p.timestamp >= s.window_start
                  AND p.timestamp < s.window_end
            )
        """)
    if non_predicted_only:
        where.append(f"""
            NOT EXISTS (
                SELECT 1
                FROM '{scores}' s
                WHERE s.predicted = 1
                  AND p.timestamp >= s.window_start
                  AND p.timestamp < s.window_end
            )
        """)
    if search:
        safe = search.replace("'", "''")
        where.append(f"content ILIKE '%{safe}%'")

    where_sql = " AND ".join(where)
    rows = query(f"""
        SELECT line_id, timestamp, date, node, level,
               is_anomaly, template, content
        FROM '{parsed}' p
        WHERE {where_sql}
        ORDER BY line_id
        LIMIT {int(limit)} OFFSET {int(offset)}
    """)
    total = scalar(f"""
        SELECT COUNT(*) AS total
        FROM '{parsed}' p
        WHERE {where_sql}
    """)
    return {"total": total, "limit": limit, "offset": offset, "data": rows}


@app.get("/levels", tags=["Overview"])
@app.get("/datasets/{dataset}/levels", tags=["Overview"])
def get_levels(dataset: str = DEFAULT_DATASET):
    paths = paths_for(dataset)
    parsed = parquet(paths, "parsed")
    rows = query(f"""
        SELECT DISTINCT level
        FROM '{parsed}'
        ORDER BY level
    """)
    return {"data": [r["level"] for r in rows]}


@app.get("/levels/distribution", tags=["Overview"])
@app.get("/datasets/{dataset}/levels/distribution", tags=["Overview"])
def get_level_distribution(dataset: str = DEFAULT_DATASET):
    paths = paths_for(dataset)
    parsed = parquet(paths, "parsed")
    rows = query(f"""
        SELECT level, COUNT(*) AS count
        FROM '{parsed}'
        GROUP BY level
        ORDER BY count DESC
    """)
    return {"data": rows}


@app.get("/levels/anomaly-distribution", tags=["Overview"])
@app.get("/datasets/{dataset}/levels/anomaly-distribution", tags=["Overview"])
def get_level_anomaly_distribution(dataset: str = DEFAULT_DATASET):
    paths = paths_for(dataset)
    parsed = parquet(paths, "parsed")
    rows = query(f"""
        SELECT
            level,
            SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) AS anomalous,
            SUM(CASE WHEN is_anomaly = 0 THEN 1 ELSE 0 END) AS normal,
            COUNT(*) AS total,
            ROUND(AVG(is_anomaly) * 100, 4) AS anomaly_rate_pct
        FROM '{parsed}'
        GROUP BY level
        ORDER BY anomalous DESC, total DESC
    """)
    return {"data": rows}


@app.get("/levels/predicted-window-distribution", tags=["Overview"])
@app.get("/datasets/{dataset}/levels/predicted-window-distribution", tags=["Overview"])
def get_level_predicted_window_distribution(dataset: str = DEFAULT_DATASET):
    paths = paths_for(dataset)
    if err := require_file(paths, "parsed", "Run pipeline.py first"):
        return err
    if err := require_file(paths, "scores", "Run pipeline.py first"):
        return err
    parsed = parquet(paths, "parsed")
    scores = parquet(paths, "scores")
    rows = query(f"""
        SELECT p.level, COUNT(*) AS count
        FROM '{parsed}' p
        WHERE EXISTS (
            SELECT 1
            FROM '{scores}' s
            WHERE s.predicted = 1
              AND p.timestamp >= s.window_start
              AND p.timestamp < s.window_end
        )
        GROUP BY p.level
        ORDER BY count DESC
    """)
    total = int(sum(int(row["count"]) for row in rows))
    return {"total": total, "data": rows}


@app.get("/templates/top", tags=["Overview"])
@app.get("/datasets/{dataset}/templates/top", tags=["Overview"])
def get_top_templates(
    dataset: str = DEFAULT_DATASET,
    limit: int = Query(15, ge=1, le=100),
):
    paths = paths_for(dataset)
    parsed = parquet(paths, "parsed")
    rows = query(f"""
        SELECT event_id, template, COUNT(*) AS count
        FROM '{parsed}'
        GROUP BY event_id, template
        ORDER BY count DESC
        LIMIT {int(limit)}
    """)
    return {"data": rows}


@app.get("/scores", tags=["Anomalies"])
@app.get("/datasets/{dataset}/scores", tags=["Anomalies"])
def get_scores(
    dataset: str = DEFAULT_DATASET,
    limit: int = Query(10000, ge=1, le=20000),
    offset: int = Query(0, ge=0),
):
    paths = paths_for(dataset)
    scores = parquet(paths, "scores")
    rows = query(f"""
        SELECT window_start, window_end, total_logs, anomaly_count,
               is_anomaly, ROUND(anomaly_score, 6) AS anomaly_score,
               predicted
        FROM '{scores}'
        ORDER BY window_start
        LIMIT {int(limit)} OFFSET {int(offset)}
    """)
    total = scalar(f"SELECT COUNT(*) AS total FROM '{scores}'")
    return {"total": total, "limit": limit, "offset": offset, "data": rows}


@app.get("/scores/timeline", tags=["Anomalies"])
@app.get("/datasets/{dataset}/scores/timeline", tags=["Anomalies"])
def get_scores_timeline(dataset: str = DEFAULT_DATASET):
    paths = paths_for(dataset)
    scores = parquet(paths, "scores")
    rows = query(f"""
        SELECT window_start, window_end, total_logs, anomaly_count,
               is_anomaly, ROUND(anomaly_score, 6) AS anomaly_score,
               predicted
        FROM '{scores}'
        ORDER BY window_start
    """)
    return {"data": rows}


@app.get("/scores/histogram", tags=["Anomalies"])
@app.get("/datasets/{dataset}/scores/histogram", tags=["Anomalies"])
def get_score_histogram(
    dataset: str = DEFAULT_DATASET,
    bins: int = Query(80, ge=5, le=200),
):
    paths = paths_for(dataset)
    scores = parquet(paths, "scores")
    bins = int(bins)
    rows = query(f"""
        WITH bounds AS (
            SELECT MIN(anomaly_score) AS min_score,
                   MAX(anomaly_score) AS max_score
            FROM '{scores}'
        ),
        binned AS (
            SELECT
                CASE
                    WHEN max_score = min_score THEN 0
                    ELSE LEAST({bins - 1}, CAST(FLOOR(
                        (anomaly_score - min_score)
                        / NULLIF(max_score - min_score, 0)
                        * {bins}
                    ) AS INTEGER))
                END AS bin_id,
                min_score,
                max_score
            FROM '{scores}', bounds
        )
        SELECT
            bin_id,
            COUNT(*) AS count,
            ROUND(MIN(min_score + (max_score - min_score)
                * bin_id / {bins}), 6) AS bin_start,
            ROUND(MIN(min_score + (max_score - min_score)
                * (bin_id + 1) / {bins}), 6) AS bin_end
        FROM binned
        GROUP BY bin_id
        ORDER BY bin_id
    """)
    return {"data": rows}


@app.get("/anomalies", tags=["Anomalies"])
@app.get("/datasets/{dataset}/anomalies", tags=["Anomalies"])
def get_anomalies(
    dataset: str = DEFAULT_DATASET,
    min_score: float = Query(0.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    paths = paths_for(dataset)
    scores = parquet(paths, "scores")
    rows = query(f"""
        SELECT window_start, window_end,
               ROUND(anomaly_score, 4) AS anomaly_score,
               predicted, is_anomaly AS ground_truth,
               total_logs, anomaly_count
        FROM '{scores}'
        WHERE predicted = 1
          AND anomaly_score >= {float(min_score)}
        ORDER BY anomaly_score DESC
        LIMIT {int(limit)} OFFSET {int(offset)}
    """)
    return {"total": len(rows), "data": rows}


@app.get("/alerts", tags=["Alerts"])
@app.get("/datasets/{dataset}/alerts", tags=["Alerts"])
def get_alerts(
    dataset: str = DEFAULT_DATASET,
    severity: Optional[str] = Query(None),
    min_score: float = Query(0.0),
    limit: int = Query(50, ge=1, le=5000),
    offset: int = Query(0, ge=0),
):
    paths = paths_for(dataset)
    if err := require_file(paths, "alerts", "Run alerts.py first"):
        return err
    alerts = parquet(paths, "alerts")

    where = [f"anomaly_score >= {float(min_score)}"]
    if severity:
        where.append(f"severity = {sql_string(severity.upper())}")
    where_sql = " AND ".join(where)

    rows = query(f"""
        SELECT window_start, window_end,
               ROUND(anomaly_score, 4) AS anomaly_score,
               severity, top_template, cluster_id, cluster_label,
               anomaly_count, total_logs
        FROM '{alerts}'
        WHERE {where_sql}
        ORDER BY anomaly_score DESC
        LIMIT {int(limit)} OFFSET {int(offset)}
    """)
    total = scalar(f"""
        SELECT COUNT(*) AS total
        FROM '{alerts}'
        WHERE {where_sql}
    """)
    return {"total": total, "data": rows}


@app.get("/alerts/summary", tags=["Alerts"])
@app.get("/datasets/{dataset}/alerts/summary", tags=["Alerts"])
def get_alert_summary(dataset: str = DEFAULT_DATASET):
    paths = paths_for(dataset)
    if err := require_file(paths, "alerts", "Run alerts.py first"):
        return err
    alerts = parquet(paths, "alerts")

    sev = query(f"""
        SELECT
            COUNT(*) AS total_alerts,
            SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) AS critical,
            SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END) AS high,
            SUM(CASE WHEN severity = 'MEDIUM' THEN 1 ELSE 0 END) AS medium,
            SUM(CASE WHEN severity = 'LOW' THEN 1 ELSE 0 END) AS low,
            COUNT(DISTINCT CASE WHEN cluster_id >= 0 THEN cluster_id END)
                AS clusters,
            SUM(CASE WHEN cluster_id < 0 THEN 1 ELSE 0 END)
                AS unique_alerts
        FROM '{alerts}'
    """)[0]

    total = sev["total_alerts"]
    groups = sev["clusters"] + sev["unique_alerts"]
    reduction = round((1 - groups / total) * 100, 1) if total > 0 else 0
    return {**sev, "distinct_groups": groups, "noise_reduction_pct": reduction}


@app.get("/alerts/minilm", tags=["Alerts"])
@app.get("/datasets/{dataset}/alerts/minilm", tags=["Alerts"])
def get_minilm_alerts(
    dataset: str = DEFAULT_DATASET,
    limit: int = Query(5000, ge=1, le=10000),
):
    paths = paths_for(dataset)
    if err := require_file(paths, "alerts_minilm", "Run alerts_minilm.py first"):
        return err
    path = parquet(paths, "alerts_minilm")
    rows = query(f"""
        SELECT *
        FROM '{path}'
        ORDER BY anomaly_score DESC
        LIMIT {int(limit)}
    """)
    return {"total": len(rows), "data": rows}


@app.get("/alerts/minilm/clusters", tags=["Alerts"])
@app.get("/datasets/{dataset}/alerts/minilm/clusters", tags=["Alerts"])
def get_minilm_clusters(
    dataset: str = DEFAULT_DATASET,
    method: str = Query("tfidf", pattern="^(tfidf|minilm)$"),
):
    paths = paths_for(dataset)
    if err := require_file(paths, "alerts_minilm", "Run alerts_minilm.py first"):
        return err
    path = parquet(paths, "alerts_minilm")
    parsed = parquet(paths, "parsed")
    cid_col = "cluster_id_tfidf" if method == "tfidf" else "cluster_id_minilm"

    rows = query(f"""
        WITH cluster_stats AS (
            SELECT {cid_col} AS cluster_id,
                   COUNT(*) AS alert_count,
                   ROUND(MAX(anomaly_score), 4) AS max_score,
                   SUM(CASE WHEN severity = 'LOW' THEN 1 ELSE 0 END)
                       AS low_count,
                   SUM(CASE WHEN severity = 'MEDIUM' THEN 1 ELSE 0 END)
                       AS medium_count,
                   SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END)
                       AS high_count,
                   SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END)
                       AS critical_count
            FROM '{path}'
            GROUP BY {cid_col}
        ),
        top_alert AS (
            SELECT *
            FROM (
                SELECT *, ROW_NUMBER() OVER (
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
                SELECT a.{cid_col} AS cluster_id,
                       p.content AS representative_content,
                       p.level AS representative_level,
                       ROW_NUMBER() OVER (
                           PARTITION BY a.{cid_col}
                           ORDER BY COUNT(*) DESC
                       ) AS rn
                FROM top_alert a
                JOIN '{parsed}' p
                  ON p.timestamp >= a.window_start
                 AND p.timestamp <  a.window_end
                 AND p.template = a.top_template
                GROUP BY a.{cid_col}, p.content, p.level
            )
            WHERE rn = 1
        )
        SELECT s.*,
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
@app.get("/datasets/{dataset}/alerts/minilm/clusters/{cluster_id}", tags=["Alerts"])
def get_minilm_cluster_alerts(
    cluster_id: int,
    dataset: str = DEFAULT_DATASET,
    method: str = Query("tfidf", pattern="^(tfidf|minilm)$"),
):
    paths = paths_for(dataset)
    if err := require_file(paths, "alerts_minilm", "Run alerts_minilm.py first"):
        return err
    path = parquet(paths, "alerts_minilm")
    parsed = parquet(paths, "parsed")
    cid_col = "cluster_id_tfidf" if method == "tfidf" else "cluster_id_minilm"

    rows = query(f"""
        WITH selected AS (
            SELECT ROW_NUMBER() OVER (ORDER BY anomaly_score DESC) AS alert_id,
                   window_start, window_end,
                   ROUND(anomaly_score, 6) AS anomaly_score,
                   severity, top_template, anomaly_count, total_logs
            FROM '{path}'
            WHERE {cid_col} = {int(cluster_id)}
        ),
        representative AS (
            SELECT *
            FROM (
                SELECT a.alert_id,
                       p.content AS representative_content,
                       p.level AS representative_level,
                       ROW_NUMBER() OVER (
                           PARTITION BY a.alert_id
                           ORDER BY COUNT(*) DESC
                       ) AS rn
                FROM selected a
                JOIN '{parsed}' p
                  ON p.timestamp >= a.window_start
                 AND p.timestamp <  a.window_end
                 AND p.template = a.top_template
                GROUP BY a.alert_id, p.content, p.level
            )
            WHERE rn = 1
        )
        SELECT a.window_start, a.window_end, a.anomaly_score,
               a.severity, a.top_template, a.anomaly_count, a.total_logs,
               COALESCE(r.representative_content, a.top_template)
                   AS representative_content,
               COALESCE(r.representative_level, 'UNKNOWN')
                   AS representative_level
        FROM selected a
        LEFT JOIN representative r USING (alert_id)
        ORDER BY a.anomaly_score DESC
    """)
    return {"total": len(rows), "data": rows}


@app.get("/clusters", tags=["Alerts"])
@app.get("/datasets/{dataset}/clusters", tags=["Alerts"])
def get_clusters(dataset: str = DEFAULT_DATASET):
    paths = paths_for(dataset)
    if err := require_file(paths, "alerts", "Run alerts.py first"):
        return err
    alerts = parquet(paths, "alerts")
    parsed = parquet(paths, "parsed")
    label_clause = "AND p.is_anomaly = 1" if get_dataset(dataset).get("has_labels", True) else ""

    rows = query(f"""
        WITH cluster_stats AS (
            SELECT cluster_id, cluster_label,
                   COUNT(*) AS alert_count,
                   ROUND(MAX(anomaly_score), 4) AS max_score,
                   ROUND(AVG(anomaly_score), 4) AS avg_score,
                   SUM(CASE WHEN severity = 'LOW' THEN 1 ELSE 0 END)
                       AS low_count,
                   SUM(CASE WHEN severity = 'MEDIUM' THEN 1 ELSE 0 END)
                       AS medium_count,
                   SUM(CASE WHEN severity = 'HIGH' THEN 1 ELSE 0 END)
                       AS high_count,
                   SUM(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END)
                       AS critical_count
            FROM '{alerts}'
            GROUP BY cluster_id, cluster_label
        ),
        top_alert AS (
            SELECT *
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY cluster_id
                    ORDER BY anomaly_score DESC
                ) AS rn
                FROM '{alerts}'
            )
            WHERE rn = 1
        ),
        representative AS (
            SELECT *
            FROM (
                SELECT a.cluster_id,
                       p.content AS representative_content,
                       p.level AS representative_level,
                       ROW_NUMBER() OVER (
                           PARTITION BY a.cluster_id
                           ORDER BY COUNT(*) DESC
                       ) AS rn
                FROM top_alert a
                JOIN '{parsed}' p
                  ON p.timestamp >= a.window_start
                 AND p.timestamp <  a.window_end
                 AND p.template = a.top_template
                 {label_clause}
                GROUP BY a.cluster_id, p.content, p.level
            )
            WHERE rn = 1
        )
        SELECT s.*,
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
@app.get("/datasets/{dataset}/clusters/{cluster_id}/alerts", tags=["Alerts"])
def get_cluster_alerts(
    cluster_id: int,
    dataset: str = DEFAULT_DATASET,
):
    paths = paths_for(dataset)
    if err := require_file(paths, "alerts", "Run alerts.py first"):
        return err
    alerts = parquet(paths, "alerts")
    parsed = parquet(paths, "parsed")

    rows = query(f"""
        WITH selected AS (
            SELECT ROW_NUMBER() OVER (ORDER BY anomaly_score DESC) AS alert_id,
                   window_start, window_end,
                   ROUND(anomaly_score, 6) AS anomaly_score,
                   severity, top_template, anomaly_count, total_logs
            FROM '{alerts}'
            WHERE cluster_id = {int(cluster_id)}
        ),
        representative AS (
            SELECT *
            FROM (
                SELECT a.alert_id,
                       p.content AS representative_content,
                       p.level AS representative_level,
                       ROW_NUMBER() OVER (
                           PARTITION BY a.alert_id
                           ORDER BY COUNT(*) DESC
                       ) AS rn
                FROM selected a
                JOIN '{parsed}' p
                  ON p.timestamp >= a.window_start
                 AND p.timestamp <  a.window_end
                 AND p.template = a.top_template
                GROUP BY a.alert_id, p.content, p.level
            )
            WHERE rn = 1
        )
        SELECT a.window_start, a.window_end, a.anomaly_score,
               a.severity, a.top_template, a.anomaly_count, a.total_logs,
               COALESCE(r.representative_content, a.top_template)
                   AS representative_content,
               COALESCE(r.representative_level, 'UNKNOWN')
                   AS representative_level
        FROM selected a
        LEFT JOIN representative r USING (alert_id)
        ORDER BY a.anomaly_score DESC
    """)
    return {"total": len(rows), "data": rows}


@app.get("/metrics", tags=["Evaluation"])
@app.get("/datasets/{dataset}/metrics", tags=["Evaluation"])
def get_metrics(dataset: str = DEFAULT_DATASET):
    paths = paths_for(dataset)
    if not paths["metrics"].exists():
        return {"error": "Run pipeline.py to generate metrics.json"}
    return json.loads(paths["metrics"].read_text(encoding="utf-8"))


@app.get("/clustering/comparison", tags=["Evaluation"])
@app.get("/datasets/{dataset}/clustering/comparison", tags=["Evaluation"])
def get_clustering_comparison(dataset: str = DEFAULT_DATASET):
    paths = paths_for(dataset)
    if not paths["clustering_comparison"].exists():
        return {"error": "Run alerts_minilm.py to generate comparison"}

    data = json.loads(
        paths["clustering_comparison"].read_text(encoding="utf-8")
    )
    alerts = parquet(paths, "alerts")
    total_alerts = int(scalar(f"SELECT COUNT(*) AS total FROM '{alerts}'") or 0)
    denom = max(total_alerts, 1)
    return {
        "tfidf": {
            "method": "TF-IDF + DBSCAN",
            "eps": 0.5,
            "clusters": data.get("tfidf_clusters"),
            "unique": data.get("tfidf_unique"),
            "silhouette": data.get("tfidf_silhouette"),
            "noise_reduction": round(
                (1 - (
                    data.get("tfidf_clusters", 0)
                    + data.get("tfidf_unique", 0)
                ) / denom) * 100,
                1,
            ),
        },
        "minilm": {
            "method": "MiniLM + DBSCAN",
            "eps": 0.4,
            "clusters": data.get("minilm_clusters"),
            "unique": data.get("minilm_unique"),
            "silhouette": data.get("minilm_silhouette"),
            "noise_reduction": round(
                (1 - (
                    data.get("minilm_clusters", 0)
                    + data.get("minilm_unique", 0)
                ) / denom) * 100,
                1,
            ),
        },
    }


@app.get("/logs/window", tags=["Logs"])
@app.get("/datasets/{dataset}/logs/window", tags=["Logs"])
def get_window_logs(
    dataset: str = DEFAULT_DATASET,
    window_start: int = Query(...),
    window_end: int = Query(...),
    limit: int = Query(50, ge=1, le=200),
):
    paths = paths_for(dataset)
    parsed = parquet(paths, "parsed")
    rows = query(f"""
        SELECT node, level, is_anomaly, template, content, timestamp
        FROM '{parsed}'
        WHERE timestamp >= {int(window_start)}
          AND timestamp <  {int(window_end)}
        ORDER BY timestamp
        LIMIT {int(limit)}
    """)
    return {
        "window_start": window_start,
        "window_end": window_end,
        "count": len(rows),
        "data": rows,
    }
