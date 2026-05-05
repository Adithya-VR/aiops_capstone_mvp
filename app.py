import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


API_BASE = os.getenv("AIOPS_API_BASE", "http://127.0.0.1:8000")
PAGE_SIZE = 500


def unix_to_readable(ts):
    try:
        return datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


@st.cache_data(ttl=30)
def api_get(path, params=None):
    url = f"{API_BASE.rstrip('/')}{path}"
    response = requests.get(url, params=params or {}, timeout=60)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"])
    return data


def api_frame(path, params=None, key="data"):
    return pd.DataFrame(api_get(path, params).get(key, []))


def dataset_path(dataset, path):
    return f"/datasets/{dataset}{path}"


def dataset_get(dataset, path, params=None):
    return api_get(dataset_path(dataset, path), params)


def dataset_frame(dataset, path, params=None, key="data"):
    return pd.DataFrame(dataset_get(dataset, path, params).get(key, []))


def load_required_data(dataset):
    try:
        stats = dataset_get(dataset, "/stats")
        levels = dataset_get(dataset, "/levels")["data"]
        scores = dataset_frame(dataset, "/scores/timeline")
        alerts = dataset_frame(dataset, "/alerts", {"limit": 5000})
        alert_summary = dataset_get(dataset, "/alerts/summary")
        return stats, levels, scores, alerts, alert_summary
    except Exception as exc:
        st.error(f"Could not load data from FastAPI: {exc}")
        st.info(
            "Start the API first: "
            "`uvicorn api.main:app --host 127.0.0.1 --port 8000`"
        )
        st.stop()


st.set_page_config(
    page_title="AIOps Anomaly Detection",
    page_icon="🔍",
    layout="wide",
)

try:
    dataset_payload = api_get("/datasets")
    dataset_rows = dataset_payload.get("datasets", [])
except Exception as exc:
    st.error(f"Could not reach FastAPI: {exc}")
    st.info(
        "Start the API first: "
        "`uvicorn api.main:app --host 127.0.0.1 --port 8000`"
    )
    st.stop()

ready_datasets = [d for d in dataset_rows if d.get("status") == "ready"]
if not ready_datasets:
    st.error("No processed datasets are ready.")
    st.info("Run the pipeline for at least one dataset first.")
    st.stop()

dataset_labels = {
    f"{d['display_name']} ({d['name']})": d["name"]
    for d in ready_datasets
}
selected_label = st.sidebar.selectbox(
    "Dataset",
    options=list(dataset_labels.keys()),
)
DATASET = dataset_labels[selected_label]
dataset_meta = next(d for d in ready_datasets if d["name"] == DATASET)
HAS_LABELS = bool(dataset_meta.get("has_labels", True))

stats, levels, scores, alerts, alert_summary = load_required_data(DATASET)

score_min = float(stats.get("score_min", scores["anomaly_score"].min()))
score_max = float(stats.get("score_max", scores["anomaly_score"].max()))
score_p90 = float(scores["anomaly_score"].quantile(0.90))

st.sidebar.title("🔍 AIOps Dashboard")
st.sidebar.caption(f"Dataset: {dataset_meta['display_name']}")
st.sidebar.caption(f"API: {API_BASE}")
st.sidebar.divider()
st.sidebar.metric("Total Log Lines", f"{int(stats['total_logs']):,}")
st.sidebar.metric("Unique Templates", f"{int(stats['unique_templates']):,}")
st.sidebar.metric(
    "Anomalous Windows",
    f"{int(stats.get('anomalous_windows', scores['predicted'].sum())):,}",
)
st.sidebar.metric(
    "Prediction Rate",
    f"{float(scores['predicted'].mean()):.1%}",
)
st.sidebar.divider()

thresh = st.sidebar.slider(
    "Anomaly Score Threshold",
    min_value=score_min,
    max_value=score_max,
    value=score_p90,
    step=0.01,
    help="Windows above this score are flagged in the timeline",
)
top_n = st.sidebar.slider("Top N Alerts", 5, 50, 20)

t1, t2, t3, t4, t5 = st.tabs(
    [
        "📊 Overview",
        "📋 Log Explorer",
        "📈 Anomaly Timeline",
        "🚨 Top Alerts",
        "🔔 Alert Clusters",
    ]
)

with t1:
    st.header("System Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Log Lines", f"{int(stats['total_logs']):,}")
    c2.metric("Unique Templates", f"{int(stats['unique_templates']):,}")
    if HAS_LABELS:
        c3.metric("Anomalous Lines", f"{int(stats['anomalous_lines']):,}")
        c4.metric("Line Anomaly Rate", f"{stats['anomaly_rate_pct']:.2f}%")
    else:
        c3.metric("Labeled Anomalous Lines", "N/A")
        c4.metric(
            "Predicted Alert Windows",
            f"{int(stats.get('anomalous_windows', scores['predicted'].sum())):,}",
        )

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Log Level Distribution")
        level_counts = dataset_frame(DATASET, "/levels/distribution")
        fig = px.pie(
            level_counts,
            names="level",
            values="count",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 15 Log Templates")
        top_templates = dataset_frame(DATASET, "/templates/top", {"limit": 15})
        top_templates["short"] = top_templates["template"].str[:55]
        fig = px.bar(
            top_templates,
            x="count",
            y="short",
            orientation="h",
            color="count",
            color_continuous_scale="Teal",
            labels={"short": "Template", "count": "Count"},
        )
        fig.update_layout(yaxis_title="", height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Anomaly Score Distribution")
    hist = dataset_frame(DATASET, "/scores/histogram", {"bins": 80})
    hist["bin_mid"] = (hist["bin_start"] + hist["bin_end"]) / 2
    fig = px.bar(
        hist,
        x="bin_mid",
        y="count",
        labels={"bin_mid": "Anomaly Score", "count": "Windows"},
        color_discrete_sequence=["#7F77DD"],
    )
    fig.add_vline(
        x=thresh,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {thresh:.2f}",
        annotation_position="top right",
    )
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.header("Log Explorer")

    c1, c2, c3 = st.columns(3)
    show_options = (
        ["All", "Anomalies only", "Normal only"]
        if HAS_LABELS
        else ["All", "Predicted alert-window logs", "Non-alert-window logs"]
    )
    f_show = c1.selectbox("Show", show_options)
    f_level = c2.multiselect("Log Level", options=levels, default=levels)
    f_search = c3.text_input("Search in content", "")

    filter_key = f"{f_show}_{','.join(f_level)}_{f_search}"
    if "last_log_filter" not in st.session_state:
        st.session_state.last_log_filter = filter_key
    if "log_page" not in st.session_state:
        st.session_state.log_page = 1
    if st.session_state.last_log_filter != filter_key:
        st.session_state.log_page = 1
        st.session_state.last_log_filter = filter_key

    params = {
        "limit": PAGE_SIZE,
        "offset": (st.session_state.log_page - 1) * PAGE_SIZE,
        "level": ",".join(f_level),
        "search": f_search or None,
        "anomaly_only": HAS_LABELS and f_show == "Anomalies only",
        "normal_only": HAS_LABELS and f_show == "Normal only",
        "predicted_only": (not HAS_LABELS)
        and f_show == "Predicted alert-window logs",
        "non_predicted_only": (not HAS_LABELS)
        and f_show == "Non-alert-window logs",
    }
    payload = dataset_get(DATASET, "/logs", params)
    total_rows = int(payload["total"])
    page_view = pd.DataFrame(payload["data"])
    total_pages = max(1, (total_rows + PAGE_SIZE - 1) // PAGE_SIZE)

    st.caption(f"**{total_rows:,} rows** match your filters")
    col_info, col_prev, col_page, col_next = st.columns([3, 1, 2, 1])
    with col_info:
        st.write(f"Page size: {PAGE_SIZE} rows per page")
    with col_prev:
        if st.button("◀ Prev", disabled=st.session_state.log_page <= 1):
            st.session_state.log_page -= 1
            st.rerun()
    with col_page:
        st.write(f"**Page {st.session_state.log_page} of {total_pages}**")
    with col_next:
        if st.button("Next ▶", disabled=st.session_state.log_page >= total_pages):
            st.session_state.log_page += 1
            st.rerun()

    jump = st.number_input(
        "Jump to page",
        min_value=1,
        max_value=total_pages,
        value=st.session_state.log_page,
        step=1,
        key="page_jump",
    )
    if jump != st.session_state.log_page:
        st.session_state.log_page = int(jump)
        st.rerun()

    start_row = (st.session_state.log_page - 1) * PAGE_SIZE + 1
    end_row = min(start_row + PAGE_SIZE - 1, total_rows)
    st.caption(f"Showing rows **{start_row:,} - {end_row:,}** of **{total_rows:,}**")

    def highlight(row):
        color = "background-color: #ffcccc" if row["is_anomaly"] else ""
        return [color] * len(row)

    display_cols = [
        "line_id",
        "date",
        "node",
        "level",
        "is_anomaly",
        "template",
        "content",
    ]
    if not page_view.empty:
        st.dataframe(
            page_view[display_cols].style.apply(highlight, axis=1),
            use_container_width=True,
            height=450,
        )
    else:
        st.info("No logs match the current filters.")

    st.divider()
    if HAS_LABELS:
        st.subheader("Anomaly Labels by Log Level")
        level_anomalies = dataset_frame(DATASET, "/levels/anomaly-distribution")
        st.dataframe(level_anomalies, use_container_width=True, hide_index=True)
        st.caption(
            "These are the dataset ground-truth labels by log level. "
            "In the current BGL labels, anomalous lines appear in FATAL and "
            "a small number of FAILURE rows; the other levels are labeled normal."
        )
    else:
        st.subheader("Predicted Alert-Window Logs by Level")
        alert_level_rows = dataset_frame(
            DATASET,
            "/levels/predicted-window-distribution",
        )
        if not alert_level_rows.empty:
            alert_level_rows = alert_level_rows.rename(
                columns={"count": "log_count"}
            )
            st.dataframe(
                alert_level_rows,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No predicted alert-window logs found.")

with t3:
    st.header("Anomaly Timeline")

    timeline = scores.copy()
    timeline["window_dt"] = timeline["window_start"].apply(unix_to_readable)
    flagged = timeline[timeline["anomaly_score"] >= thresh]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline["window_dt"],
            y=timeline["anomaly_score"],
            mode="lines",
            name="Anomaly Score",
            line=dict(color="#7F77DD", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=flagged["window_dt"],
            y=flagged["anomaly_score"],
            mode="markers",
            name="Flagged",
            marker=dict(color="red", size=5, symbol="circle"),
        )
    )
    fig.add_hline(
        y=thresh,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {thresh:.2f}",
    )
    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Anomaly Score",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score vs Log Volume")
        point_color = (
            timeline["is_anomaly"].map({0: "Normal", 1: "Anomaly"})
            if HAS_LABELS
            else timeline["predicted"].map({0: "Not Flagged", 1: "Flagged"})
        )
        color_map = (
            {"Normal": "#1D9E75", "Anomaly": "#E24B4A"}
            if HAS_LABELS
            else {"Not Flagged": "#1D9E75", "Flagged": "#E24B4A"}
        )
        fig = px.scatter(
            timeline,
            x="total_logs",
            y="anomaly_score",
            color=point_color,
            color_discrete_map=color_map,
            opacity=0.6,
            labels={
                "total_logs": "Logs in Window",
                "anomaly_score": "Anomaly Score",
                "color": "Ground Truth" if HAS_LABELS else "Prediction",
            },
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if HAS_LABELS:
            st.subheader("Confusion Matrix")
            cm = pd.crosstab(
                timeline["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
                timeline["predicted"].map({0: "Normal", 1: "Anomaly"}),
                rownames=["Actual"],
                colnames=["Predicted"],
            )
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues")
        else:
            st.subheader("Prediction Counts")
            pred_counts = (
                timeline["predicted"]
                .map({0: "Not Flagged", 1: "Flagged"})
                .value_counts()
                .reset_index()
            )
            pred_counts.columns = ["prediction", "windows"]
            fig = px.bar(
                pred_counts,
                x="prediction",
                y="windows",
                color="prediction",
                color_discrete_map=color_map,
            )
        st.plotly_chart(fig, use_container_width=True)

with t4:
    st.header("Top Anomalous Windows" if HAS_LABELS else "Top Predicted Alert Windows")

    anomalous_scores = scores[scores["predicted"] == 1]["anomaly_score"]
    p95 = float(anomalous_scores.quantile(0.95))
    p85 = float(anomalous_scores.quantile(0.85))
    p70 = float(anomalous_scores.quantile(0.70))
    top = scores.nlargest(top_n, "anomaly_score")

    for _, row in top.iterrows():
        score = row["anomaly_score"]
        if score >= p95:
            sev = "🔴 CRITICAL"
        elif score >= p85:
            sev = "🟠 HIGH"
        elif score >= p70:
            sev = "🟡 MEDIUM"
        else:
            sev = "🟢 LOW"

        evidence_label = (
            f"Anomalous lines: {int(row['anomaly_count'])}"
            if HAS_LABELS
            else f"Logs in window: {int(row['total_logs'])}"
        )
        label = (
            f"{sev} | Score: {score:.3f} | "
            f"{evidence_label} | {unix_to_readable(row['window_start'])}"
        )

        with st.expander(label):
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomaly Score", f"{score:.3f}")
            c2.metric("Total Logs", f"{int(row['total_logs'])}")
            if HAS_LABELS:
                c3.metric("Anomalous Lines", f"{int(row['anomaly_count'])}")
            else:
                c3.metric("Ground Truth", "N/A")

            logs = dataset_frame(
                DATASET,
                "/logs/window",
                {
                    "window_start": int(row["window_start"]),
                    "window_end": int(row["window_end"]),
                    "limit": 50,
                },
            )
            if not logs.empty:
                st.caption(f"{len(logs)} log lines returned for this window")
                st.dataframe(
                    logs[["node", "level", "is_anomaly", "template", "content"]],
                    use_container_width=True,
                )
            else:
                st.caption("No log lines found for this window.")

with t5:
    st.header("Alert Clusters")

    if alerts.empty:
        st.warning("Run `python alerts.py` first to generate clusters.")
        st.stop()

    try:
        comp = dataset_get(DATASET, "/clustering/comparison")
        minilm_clusters = dataset_frame(
            DATASET,
            "/alerts/minilm/clusters",
            {"method": "minilm"},
        )
        cluster_method = "MiniLM + DBSCAN"
        n_clusters = int(comp["minilm"]["clusters"])
        n_unique = int(comp["minilm"]["unique"])
        noise_reduction = float(comp["minilm"]["noise_reduction"])
    except Exception:
        comp = None
        minilm_clusters = pd.DataFrame()
        cluster_method = "TF-IDF + DBSCAN"
        n_clusters = int(alert_summary["clusters"])
        n_unique = int(alert_summary["unique_alerts"])
        noise_reduction = float(alert_summary["noise_reduction_pct"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alerts", f"{int(alert_summary['total_alerts']):,}")
    c2.metric("Clusters Found", f"{n_clusters}")
    c3.metric("Unique Alerts", f"{n_unique}")
    c4.metric("Noise Reduced", f"{noise_reduction:.1f}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Alerts by Severity")
        sev = alerts["severity"].value_counts().reset_index()
        sev.columns = ["severity", "count"]
        fig = px.bar(
            sev,
            x="severity",
            y="count",
            color="severity",
            color_discrete_map={
                "CRITICAL": "#E24B4A",
                "HIGH": "#EF9F27",
                "MEDIUM": "#EDD94C",
                "LOW": "#4CAF50",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Clusters by Size")
        if not minilm_clusters.empty:
            cluster_summary = minilm_clusters
        else:
            cluster_summary = dataset_frame(DATASET, "/clusters")
        top_clusters = cluster_summary.head(10).copy()
        if "cluster_label" not in top_clusters.columns:
            top_clusters["cluster_label"] = top_clusters["representative_content"]
        top_clusters["label"] = top_clusters["representative_content"].str[:70]
        fig = px.bar(
            top_clusters,
            x="alert_count",
            y="label",
            orientation="h",
            color="max_score",
            color_continuous_scale="Reds",
            hover_data=[
                "representative_level",
                "critical_count",
                "max_score",
                "cluster_label",
            ],
            labels={"alert_count": "Alert Count", "label": "Representative Error"},
        )
        fig.update_layout(yaxis_title="", height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader(f"All Alert Clusters - {cluster_method}")
    if not minilm_clusters.empty:
        cluster_summary = minilm_clusters
    else:
        cluster_summary = dataset_frame(DATASET, "/clusters")
    for _, cluster in cluster_summary.sort_values("cluster_id").iterrows():
        cid = int(cluster["cluster_id"])
        label = str(cluster["representative_content"])
        worst = float(cluster["max_score"])
        alert_count = int(cluster["alert_count"])
        critical_count = int(cluster["critical_count"])

        # icon = "🔴" if critical_count > 0 else "🟠"
        cluster_alerts = (
            pd.DataFrame({"severity": []})
            if not minilm_clusters.empty
            else alerts[alerts["cluster_id"] == cid]
        )
        sev_counts = cluster_alerts["severity"].value_counts().to_dict()
        if not minilm_clusters.empty and critical_count > 0:
            sev_counts["CRITICAL"] = critical_count
        if sev_counts.get("CRITICAL", 0) > 0:
            icon = "🔴"
        elif sev_counts.get("HIGH", 0) > 0:
            icon = "🟠"
        elif sev_counts.get("MEDIUM", 0) > 0:
            icon = "🟡"
        else:
            icon = "🟢"  #Low
        title = (
            f"{icon} {'Cluster ' + str(cid) if cid >= 0 else 'Unique'} | "
            f"{alert_count} alerts | Max score: {worst:.3f} | {label[:70]}"
        )

        with st.expander(title):
            c1, c2, c3 = st.columns(3)
            c1.metric("Alerts in cluster", alert_count)
            c2.metric("Max anomaly score", f"{worst:.3f}")
            c3.metric("Critical alerts", critical_count)

            if not minilm_clusters.empty:
                display = dataset_frame(
                    DATASET,
                    f"/alerts/minilm/clusters/{cid}",
                    {"method": "minilm"},
                )
            else:
                display = dataset_frame(DATASET, f"/clusters/{cid}/alerts")
            display["Time (UTC)"] = display["window_start"].apply(unix_to_readable)
            display = display.rename(
                columns={
                    "severity": "Severity",
                    "anomaly_score": "Score",
                    "anomaly_count": "Anomaly Lines",
                    "total_logs": "Total Logs",
                    "representative_level": "Level",
                    "representative_content": "Error Content",
                }
            )
            display["Score"] = display["Score"].round(3)
            cluster_columns = [
                "Time (UTC)",
                "Severity",
                "Score",
                "Total Logs",
                "Level",
                "Error Content",
            ]
            if HAS_LABELS:
                cluster_columns.insert(3, "Anomaly Lines")
            st.dataframe(
                display[cluster_columns],
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("🔬 Clustering Method Comparison")
    if comp:
        comparison = pd.DataFrame(
            [
                {
                    "Method": comp["tfidf"]["method"],
                    "Clusters Found": comp["tfidf"]["clusters"],
                    "Unique Alerts": comp["tfidf"]["unique"],
                    "Noise Reduction": f"{comp['tfidf']['noise_reduction']:.1f}%",
                    "Silhouette": comp["tfidf"]["silhouette"],
                },
                {
                    "Method": comp["minilm"]["method"],
                    "Clusters Found": comp["minilm"]["clusters"],
                    "Unique Alerts": comp["minilm"]["unique"],
                    "Noise Reduction": f"{comp['minilm']['noise_reduction']:.1f}%",
                    "Silhouette": comp["minilm"]["silhouette"],
                },
            ]
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        method = st.radio(
            "View clusters for:",
            ["MiniLM + DBSCAN", "TF-IDF + DBSCAN"],
            horizontal=True,
        )
        method_key = "minilm" if method == "MiniLM + DBSCAN" else "tfidf"
        comparison_clusters = dataset_frame(
            DATASET,
            "/alerts/minilm/clusters", {"method": method_key}
        )

        st.subheader(f"Clusters - {method}")
        for _, cluster in comparison_clusters.iterrows():
            cid = int(cluster["cluster_id"])
            label = str(cluster["representative_content"])
            alert_count = int(cluster["alert_count"])
            critical_count = int(cluster["critical_count"])
            worst = float(cluster["max_score"])
            icon = "🔴" if critical_count > 0 else "🟠"

            title = (
                f"{icon} {'Cluster ' + str(cid) if cid >= 0 else 'Unique'} | "
                f"{alert_count} alerts | "
                f"{critical_count} critical | "
                f"Score: {worst:.3f} | {label[:70]}"
            )

            with st.expander(title):
                rows = dataset_frame(
                    DATASET,
                    f"/alerts/minilm/clusters/{cid}",
                    {"method": method_key},
                )
                rows["Time (UTC)"] = rows["window_start"].apply(unix_to_readable)
                rows = rows.rename(
                    columns={
                        "severity": "Severity",
                        "anomaly_score": "Score",
                        "anomaly_count": "Anomaly Lines",
                        "total_logs": "Total Logs",
                        "representative_level": "Level",
                        "representative_content": "Error Content",
                    }
                )
                rows["Score"] = rows["Score"].round(3)
                comparison_columns = [
                    "Time (UTC)",
                    "Severity",
                    "Score",
                    "Total Logs",
                    "Level",
                    "Error Content",
                ]
                if HAS_LABELS:
                    comparison_columns.insert(3, "Anomaly Lines")
                st.dataframe(
                    rows[comparison_columns],
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("Run `python alerts_minilm.py` to see clustering comparison.")
