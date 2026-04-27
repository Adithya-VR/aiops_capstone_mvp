import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import duckdb
from pathlib import Path

from datetime import datetime

def unix_to_readable(ts):
    """Convert Unix timestamp to readable string."""
    try:
        return datetime.utcfromtimestamp(int(ts)).strftime(
            "%Y-%m-%d %H:%M"
        )
    except Exception:
        return str(ts)

st.set_page_config(
    page_title="AIOps — BGL Anomaly Detection",
    page_icon="🔍",
    layout="wide"
)

# ── Load data via DuckDB ───────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        con = duckdb.connect()
        con.execute(
            "CREATE VIEW logs AS "
            "SELECT * FROM 'output/parsed.parquet'"
        )
        con.execute(
            "CREATE VIEW wins AS "
            "SELECT * FROM 'output/scores.parquet'"
        )
        parsed = con.execute("SELECT * FROM logs").df()
        scores = con.execute("SELECT * FROM wins").df()
        return parsed, scores
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.info("Make sure you ran pipeline.py first.")
        st.stop()

parsed, scores = load_data()

# ── Sidebar ────────────────────────────────────────────────────────
st.sidebar.title("🔍 AIOps Dashboard")
st.sidebar.caption("Dataset: BGL (BlueGene/L)")
st.sidebar.divider()
st.sidebar.metric("Total Log Lines",    f"{len(parsed):,}")
st.sidebar.metric("Unique Templates",   f"{parsed['event_id'].nunique()}")
st.sidebar.metric("Anomalous Windows",  f"{scores['predicted'].sum():,}")
st.sidebar.metric("Anomaly Rate",       f"{scores['predicted'].mean():.1%}")
st.sidebar.divider()

thresh = st.sidebar.slider(
    "Anomaly Score Threshold",
    min_value=float(scores["anomaly_score"].min()),
    max_value=float(scores["anomaly_score"].max()),
    value=float(scores["anomaly_score"].quantile(0.90)),
    step=0.01,
    help="Windows above this score are flagged as anomalies"
)
top_n = st.sidebar.slider("Top N Alerts", 5, 50, 20)

# ── Tabs ───────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📊 Overview",
    "📋 Log Explorer",
    "📈 Anomaly Timeline",
    "🚨 Top Alerts",
    "🔔 Alert Clusters"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with t1:
    st.header("System Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Log Lines",    f"{len(parsed):,}")
    c2.metric("Unique Templates",   f"{parsed['event_id'].nunique()}")
    c3.metric("Anomalous Lines",    f"{parsed['is_anomaly'].sum():,}")
    c4.metric("Anomaly Rate",       f"{parsed['is_anomaly'].mean():.2%}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Log Level Distribution")
        lc = (parsed["level"]
              .value_counts()
              .reset_index())
        lc.columns = ["level", "count"]
        fig = px.pie(
            lc, names="level", values="count",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 15 Log Templates")
        tt = (
            parsed.groupby(["event_id", "template"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(15)
        )
        tt["short"] = tt["template"].str[:55]
        fig2 = px.bar(
            tt, x="count", y="short",
            orientation="h",
            color="count",
            color_continuous_scale="Teal",
            labels={"short": "Template", "count": "Count"}
        )
        fig2.update_layout(yaxis_title="", height=420)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Anomaly Score Distribution")
    fig3 = px.histogram(
        scores, x="anomaly_score", nbins=80,
        color_discrete_sequence=["#7F77DD"],
        labels={"anomaly_score": "Anomaly Score"}
    )
    fig3.add_vline(
        x=thresh, line_dash="dash", line_color="red",
        annotation_text=f"Threshold: {thresh:.2f}",
        annotation_position="top right"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — LOG EXPLORER
# ══════════════════════════════════════════════════════════════════
with t2:
    st.header("Log Explorer")

    # ── Filters ───────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    f_show   = c1.selectbox(
        "Show", ["All", "Anomalies only", "Normal only"]
    )
    f_level  = c2.multiselect(
        "Log Level",
        options=sorted(parsed["level"].unique().tolist()),
        default=sorted(parsed["level"].unique().tolist())
    )
    f_search = c3.text_input("Search in content", "")

    # ── Apply filters ─────────────────────────────────────────────
    view = parsed.copy()
    if f_show == "Anomalies only":
        view = view[view["is_anomaly"] == 1]
    elif f_show == "Normal only":
        view = view[view["is_anomaly"] == 0]
    if f_level:
        view = view[view["level"].isin(f_level)]
    if f_search:
        view = view[
            view["content"].str.contains(
                f_search, case=False, na=False
            )
        ]

    total_rows  = len(view)
    page_size   = 500
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    # ── Pagination controls ───────────────────────────────────────
    st.caption(f"**{total_rows:,} rows** match your filters")

    col_info, col_prev, col_page, col_next = st.columns([3, 1, 2, 1])

    with col_info:
        st.write(f"Page size: {page_size} rows per page")

    # Use session state to remember current page
    if "log_page" not in st.session_state:
        st.session_state.log_page = 1

    # Reset to page 1 when filters change
    filter_key = f"{f_show}_{f_level}_{f_search}"
    if "last_filter" not in st.session_state:
        st.session_state.last_filter = filter_key
    if st.session_state.last_filter != filter_key:
        st.session_state.log_page   = 1
        st.session_state.last_filter = filter_key

    with col_prev:
        if st.button("◀ Prev",
                     disabled=st.session_state.log_page <= 1):
            st.session_state.log_page -= 1
            st.rerun()

    with col_page:
        st.write(
            f"**Page {st.session_state.log_page}"
            f" of {total_pages}**"
        )

    with col_next:
        if st.button("Next ▶",
                     disabled=st.session_state.log_page >= total_pages):
            st.session_state.log_page += 1
            st.rerun()

    # ── Jump to page ──────────────────────────────────────────────
    jump = st.number_input(
        "Jump to page",
        min_value=1,
        max_value=total_pages,
        value=st.session_state.log_page,
        step=1,
        key="page_jump"
    )
    if jump != st.session_state.log_page:
        st.session_state.log_page = int(jump)
        st.rerun()

    # ── Slice the data for current page ───────────────────────────
    page     = st.session_state.log_page
    start    = (page - 1) * page_size
    end      = start + page_size
    page_view = view.iloc[start:end]

    start_row = start + 1
    end_row   = min(end, total_rows)
    st.caption(
        f"Showing rows **{start_row:,} – {end_row:,}** "
        f"of **{total_rows:,}**"
    )

    # ── Display ───────────────────────────────────────────────────
    def highlight(row):
        color = ("background-color: #ffcccc"
                 if row["is_anomaly"] else "")
        return [color] * len(row)

    display_cols = [
        "line_id", "date", "node", "level",
        "is_anomaly", "template", "content"
    ]
    st.dataframe(
        page_view[display_cols]
        .style.apply(highlight, axis=1),
        use_container_width=True,
        height=450
    )

    # ── Quick stats for current filter ───────────────────────────
    st.divider()
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Filtered rows",    f"{total_rows:,}")
    sc2.metric("Anomalous",
               f"{view['is_anomaly'].sum():,}")
    sc3.metric("Normal",
               f"{(view['is_anomaly']==0).sum():,}")
    sc4.metric("Anomaly rate",
               f"{view['is_anomaly'].mean():.2%}"
               if total_rows > 0 else "N/A")

# ══════════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY TIMELINE
# ══════════════════════════════════════════════════════════════════
with t3:
    st.header("Anomaly Timeline")

    # Convert timestamps to readable format
    scores["window_dt"] = scores["window_start"].apply(unix_to_readable)

    flagged = scores[scores["anomaly_score"] >= thresh]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scores["window_dt"],
        y=scores["anomaly_score"],
        mode="lines",
        name="Anomaly Score",
        line=dict(color="#7F77DD", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=flagged["window_dt"],
        y=flagged["anomaly_score"],
        mode="markers",
        name="Flagged",
        marker=dict(color="red", size=5, symbol="circle")
    ))
    fig.add_hline(
        y=thresh, line_dash="dash", line_color="red",
        annotation_text=f"Threshold: {thresh:.2f}"
    )
    fig.update_layout(
        xaxis_title="Time (UTC)",
        yaxis_title="Anomaly Score",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score vs Log Volume")
        fig2 = px.scatter(
            scores, x="total_logs", y="anomaly_score",
            color=scores["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
            color_discrete_map={
                "Normal":  "#1D9E75",
                "Anomaly": "#E24B4A"
            },
            opacity=0.6,
            labels={
                "total_logs":    "Logs in Window",
                "anomaly_score": "Anomaly Score",
                "color":         "Ground Truth"
            }
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Confusion Matrix")
        cm = pd.crosstab(
            scores["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
            scores["predicted"].map({0: "Normal", 1: "Anomaly"}),
            rownames=["Actual"],
            colnames=["Predicted"]
        )
        fig3 = px.imshow(
            cm, text_auto=True,
            color_continuous_scale="Blues",
            title="Actual vs Predicted"
        )
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — TOP ALERTS
# ══════════════════════════════════════════════════════════════════
with t4:
    st.header("Top Anomalous Windows")

    p95 = float(scores["anomaly_score"].quantile(0.95))
    p85 = float(scores["anomaly_score"].quantile(0.85))
    p70 = float(scores["anomaly_score"].quantile(0.70))
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

        label = (
            f"{sev}  |  "
            f"Score: {score:.3f}  |  "
            f"Anomalous lines: {int(row['anomaly_count'])}  |  "
            f"{unix_to_readable(row['window_start'])}"
        )

        with st.expander(label):
            c1, c2, c3 = st.columns(3)
            c1.metric("Anomaly Score",   f"{score:.3f}")
            c2.metric("Total Logs",      f"{int(row['total_logs'])}")
            c3.metric("Anomalous Lines", f"{int(row['anomaly_count'])}")

            window_logs = parsed[
                (parsed["timestamp"] >= row["window_start"]) &
                (parsed["timestamp"] <  row["window_end"])
            ]

            if not window_logs.empty:
                st.caption(
                    f"{len(window_logs)} log lines in this window"
                )
                st.dataframe(
                    window_logs[[
                        "node", "level", "is_anomaly",
                        "template", "content"
                    ]].head(20),
                    use_container_width=True
                )
            else:
                st.caption("No log lines found for this window.")
# ══════════════════════════════════════════════════════════════════
# TAB 5 — ALERT CLUSTERS
# ══════════════════════════════════════════════════════════════════
with t5:
    st.header("Alert Clusters")

    alerts_path = Path("output/alerts.parquet")
    if not alerts_path.exists():
        st.warning("Run `python alerts.py` first to generate clusters.")
        st.stop()

    @st.cache_data
    def load_alerts():
        return pd.read_parquet(
            "output/alerts.parquet", engine="pyarrow"
        )

    alerts = load_alerts()

    # Summary metrics
    n_clusters = alerts[alerts["cluster_id"] >= 0]["cluster_id"].nunique()
    n_unique   = (alerts["cluster_id"] == -1).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Alerts",    f"{len(alerts):,}")
    c2.metric("Clusters Found",  f"{n_clusters}")
    c3.metric("Unique Alerts",   f"{n_unique}")
    c4.metric("Noise Reduced",
              f"{(1 - (n_clusters + n_unique)/len(alerts)):.1%}")

    st.divider()

    # Severity breakdown
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Alerts by Severity")
        sev = alerts["severity"].value_counts().reset_index()
        sev.columns = ["severity", "count"]
        color_map = {
            "CRITICAL": "#E24B4A",
            "HIGH":     "#EF9F27",
            "MEDIUM":   "#EDD94C",
            "LOW":      "#4CAF50"    # green for low severity
        }
        fig = px.bar(
            sev, x="severity", y="count",
            color="severity",
            color_discrete_map=color_map
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Clusters by Size")
        cluster_summary = (
            alerts[alerts["cluster_id"] >= 0]
            .groupby("cluster_label")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        cluster_summary["label_short"] = (
            cluster_summary["cluster_label"].str[:45]
        )
        fig2 = px.bar(
            cluster_summary,
            x="count", y="label_short",
            orientation="h",
            color="count",
            color_continuous_scale="Reds"
        )
        fig2.update_layout(yaxis_title="", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("All Alert Clusters")

    # Group alerts by cluster and show expandable detail
    for cid in sorted(alerts["cluster_id"].unique()):
        group      = alerts[alerts["cluster_id"] == cid]
        label      = group["cluster_label"].iloc[0]
        worst      = group["anomaly_score"].max()
        sev_counts = group["severity"].value_counts().to_dict()

        icon = ("🔴" if "CRITICAL" in sev_counts
                else "🟠" if "HIGH" in sev_counts
                else "🟡" if "MEDIUM" in sev_counts
                else "🟢")

        title = (
            f"{icon} {'Cluster ' + str(cid) if cid >= 0 else 'Unique'}  |  "
            f"{len(group)} alerts  |  "
            f"Max score: {worst:.3f}  |  "
            f"{label[:50]}"
        )

        with st.expander(title):
            c1, c2, c3 = st.columns(3)
            c1.metric("Alerts in cluster", len(group))
            c2.metric("Max anomaly score", f"{worst:.3f}")
            c3.metric("Critical alerts",
                       sev_counts.get("CRITICAL", 0))

            # Readable timestamps — all indented inside expander
            display_group = group[[
                "window_start", "anomaly_score",
                "severity", "anomaly_count",
                "total_logs", "top_template"
            ]].copy()
            display_group["window_start"] = (
                display_group["window_start"].apply(unix_to_readable)
            )
            display_group = display_group.rename(
                columns={"window_start": "time (UTC)"}
            )
            st.dataframe(
                display_group.sort_values(
                    "anomaly_score", ascending=False
                ),
                use_container_width=True
            )