import pandas as pd


PARSED = "output/parsed.parquet"
FEATURES = "output/features.parquet"
WINDOW = 3600
STEP = 1800


def main() -> None:
    parsed = pd.read_parquet(PARSED, engine="pyarrow")
    feat = pd.read_parquet(FEATURES, engine="pyarrow")

    print("=" * 55)
    print("FEATURE MATRIX VERIFICATION")
    print("=" * 55)

    errors = []
    event_cols = [c for c in feat.columns if c.startswith("e_")]

    print("\n[1] Shape and window count")
    print(f"    Parsed rows   : {len(parsed):,}")
    print(f"    Feature rows  : {len(feat):,}")
    print(f"    Event columns : {len(event_cols):,}")
    if len(feat) == 0:
        errors.append("feature matrix has zero rows")
    if len(event_cols) == 0:
        errors.append("feature matrix has no event columns")

    print("\n[1b] Window configuration")
    required_cols = {"window_mode", "window_size_sec", "step_size_sec"}
    missing_config = required_cols - set(feat.columns)
    if missing_config:
        errors.append(
            "missing window config columns: "
            + ", ".join(sorted(missing_config))
        )
    else:
        mode = feat["window_mode"].iloc[0]
        window_size = int(feat["window_size_sec"].iloc[0])
        step_size = int(feat["step_size_sec"].iloc[0])
        print(f"    Mode       : {mode}")
        print(f"    Window sec : {window_size}")
        print(f"    Step sec   : {step_size}")
        if mode != "sliding":
            errors.append(f"unexpected window_mode: {mode}")
        if window_size != WINDOW:
            errors.append(f"unexpected window_size_sec: {window_size}")
        if step_size != STEP:
            errors.append(f"unexpected step_size_sec: {step_size}")

    print("\n[2] Event ID coverage")
    parsed_eids = set(parsed["event_id"].unique())
    feat_eids = set(int(c[2:]) for c in event_cols)
    missing_eids = parsed_eids - feat_eids
    extra_eids = feat_eids - parsed_eids
    print(f"    Event IDs in parsed : {len(parsed_eids):,}")
    print(f"    Event cols in feat  : {len(feat_eids):,}")
    if missing_eids:
        errors.append(f"{len(missing_eids)} event IDs missing from features")
    if extra_eids:
        errors.append(f"{len(extra_eids)} feature event columns not in parsed")

    print("\n[3] Label consistency")
    inconsistent = feat[
        ((feat["anomaly_count"] > 0) & (feat["is_anomaly"] == 0))
        | ((feat["anomaly_count"] == 0) & (feat["is_anomaly"] == 1))
    ]
    print(f"    Inconsistent rows: {len(inconsistent):,}")
    if len(inconsistent) > 0:
        errors.append(f"{len(inconsistent)} rows have inconsistent labels")

    print("\n[4] Data quality")
    nan_count = int(feat.isnull().sum().sum())
    neg_count = int((feat[event_cols] < 0).sum().sum())
    print(f"    NaN values           : {nan_count:,}")
    print(f"    Negative event counts: {neg_count:,}")
    if nan_count:
        errors.append(f"{nan_count} NaN values found")
    if neg_count:
        errors.append(f"{neg_count} negative event counts found")

    print("\n[5] Manual sample window count")
    sample_idx = len(feat) // 2
    sample_window = feat.iloc[sample_idx]
    w_start = int(sample_window["window_start"])
    w_end = int(sample_window["window_end"])

    actual_logs = parsed[
        (parsed["timestamp"] >= w_start)
        & (parsed["timestamp"] < w_end)
    ]
    sample_total_feat = int(sample_window["total_logs"])
    sample_total_actual = len(actual_logs)
    print(f"    Window start : {w_start}")
    print(f"    Window end   : {w_end}")
    print(f"    feat total   : {sample_total_feat:,}")
    print(f"    actual total : {sample_total_actual:,}")
    if sample_total_feat != sample_total_actual:
        errors.append(
            "sample total_logs mismatch: "
            f"{sample_total_feat} vs {sample_total_actual}"
        )

    manual_counts = actual_logs["event_id"].value_counts().to_dict()
    count_errors = 0
    for eid, manual_count in manual_counts.items():
        col = f"e_{eid}"
        if col not in feat.columns:
            count_errors += 1
            continue
        feat_count = int(sample_window[col])
        if feat_count != int(manual_count):
            count_errors += 1
            print(
                f"    mismatch {col}: "
                f"feature={feat_count} actual={int(manual_count)}"
            )
    if count_errors:
        errors.append(f"{count_errors} event counts differ in sample window")

    print("\n[6] Sliding-window width sanity")
    window_span = feat["window_end"] - feat["window_start"]
    wrong_span = int((window_span != WINDOW).sum())
    print(f"    Rows not exactly {WINDOW}s wide: {wrong_span:,}")
    if wrong_span:
        errors.append(f"{wrong_span} windows are not exactly {WINDOW} seconds")

    starts = feat["window_start"].sort_values().drop_duplicates()
    gaps = starts.diff().dropna()
    bad_steps = int((gaps % STEP != 0).sum())
    missing_empty = int((gaps > STEP).sum())
    print(f"    Gaps over {STEP}s, empty windows omitted: {missing_empty:,}")
    print(f"    Gaps not divisible by {STEP}s: {bad_steps:,}")
    if bad_steps:
        errors.append(f"{bad_steps} adjacent starts are not aligned to {STEP}s")

    print("\nFeature matrix statistics:")
    print(f"  Shape             : {feat.shape}")
    print(f"  Total windows     : {len(feat):,}")
    print(f"  Anomalous windows : {int(feat['is_anomaly'].sum()):,}")
    print(f"  Normal windows    : {int((feat['is_anomaly'] == 0).sum()):,}")
    print(f"  Anomaly rate      : {feat['is_anomaly'].mean():.2%}")
    print(f"  Avg logs/window   : {feat['total_logs'].mean():.0f}")
    print(f"  Max logs/window   : {int(feat['total_logs'].max()):,}")
    print(f"  Avg error_ratio   : {feat['error_ratio'].mean():.4f}")

    print("\nTop 10 event types by mean count:")
    event_means = feat[event_cols].mean().sort_values(ascending=False)
    for col, val in event_means.head(10).items():
        eid = int(col[2:])
        template_rows = parsed[parsed["event_id"] == eid]["template"]
        template = template_rows.iloc[0] if len(template_rows) else "unknown"
        print(f"  {col}: mean={val:.1f} | {template[:55]}")

    print("\n" + "=" * 55)
    if errors:
        print(f"FAILED: {len(errors)} issue(s) found:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)

    print("ALL CHECKS PASSED - feature matrix is consistent")


if __name__ == "__main__":
    main()
