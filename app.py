"""Auto-Analytics Engine for District Healthcare - Streamlit UI.

All analytics live in ``src/insight_engine.py``; this file only wires widgets to it.
Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.insight_engine import (
    INSIGHT_COLUMNS, SEVERITY_ORDER, INSIGHT_TYPES, EngineConfig,
    adequacy_warnings, build_panel, clean_dataset, correlation_insights, correlation_matrix, default_breach_limits,
    _label as pretty_label, finalize_insights, infer_columns, normalize_column_names, run_engine, validation_report,
)

APP_DIR = Path(__file__).resolve().parent
SOURCES = {
    "Bundled: 15k-row synthetic dataset": APP_DIR / "data" / "raw_district_healthcare_dataset_15k.csv",
}
SEVERITY_COLORS = {"High": "#d62728", "Medium": "#ff9f1c", "Low": "#2a9d8f"}
TYPE_COLORS = {"trend": "#4C78A8", "outlier": "#B279A2", "correlation": "#9D755D", "threshold_breach": "#8C8C8C"}
BADGE = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

st.set_page_config(page_title="Auto Healthcare Analytics", page_icon="🏥", layout="wide")
st.title("🏥 Auto-Analytics Engine for District Healthcare")
st.caption("Automatically detects trends, outliers, correlations and threshold breaches, "
           "and explains them in plain language. Every number in an insight comes from the data.")


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def read_csv_cached(path_or_bytes, name: str) -> pd.DataFrame:
    import io
    src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    return normalize_column_names(pd.read_csv(src))


@st.cache_data(show_spinner="Analysing...")
def analyze(df: pd.DataFrame, date_col, district_col, indicators: tuple, cfg_dict: dict):
    cfg = EngineConfig(**cfg_dict)
    return run_engine(df, date_col, district_col, list(indicators), cfg)


@st.cache_data(show_spinner=False)
def prepare_panel(df: pd.DataFrame, date_col, district_col, indicators: tuple, peer_cols: tuple):
    clean, _ = clean_dataset(df, date_col, district_col, list(indicators))
    return build_panel(clean, list(indicators), peer_cols)


# --------------------------------------------------------------------------- #
# Sidebar - data source
# --------------------------------------------------------------------------- #
st.sidebar.header("1 · Data")
uploaded = st.sidebar.file_uploader("Upload a CSV (optional)", type=["csv"])
source_name = st.sidebar.selectbox("…or use a bundled dataset", list(SOURCES), disabled=uploaded is not None)
try:
    if uploaded is not None:
        raw = read_csv_cached(uploaded.getvalue(), uploaded.name)
        using_bundled_15k = False
    else:
        raw = read_csv_cached(SOURCES[source_name], source_name)
        using_bundled_15k = "15k" in source_name
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load CSV: {exc}")
    st.stop()

if using_bundled_15k:
    st.warning("Dataset notice: the bundled 15k dataset is **synthetic**, raw-style data created for software "
               "testing. It is not official HMIS data and must not be used for real healthcare decisions.")

date_guess, district_guess, state_guess, numeric_guess = infer_columns(raw)
if not numeric_guess:
    st.error("No numerical indicator columns were detected. Check the CSV schema.")
    st.stop()

cols = ["(none)"] + raw.columns.tolist()
date_col = st.sidebar.selectbox("Month / date column", cols, index=cols.index(date_guess) if date_guess in cols else 0)
district_col = st.sidebar.selectbox("District column", cols, index=cols.index(district_guess) if district_guess in cols else 0)
peer_choice = st.sidebar.selectbox(
    "Compare outliers within the same… (optional)", cols,
    index=cols.index(state_guess) if state_guess in cols else 0,
    help="e.g. pick 'state' to judge a district only against districts of its own state.")
date_col = None if date_col == "(none)" else date_col
district_col = None if district_col == "(none)" else district_col
peer_cols = [] if peer_choice == "(none)" else [peer_choice]
if date_col is None:
    st.error("A month/date column is required for trend detection. Select one in the sidebar.")
    st.stop()

indicators = st.sidebar.multiselect("Indicators (filter + analysis set)", numeric_guess, default=numeric_guess)
if not indicators:
    st.info("Select at least one indicator in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar - thresholds (all configurable)
# --------------------------------------------------------------------------- #
st.sidebar.header("2 · Thresholds")
change_threshold = st.sidebar.slider("Significant month-to-month change (± %)", 1, 100, 10)
consecutive_only = st.sidebar.checkbox("Only compare consecutive calendar months", value=True,
                                       help="If off, a gap in the data compares against the last available month.")
outlier_method = st.sidebar.selectbox("Outlier method", ["IQR", "Z-score"])
if outlier_method == "IQR":
    outlier_threshold = st.sidebar.slider("IQR multiplier", 0.5, 4.0, 1.5, 0.25)
else:
    outlier_threshold = st.sidebar.slider("|Z-score| threshold", 1.0, 5.0, 3.0, 0.1)
outlier_scope = st.sidebar.radio("Outlier comparison", ["period", "pooled"], horizontal=True,
                                 format_func=lambda x: "Within each month" if x == "period" else "All months pooled")
min_peer_group = st.sidebar.slider("Minimum districts in a comparison group", 3, 30, 5)
correlation_threshold = st.sidebar.slider("|r| correlation threshold", 0.10, 1.00, 0.70, 0.05)
entity_corr = st.sidebar.checkbox("Also test correlations over time within each district", value=True)
entity_corr_min = st.sidebar.slider("Months a district needs for its own correlation", 3, 36, 12,
                                    disabled=not entity_corr)
st.sidebar.caption("Severity = how many multiples of the tested threshold an observation reaches.")
medium_at = st.sidebar.slider("Medium severity from (× threshold)", 1.0, 3.0, 1.25, 0.05)
high_at = st.sidebar.slider("High severity from (× threshold)", 1.0, 4.0, 1.5, 0.05)
if high_at <= medium_at:
    st.sidebar.warning("High should be above Medium; using Medium + 0.05.")
    high_at = medium_at + 0.05
breach_pct = st.sidebar.slider("Default limit = tail percentile", 1, 25, 5,
                               help="Data-derived default for the absolute limits below.")

base_cfg = EngineConfig(
    change_threshold=float(change_threshold), consecutive_only=consecutive_only,
    outlier_method=outlier_method, outlier_threshold=float(outlier_threshold), outlier_scope=outlier_scope,
    min_peer_group=int(min_peer_group), peer_cols=peer_cols,
    correlation_threshold=float(correlation_threshold), entity_corr=entity_corr,
    entity_corr_min_periods=int(entity_corr_min),
    medium_at=float(medium_at), high_at=float(high_at), breach_percentile=float(breach_pct),
)

# The cleaned panel gives us data-derived default limits that the user may override.
limits = default_breach_limits(prepare_panel(raw, date_col, district_col, tuple(indicators), tuple(peer_cols)),
                               indicators, base_cfg)
with st.sidebar.expander("Absolute limits (threshold breach)"):
    manual = {}
    for ind in indicators:
        if ind not in limits:
            continue
        lim, direction = limits[ind]
        # Key includes the derived default so widgets reset when the dataset/percentile changes.
        sig = f"{ind}_{lim:.4f}_{direction}"
        c1, c2 = st.columns([3, 2])
        new_lim = c1.number_input(ind, value=float(round(lim, 2)), key=f"lim_{sig}")
        new_dir = c2.selectbox("alert", ["below", "above"], index=0 if direction == "below" else 1,
                               key=f"dir_{sig}", label_visibility="collapsed")
        manual[ind] = (float(new_lim), new_dir)
cfg_dict = asdict(base_cfg)
cfg_dict["breach_limits"] = manual
result = analyze(raw, date_col, district_col, tuple(indicators), cfg_dict)
cfg = EngineConfig(**cfg_dict)
panel = result.panel

# --------------------------------------------------------------------------- #
# Sidebar - live filters
# --------------------------------------------------------------------------- #
st.sidebar.header("3 · Filters")
all_entities = sorted(panel["_entity"].unique())
chosen_entities = st.sidebar.multiselect("Districts (empty = all)", all_entities)
periods = sorted(panel["_period"].dropna().unique())
period_labels = [pd.Timestamp(p).strftime("%Y-%m") for p in periods]
if len(period_labels) > 1:
    month_range = st.sidebar.select_slider("Months", options=period_labels,
                                           value=(period_labels[-1], period_labels[-1]))
else:
    month_range = (period_labels[0], period_labels[0])
sev_filter = st.sidebar.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER)
type_filter = st.sidebar.multiselect("Insight type", INSIGHT_TYPES, default=INSIGHT_TYPES)
max_shown = st.sidebar.slider("Max insights listed", 10, 1000, 200, 10)

# --------------------------------------------------------------------------- #
# Apply filters to the view
# --------------------------------------------------------------------------- #
lo, hi = month_range
view_panel = panel[(panel["_period"] >= pd.Timestamp(lo + "-01")) & (panel["_period"] <= pd.Timestamp(hi + "-01"))]
if chosen_entities:
    view_panel = view_panel[view_panel["_entity"].isin(chosen_entities)]

ins = result.insights.copy()
if not ins.empty:
    # Dataset-level correlations are recomputed on exactly the data in view (filters change the sample).
    ins = ins[~((ins["type"] == "correlation") & (ins["entity"] == "All districts"))]
    keep = ((ins["type"] == "correlation") | ((ins["period"] >= lo) & (ins["period"] <= hi)))
    ins = ins[keep]
    if chosen_entities:
        ins = ins[ins["entity"].isin(chosen_entities)]
    ins = ins[ins["indicator"].apply(lambda s: all(p in indicators for p in s.split(":")))]
corr_rows = []
if len(indicators) >= 2 and len(view_panel):
    view_cfg = EngineConfig(**{**cfg_dict, "entity_corr": False})
    corr_rows = correlation_insights(view_panel, indicators, view_cfg)
    for r in corr_rows:
        r["entity"] = "Selected districts" if chosen_entities else "All districts"
rows = ins.drop(columns="insight_id", errors="ignore").to_dict("records") + corr_rows
view_ins = finalize_insights(rows)
if not view_ins.empty:
    view_ins = view_ins[view_ins["severity"].isin(sev_filter) & view_ins["type"].isin(type_filter)]

# --------------------------------------------------------------------------- #
# Header metrics
# --------------------------------------------------------------------------- #
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Rows loaded", f"{len(raw):,}")
m2.metric("District-months analysed", f"{len(panel):,}")
m3.metric("Districts in view", f"{view_panel['_entity'].nunique():,}")
m4.metric("Months in view", f"{view_panel['_period'].nunique():,}")
m5.metric("Insights in view", f"{len(view_ins):,}")
for w in adequacy_warnings(panel, cfg):
    st.warning(w)
if lo == hi:
    st.caption(f"Showing **{lo}** only (default = latest month). Widen the *Months* filter to review history.")

tab_ins, tab_charts, tab_data, tab_method = st.tabs(
    ["💡 Insights", "📊 Charts", "🧾 Data & validation", "📘 Methodology"])

# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #
with tab_ins:
    if view_ins.empty:
        st.info("No insights cross the current thresholds/filters. Lower a threshold or widen the filters.")
    else:
        counts = (view_ins.groupby(["severity", "type"]).size().reset_index(name="count"))
        cA, cB = st.columns(2)
        sev_totals = view_ins["severity"].value_counts().reindex(SEVERITY_ORDER, fill_value=0)
        cA.markdown("**Severity counts**  " + "  ·  ".join(f"{BADGE[s]} {s}: {int(n):,}" for s, n in sev_totals.items()))
        fig = px.bar(counts, x="severity", y="count", color="type", category_orders={"severity": SEVERITY_ORDER},
                     color_discrete_map=TYPE_COLORS, title="Insights by severity (stacked by type)", barmode="stack")
        cA.plotly_chart(fig, width="stretch")
        type_counts = view_ins["type"].value_counts().rename_axis("type").reset_index(name="count")
        cB.plotly_chart(px.bar(type_counts, x="type", y="count", color="type", color_discrete_map=TYPE_COLORS,
                                title="Insights by type"),
                        width="stretch")

        st.subheader(f"Top insights ({min(len(view_ins), max_shown):,} of {len(view_ins):,})")
        for r in view_ins.head(8).itertuples(index=False):
            st.markdown(f"{BADGE[r.severity]} **{r.severity}** · `{r.type}` · {r.explanation}")

        shown = view_ins.head(max_shown)
        st.dataframe(shown[INSIGHT_COLUMNS], width="stretch", hide_index=True)
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ Insights CSV (all filtered)", view_ins[INSIGHT_COLUMNS].to_csv(index=False).encode(),
                           "insights.csv", "text/csv")
        d2.download_button("⬇ Insights JSON (all filtered)",
                           json.dumps(json.loads(view_ins[INSIGHT_COLUMNS].to_json(orient="records")), indent=2),
                           "insights.json", "application/json")
        if len(indicators) >= 2:
            d3.download_button("⬇ Correlation matrix CSV",
                               correlation_matrix(view_panel, indicators, cfg).to_csv().encode(),
                               "correlation_matrix.csv", "text/csv")

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
with tab_charts:
    st.subheader("Correlation heatmap (Pearson)")
    if len(indicators) < 2 or len(view_panel) < 3:
        st.info("Select at least two indicators and enough rows in view (≥3) to compute correlations.")
    else:
        corr = correlation_matrix(view_panel, indicators, cfg)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                                  aspect="auto"), width="stretch")
        n_rows = len(view_panel)
        st.caption(f"Computed on {n_rows:,} district-month rows in view. Pairs with |r| ≥ "
                   f"{correlation_threshold:g} are flagged as insights. "
                   + ("**With so few rows the matrix is fragile - read it as indicative only.**" if n_rows < 30 else ""))

    st.subheader("Per-district trend")
    ind_chart = st.selectbox("Indicator", indicators, key="line_ind")
    flagged = view_ins[(view_ins["indicator"] == ind_chart) & (view_ins["entity"].isin(all_entities))] \
        if not view_ins.empty else pd.DataFrame()
    default_ents = (chosen_entities or (flagged["entity"].drop_duplicates().head(5).tolist() if len(flagged) else [])
                    or all_entities[:5])
    plot_ents = st.multiselect("Districts to plot", all_entities, default=default_ents[:10], key=f"line_ents_{ind_chart}")
    if plot_ents:
        line = panel[panel["_entity"].isin(plot_ents)]
        fig = px.line(line, x="_period", y=ind_chart, color="_entity", markers=True,
                      labels={"_period": "Month", ind_chart: pretty_label(ind_chart), "_entity": "District"})
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Pick at least one district to plot.")

# --------------------------------------------------------------------------- #
# Data & validation (Part A)
# --------------------------------------------------------------------------- #
with tab_data:
    rep = validation_report(raw, indicators)
    st.subheader("head()")
    st.dataframe(rep["head"], width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("info()")
        st.code(rep["info"])
    with c2:
        st.subheader("Missing values per column")
        st.dataframe(rep["missing"], width="stretch")
    st.subheader("Range checks")
    st.dataframe(rep["range_checks"], width="stretch", hide_index=True)
    st.subheader("Cleaning log")
    log = result.clean_log
    st.write({
        "Rows loaded": log["rows_in"],
        "Unparseable dates (excluded from analysis)": log["unparseable_dates"],
        "District spellings merged (case / whitespace)": log["district_names_merged"],
        "Exact duplicate rows removed": log["exact_duplicates_removed"],
        "Rows sharing a district+month with different values (averaged)": log["conflicting_duplicate_rows"],
    })
    st.caption("The original upload is never modified; the engine works on a cleaned copy.")

# --------------------------------------------------------------------------- #
# Methodology
# --------------------------------------------------------------------------- #
with tab_method:
    st.markdown(f"""
**Trend** - `pct_change = (current − previous) / |previous| × 100` per district × indicator. Flagged
(`is_significant`) when `|pct_change| ≥ {change_threshold}%`. {'Only the immediately preceding calendar month is used.' if consecutive_only else 'The previous *available* month is used, even across gaps.'}

**Outlier** - {outlier_method} rule ({'value < Q1 − k·IQR or > Q3 + k·IQR, k = ' + str(outlier_threshold) if outlier_method == 'IQR' else '|z| ≥ ' + str(outlier_threshold)}),
computed per indicator {'within each month' if outlier_scope == 'period' else 'over all months pooled'}{' and within the same ' + peer_cols[0] if peer_cols else ''};
groups with fewer than {min_peer_group} districts are skipped. A z-score cannot exceed (n−1)/√n, so for small groups
|z| ≥ 3 is mathematically unreachable - use IQR there.

**Correlation** - Pearson matrix over district-month rows; pairs with `|r| ≥ {correlation_threshold}` are flagged, and optionally
repeated inside each district over time. *With few months × few districts correlations are fragile; every correlation insight states its sample size.*

**Threshold breach** - a value beyond an absolute limit (default = data-derived {breach_pct}th-percentile tail: low tail for
percentage-like indicators, high tail for counts; editable in the sidebar).

**Severity** - ratio of the observation to the threshold being tested: trend `|Δ%| / threshold`, outlier distance beyond the
fence in fence-units (IQR) or `|z| / threshold`, breach gap `/` change threshold. Low `< {medium_at}×`, Medium `{medium_at}–{high_at}×`,
High `≥ {high_at}×`. Correlation severity splits the range between the |r| threshold and 1.0 into thirds.

**Limitations** - Percent changes on very small counts explode (1 → 23 = +2,200%). Flags are prompts for review, not diagnoses.
Correlation is not causation. Testing hundreds of districts × pairs will produce some chance correlations.
""")
