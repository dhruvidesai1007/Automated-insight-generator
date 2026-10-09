"""General-purpose Auto-Analytics Engine.

Pure pandas/numpy/scipy logic (no UI code) so it can be unit-tested and reused
from the Streamlit app, the CLI, or a notebook.

Design rules (from the assignment):
  * No hardcoded district/indicator narratives - every sentence is a template
    and every number inside it is computed from the data.
  * Severity is derived from the *configurable* thresholds, never from magic
    constants such as ``if change > 50``.
  * Trend, outlier, correlation and threshold-breach are separate insight types.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats

INSIGHT_COLUMNS = [
    "insight_id", "type", "indicator", "entity", "period",
    "metric", "value", "prev_value", "change_pct", "change",
    "severity", "score", "explanation",
]
SEVERITY_ORDER = ["High", "Medium", "Low"]
INSIGHT_TYPES = ["trend", "outlier", "correlation", "threshold_breach"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class EngineConfig:
    # Part B - trends
    change_threshold: float = 10.0        # |pct_change| >= this is significant (%)
    consecutive_only: bool = True         # only compare with the immediately preceding calendar month
    # Part C - outliers
    outlier_method: str = "IQR"           # "IQR" | "Z-score"
    outlier_threshold: float = 1.5        # IQR multiplier or |z|
    outlier_scope: str = "period"         # "period" (cross-section per month) | "pooled"
    min_peer_group: int = 5               # skip peer groups smaller than this
    peer_cols: list[str] = field(default_factory=list)  # extra peer grouping, e.g. ["state"]
    # Part D - correlation
    correlation_threshold: float = 0.70
    min_corr_obs: int = 3                 # min_periods for the overall matrix
    entity_corr: bool = True              # also test correlations over time inside each district
    entity_corr_min_periods: int = 12     # months a district needs for its own correlation
    # Severity (multiples of the threshold that is being tested)
    medium_at: float = 1.25
    high_at: float = 1.5
    # Threshold breach: {indicator: (limit, "below" | "above")}; None = derive from data
    breach_limits: Optional[dict] = None
    breach_percentile: float = 5.0        # data-derived default limit (tail percentile)


# --------------------------------------------------------------------------- #
# Part A - loading, cleaning, validation
# --------------------------------------------------------------------------- #
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def clean_names(series: pd.Series) -> pd.Series:
    """Trim, collapse whitespace and title-case so ' north  district ' == 'North District'."""
    return (series.astype("string").str.strip()
            .str.replace(r"\s+", " ", regex=True).str.title())


def infer_columns(df: pd.DataFrame):
    """Best-effort guess of (date, district, state, numeric indicator columns)."""
    cols = list(df.columns)

    def first(exact: set, hints: tuple):
        for c in cols:
            if c in exact:
                return c
        for c in cols:
            if any(h in c for h in hints):
                return c
        return None

    date_col = first({"month", "date", "period", "reporting_month"}, ("month", "date"))
    district_col = first({"district", "district_name", "location"}, ("district",))
    state_col = first({"state", "state_name"}, ("state",))
    excluded = {c for c in (date_col, district_col, state_col) if c}
    numeric = []
    for c in cols:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            parsed = pd.to_numeric(df[c], errors="coerce")
            if parsed.notna().mean() >= 0.7:
                numeric.append(c)
    return date_col, district_col, state_col, numeric


def clean_dataset(df: pd.DataFrame, date_col: Optional[str], district_col: Optional[str],
                  indicators: list[str]):
    """Return (clean_df, log).  Never mutates the input.

    clean_df has helper columns ``_entity`` and ``_period`` (month start).
    Steps are logged so the UI can report exactly what happened.
    """
    log = {"rows_in": int(len(df))}
    out = df.copy()
    for c in indicators:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["_entity"] = clean_names(out[district_col]).fillna("Unknown") if district_col else "All records"
    out["_entity"] = out["_entity"].astype(str)
    if date_col:
        parsed = pd.to_datetime(out[date_col], errors="coerce")
        out["_period"] = parsed.dt.to_period("M").dt.to_timestamp()
    else:
        out["_period"] = pd.NaT
    log["unparseable_dates"] = int(out["_period"].isna().sum()) if date_col else 0
    log["district_names_merged"] = (
        int(df[district_col].nunique() - out["_entity"].nunique()) if district_col else 0)
    before = len(out)
    out = out.drop_duplicates(subset=["_entity", "_period"] + indicators)
    log["exact_duplicates_removed"] = int(before - len(out))
    dup_keys = int(out.duplicated(subset=["_entity", "_period"], keep=False).sum())
    log["conflicting_duplicate_rows"] = dup_keys     # same district+month, different values -> averaged
    return out, log


def validation_report(df: pd.DataFrame, indicators: Optional[list[str]] = None) -> dict:
    """Part A: head(), info(), missing counts (+ range checks)."""
    buf = io.StringIO()
    df.info(buf=buf)
    report = {
        "head": df.head(),
        "info": buf.getvalue(),
        "missing": df.isna().sum().rename("missing_count").to_frame()
                     .assign(missing_pct=lambda t: (t.missing_count / max(len(df), 1) * 100).round(2)),
    }
    checks = []
    for c in (indicators or []):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        pct_like = s.quantile(0.95) <= 100
        checks.append({
            "indicator": c,
            "negative_values": int((s < 0).sum()),
            "above_100_if_percentage": int((s > 100).sum()) if pct_like else None,
            "min": float(s.min()), "max": float(s.max()),
        })
    report["range_checks"] = pd.DataFrame(checks)
    return report


def adequacy_warnings(panel: pd.DataFrame, cfg: EngineConfig) -> list[str]:
    """Data-driven caveats (recommended minimums come from the assignment brief)."""
    warns = []
    n_entities = panel["_entity"].nunique()
    months_per_entity = panel.groupby("_entity")["_period"].nunique()
    n_periods = panel["_period"].nunique()
    if n_periods < 3:
        warns.append(f"Only {n_periods} month(s) of data - fewer than the recommended 3, so trend "
                     "detection is limited to a single month-to-month comparison.")
    if n_entities < 10:
        warns.append(f"Only {n_entities} districts - fewer than the recommended 10, so correlations "
                     "are fragile (a handful of points can produce |r| close to 1).")
    if cfg.entity_corr and months_per_entity.max() < cfg.entity_corr_min_periods:
        warns.append(f"No district has {cfg.entity_corr_min_periods}+ months, so per-district "
                     "time-series correlations were skipped.")
    return warns


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
def build_panel(clean_df: pd.DataFrame, indicators: list[str], peer_cols: Iterable[str] = ()) -> pd.DataFrame:
    """One row per (entity, period); duplicates are averaged."""
    peer_cols = [c for c in peer_cols if c]
    keep_first = {c: "first" for c in peer_cols}
    agg = {**{i: "mean" for i in indicators}, **keep_first}
    panel = (clean_df.dropna(subset=["_period"])
             .groupby(["_entity", "_period"], as_index=False).agg(agg))
    for c in peer_cols:
        panel[c] = clean_names(panel[c]).astype(str)
    return panel.sort_values(["_entity", "_period"]).reset_index(drop=True)


def _long(panel: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    keep = ["_entity", "_period"] + [c for c in panel.columns if c not in indicators + ["_entity", "_period"]]
    return panel.melt(id_vars=keep, value_vars=indicators, var_name="indicator", value_name="value")


# --------------------------------------------------------------------------- #
# Severity (derived from the configurable thresholds)
# --------------------------------------------------------------------------- #
def severity_from_ratio(ratio: float, cfg: EngineConfig) -> str:
    """ratio = how many multiples of the tested threshold the observation reached."""
    if ratio >= cfg.high_at:
        return "High"
    if ratio >= cfg.medium_at:
        return "Medium"
    return "Low"


def severity_from_correlation(r: float, threshold: float) -> str:
    """Position of |r| between the threshold and a perfect correlation, in thirds."""
    headroom = max(1.0 - threshold, 1e-9)
    position = (abs(r) - threshold) / headroom
    return "High" if position >= 2 / 3 else "Medium" if position >= 1 / 3 else "Low"


_LOWER_WORDS = {"of", "in", "per", "and", "to", "for", "the", "vs"}


def _label(indicator: str) -> str:
    """'anc_coverage' -> 'ANC Coverage' (short tokens are treated as acronyms)."""
    words = indicator.replace("_", " ").split()
    return " ".join(w.upper() if len(w) <= 3 and w.lower() not in _LOWER_WORDS else w.capitalize()
                    for w in words)


def _fmt(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- #
# Part B - trend detection
# --------------------------------------------------------------------------- #
def compute_trends(panel: pd.DataFrame, indicators: list[str], cfg: EngineConfig) -> pd.DataFrame:
    """pct_change = (current - previous) / previous * 100 for every (district, indicator, month).

    Returns *all* comparable pairs with an ``is_significant`` flag.
    """
    long = _long(panel, indicators)[["_entity", "_period", "indicator", "value"]].dropna(subset=["value"])
    if cfg.consecutive_only:
        prev = long.copy()
        prev["_period"] = prev["_period"] + pd.offsets.MonthBegin(1)
        trends = long.merge(prev.rename(columns={"value": "prev_value"}),
                            on=["_entity", "_period", "indicator"], how="inner")
    else:
        long = long.sort_values(["_entity", "indicator", "_period"])
        long["prev_value"] = long.groupby(["_entity", "indicator"])["value"].shift(1)
        trends = long.dropna(subset=["prev_value"]).copy()
    trends = trends[trends["prev_value"] != 0].copy()
    trends["pct_change"] = (trends["value"] - trends["prev_value"]) / trends["prev_value"].abs() * 100
    trends["is_significant"] = trends["pct_change"].abs() >= cfg.change_threshold
    return trends.sort_values(["_entity", "indicator", "_period"]).reset_index(drop=True)


def trend_insights(trends: pd.DataFrame, cfg: EngineConfig) -> list[dict]:
    rows = []
    for r in trends[trends["is_significant"]].to_dict("records"):
        pct = r["pct_change"]
        ratio = abs(pct) / cfg.change_threshold
        direction = "increased" if pct > 0 else "dropped"
        basis = "the previous month" if cfg.consecutive_only else "the previous available month"
        rows.append(dict(
            type="trend", indicator=r["indicator"], entity=r["_entity"],
            period=r["_period"].strftime("%Y-%m"), metric="pct_change_vs_previous_month",
            value=float(r["value"]), prev_value=float(r["prev_value"]),
            change_pct=float(pct), change=float(pct),
            severity=severity_from_ratio(ratio, cfg), score=float(ratio),
            explanation=(f"{_label(r['indicator'])} in {r['_entity']} {direction} by {abs(pct):.1f}% "
                         f"compared to {basis} ({_fmt(r['prev_value'])} -> {_fmt(r['value'])}), "
                         f"exceeding the {cfg.change_threshold:g}% significant-change threshold."),
        ))
    return rows


# --------------------------------------------------------------------------- #
# Part C - outlier detection
# --------------------------------------------------------------------------- #
def outlier_insights(panel: pd.DataFrame, indicators: list[str], cfg: EngineConfig) -> list[dict]:
    long = _long(panel, indicators).dropna(subset=["value"])
    if cfg.outlier_scope == "period":
        group_cols = ["_period"] + list(cfg.peer_cols)
    else:
        group_cols = list(cfg.peer_cols)
    group_cols_full = ["indicator"] + group_cols
    rows = []
    peer_text = {"period": "that month", "pooled": "all months"}[cfg.outlier_scope]
    if cfg.peer_cols:
        peer_text += " in the same " + "/".join(cfg.peer_cols)
    for _, grp in long.groupby(group_cols_full, sort=False):
        if len(grp) < cfg.min_peer_group:
            continue
        vals = grp["value"]
        mean = vals.mean()
        if cfg.outlier_method == "IQR":
            q1, q3 = vals.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - cfg.outlier_threshold * iqr, q3 + cfg.outlier_threshold * iqr
            mask = (vals < lo) | (vals > hi)
            bounds_txt = f"IQR fences {lo:.2f} to {hi:.2f}"
        else:
            std = vals.std(ddof=0)
            if not std or np.isnan(std):
                continue
            z = (vals - mean) / std
            mask = z.abs() >= cfg.outlier_threshold
            bounds_txt = f"|z| >= {cfg.outlier_threshold:g}"
        for idx in grp.index[mask]:
            r = grp.loc[idx]
            v = float(r["value"])
            if cfg.outlier_method == "IQR":
                dist = (q1 - v) if v < q1 else (v - q3)
                ratio = dist / iqr / cfg.outlier_threshold
                sigma_txt = ""
            else:
                zi = float((v - mean) / std)
                ratio = abs(zi) / cfg.outlier_threshold
                sigma_txt = f" ({abs(zi):.1f} standard deviations {'below' if zi < 0 else 'above'} the mean)"
            dev_pct = (v - mean) / abs(mean) * 100 if mean else np.nan
            side = "below" if v < mean else "above"
            rows.append(dict(
                type="outlier", indicator=r["indicator"], entity=r["_entity"],
                period=r["_period"].strftime("%Y-%m"),
                metric="deviation_from_peer_mean_pct", value=v, prev_value=float(mean),
                change_pct=float(dev_pct), change=float(dev_pct),
                severity=severity_from_ratio(ratio, cfg), score=float(ratio),
                explanation=(f"{r['_entity']}'s {_label(r['indicator'])} of {_fmt(v)} is {abs(dev_pct):.1f}% "
                             f"{side} the peer mean of {_fmt(mean)} across {len(grp)} districts in {peer_text}"
                             f"{sigma_txt}; outside the {cfg.outlier_method} bounds ({bounds_txt}) - flagged for review."),
            ))
    return rows


# --------------------------------------------------------------------------- #
# Part D - correlation detection
# --------------------------------------------------------------------------- #
def correlation_matrix(panel: pd.DataFrame, indicators: list[str], cfg: EngineConfig) -> pd.DataFrame:
    """Standard ``DataFrame.corr()`` output (Pearson)."""
    return panel[indicators].corr(method="pearson", min_periods=cfg.min_corr_obs)


def _fmt_p(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    return ", p < 0.001" if p < 0.001 else f", p = {p:.3f}"


def _pair_caveat(n: int) -> str:
    if n < 10:
        return f" Based on only {n} observations - treat as fragile."
    if n < 30:
        return f" Based on {n} observations - moderately stable at best."
    return ""


def correlation_insights(panel: pd.DataFrame, indicators: list[str], cfg: EngineConfig,
                         corr: Optional[pd.DataFrame] = None) -> list[dict]:
    rows = []
    if len(indicators) < 2:
        return rows
    corr = corr if corr is not None else correlation_matrix(panel, indicators, cfg)
    p0, p1 = panel["_period"].min(), panel["_period"].max()
    span = p0.strftime("%Y-%m") if p0 == p1 else f"{p0:%Y-%m}..{p1:%Y-%m}"
    for i, a in enumerate(indicators):
        for b in indicators[i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r) or abs(r) < cfg.correlation_threshold:
                continue
            n = int(panel[[a, b]].dropna().shape[0])
            rows.append(dict(
                type="correlation", indicator=f"{a}:{b}", entity="All districts", period=span,
                metric="pearson_r", value=float(r), prev_value=np.nan, change_pct=np.nan, change=float(r),
                severity=severity_from_correlation(r, cfg.correlation_threshold),
                score=float(abs(r) / cfg.correlation_threshold),
                explanation=(f"{_label(a)} and {_label(b)} show a {'positive' if r > 0 else 'negative'} "
                             f"Pearson correlation of r = {r:.2f} across {n} district-month records, "
                             f"at or above the |r| >= {cfg.correlation_threshold:g} threshold. "
                             f"Correlation does not imply causation.{_pair_caveat(n)}"),
            ))
    if cfg.entity_corr:
        for entity, g in panel.groupby("_entity"):
            if g["_period"].nunique() < cfg.entity_corr_min_periods:
                continue
            m = g[indicators].corr(method="pearson", min_periods=cfg.entity_corr_min_periods)
            ep0, ep1 = g["_period"].min(), g["_period"].max()
            espan = f"{ep0:%Y-%m}..{ep1:%Y-%m}"
            for i, a in enumerate(indicators):
                for b in indicators[i + 1:]:
                    r = m.loc[a, b]
                    if pd.isna(r) or abs(r) < cfg.correlation_threshold:
                        continue
                    pair = g[[a, b]].dropna()
                    n = len(pair)
                    pval = stats.pearsonr(pair[a], pair[b]).pvalue if n >= 3 else np.nan
                    rows.append(dict(
                        type="correlation", indicator=f"{a}:{b}", entity=entity, period=espan,
                        metric="pearson_r_over_time", value=float(r), prev_value=np.nan,
                        change_pct=np.nan, change=float(r),
                        severity=severity_from_correlation(r, cfg.correlation_threshold),
                        score=float(abs(r) / cfg.correlation_threshold),
                        explanation=(f"In {entity}, {_label(a)} and {_label(b)} moved together over time "
                                     f"({'positive' if r > 0 else 'negative'}, r = {r:.2f}, n = {n} months"
                                     f"{_fmt_p(pval)}). "
                                     f"Correlation does not imply causation.{_pair_caveat(n)}"),
                    ))
    return rows


# --------------------------------------------------------------------------- #
# Threshold breach (absolute limits, configurable per indicator)
# --------------------------------------------------------------------------- #
def default_breach_limits(panel: pd.DataFrame, indicators: list[str], cfg: EngineConfig) -> dict:
    """Data-derived defaults: low tail for percentage-like indicators (higher = better),
    high tail for counts.  Fully overridable by the user."""
    limits = {}
    for ind in indicators:
        s = panel[ind].dropna()
        if s.empty:
            continue
        pct_like = s.quantile(0.95) <= 100 and s.min() >= 0
        if pct_like:
            limits[ind] = (float(s.quantile(cfg.breach_percentile / 100)), "below")
        else:
            limits[ind] = (float(s.quantile(1 - cfg.breach_percentile / 100)), "above")
    return limits


def breach_insights(panel: pd.DataFrame, indicators: list[str], cfg: EngineConfig,
                    limits: Optional[dict] = None) -> list[dict]:
    limits = limits or cfg.breach_limits or default_breach_limits(panel, indicators, cfg)
    long = _long(panel, indicators).dropna(subset=["value"])
    rows = []
    for ind, (limit, direction) in limits.items():
        if ind not in indicators or limit == 0:
            continue
        sub = long[long["indicator"] == ind]
        hit = sub[sub["value"] < limit] if direction == "below" else sub[sub["value"] > limit]
        for r in hit.to_dict("records"):
            gap_pct = (r["value"] - limit) / abs(limit) * 100
            ratio = abs(gap_pct) / cfg.change_threshold
            word = "below the minimum" if direction == "below" else "above the maximum"
            rows.append(dict(
                type="threshold_breach", indicator=ind, entity=r["_entity"],
                period=r["_period"].strftime("%Y-%m"), metric="gap_to_limit_pct",
                value=float(r["value"]), prev_value=float(limit), change_pct=float(gap_pct), change=float(gap_pct),
                severity=severity_from_ratio(ratio, cfg), score=float(ratio),
                explanation=(f"{_label(ind)} in {r['_entity']} was {_fmt(r['value'])}, {abs(gap_pct):.1f}% "
                             f"{word} limit of {_fmt(limit)}."),
            ))
    return rows


# --------------------------------------------------------------------------- #
# Part E - orchestration
# --------------------------------------------------------------------------- #
def finalize_insights(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=[c for c in INSIGHT_COLUMNS if c != "insight_id"])
    if df.empty:
        return pd.DataFrame(columns=INSIGHT_COLUMNS)
    sev_rank = df["severity"].map({s: i for i, s in enumerate(SEVERITY_ORDER)})
    df = (df.assign(_s=sev_rank)
            .sort_values(["_s", "score"], ascending=[True, False], kind="stable")
            .drop(columns="_s").reset_index(drop=True))
    df.insert(0, "insight_id", [f"INS-{i:04d}" for i in range(1, len(df) + 1)])
    for c in ("value", "prev_value", "change_pct", "change", "score"):
        df[c] = df[c].astype(float).round(3)
    return df


@dataclass
class EngineResult:
    panel: pd.DataFrame
    trends: pd.DataFrame
    corr: pd.DataFrame
    insights: pd.DataFrame
    clean_log: dict
    warnings: list[str]
    breach_limits: dict


def run_engine(df: pd.DataFrame, date_col: Optional[str], district_col: Optional[str],
               indicators: list[str], cfg: Optional[EngineConfig] = None) -> EngineResult:
    cfg = cfg or EngineConfig()
    clean, log = clean_dataset(df, date_col, district_col, indicators)
    panel = build_panel(clean, indicators, cfg.peer_cols)
    trends = compute_trends(panel, indicators, cfg)
    corr = correlation_matrix(panel, indicators, cfg) if len(indicators) > 1 else pd.DataFrame()
    limits = cfg.breach_limits or default_breach_limits(panel, indicators, cfg)
    rows = (trend_insights(trends, cfg) + outlier_insights(panel, indicators, cfg)
            + correlation_insights(panel, indicators, cfg, corr if len(indicators) > 1 else None)
            + breach_insights(panel, indicators, cfg, limits))
    return EngineResult(panel, trends, corr, finalize_insights(rows), log,
                        adequacy_warnings(panel, cfg), limits)
