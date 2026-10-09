"""Unit tests for the insight engine (run: pytest -q)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.insight_engine import (
    EngineConfig, INSIGHT_COLUMNS, INSIGHT_TYPES, SEVERITY_ORDER, build_panel, clean_dataset,
    compute_trends, correlation_matrix, correlation_insights, infer_columns, normalize_column_names,
    outlier_insights, run_engine, severity_from_correlation, severity_from_ratio, validation_report,
)

ROOT = Path(__file__).resolve().parents[1]
INDS = ["anc_coverage", "institutional_delivery", "immunization", "high_risk_cases"]


@pytest.fixture(scope="module")
def sample():
    df = normalize_column_names(pd.read_csv(ROOT / "data" / "raw_district_healthcare_dataset_15k.csv"))
    d, di, _, ind = infer_columns(df)
    return df, d, di, ind


def _panel(rows, inds=("x",)):
    df = pd.DataFrame(rows, columns=["month", "district", *inds])
    clean, _ = clean_dataset(df, "month", "district", list(inds))
    return build_panel(clean, list(inds))


# ---------------- Part B: trends ---------------- #
def test_pct_change_formula():
    panel = _panel([("2026-07", "A", 85), ("2026-08", "A", 69)])
    t = compute_trends(panel, ["x"], EngineConfig(change_threshold=10))
    assert len(t) == 1
    assert t["pct_change"].iloc[0] == pytest.approx((69 - 85) / 85 * 100)
    assert bool(t.is_significant.iloc[0])


def test_threshold_is_configurable():
    panel = _panel([("2026-07", "A", 100), ("2026-08", "A", 108)])
    assert not compute_trends(panel, ["x"], EngineConfig(change_threshold=10)).is_significant.iloc[0]
    assert compute_trends(panel, ["x"], EngineConfig(change_threshold=5)).is_significant.iloc[0]


def test_consecutive_only_skips_gaps():
    panel = _panel([("2026-01", "A", 100), ("2026-03", "A", 50)])
    assert compute_trends(panel, ["x"], EngineConfig(consecutive_only=True)).empty
    assert len(compute_trends(panel, ["x"], EngineConfig(consecutive_only=False))) == 1


def test_zero_previous_is_skipped():
    panel = _panel([("2026-01", "A", 0), ("2026-02", "A", 5)])
    assert compute_trends(panel, ["x"], EngineConfig()).empty


# ---------------- Part C: outliers ---------------- #
def _cross_section(values):
    return _panel([("2026-08", f"D{i}", v) for i, v in enumerate(values)])


def test_iqr_flags_extreme_value_only():
    panel = _cross_section([80, 82, 79, 81, 83, 42])
    rows = outlier_insights(panel, ["x"], EngineConfig(outlier_method="IQR", outlier_threshold=1.5))
    assert [r["entity"] for r in rows] == ["D5"]
    assert rows[0]["value"] == 42


def test_zscore_ceiling_small_sample():
    # With n=6 the max possible |z| is (n-1)/sqrt(n) ~ 2.04, so |z|>=3 can never fire.
    panel = _cross_section([80, 82, 79, 81, 83, 42])
    assert outlier_insights(panel, ["x"], EngineConfig(outlier_method="Z-score", outlier_threshold=3)) == []
    assert outlier_insights(panel, ["x"], EngineConfig(outlier_method="Z-score", outlier_threshold=1.8))


def test_small_groups_skipped_and_constant_groups_safe():
    assert outlier_insights(_cross_section([1, 2, 100]), ["x"], EngineConfig(min_peer_group=5)) == []
    assert outlier_insights(_cross_section([5] * 8), ["x"], EngineConfig()) == []


def test_outliers_are_per_month_not_pooled():
    # 60 is normal in July (everyone ~60) but extreme in August (everyone ~90).
    rows = [("2026-07", f"D{i}", 60 + i % 3) for i in range(8)] + \
           [("2026-08", f"D{i}", 90 + i % 3) for i in range(7)] + [("2026-08", "D7", 60)]
    panel = _panel(rows)
    per = outlier_insights(panel, ["x"], EngineConfig(outlier_scope="period"))
    assert [(r["entity"], r["period"]) for r in per] == [("D7", "2026-08")]


# ---------------- Part D: correlation ---------------- #
def test_correlation_flagged_only_above_threshold():
    rng = np.random.default_rng(0)
    a = rng.normal(size=40)
    df = pd.DataFrame({"month": "2026-01", "district": [f"D{i}" for i in range(40)],
                       "a": a, "b": a * 2 + rng.normal(scale=0.01, size=40), "c": rng.normal(size=40)})
    clean, _ = clean_dataset(df, "month", "district", ["a", "b", "c"])
    panel = build_panel(clean, ["a", "b", "c"])
    cfg = EngineConfig(entity_corr=False)
    rows = correlation_insights(panel, ["a", "b", "c"], cfg)
    assert [r["indicator"] for r in rows] == ["a:b"]
    assert rows[0]["value"] > 0.99
    assert correlation_insights(panel, ["a", "b", "c"], EngineConfig(correlation_threshold=1.0, entity_corr=False)) == []
    assert correlation_matrix(panel, ["a", "b", "c"], cfg).shape == (3, 3)


def test_fragile_correlation_is_labelled():
    panel = _panel([("2026-01", f"D{i}", i, 2 * i) for i in range(5)], inds=("a", "b"))
    rows = correlation_insights(panel, ["a", "b"], EngineConfig(entity_corr=False))
    assert "fragile" in rows[0]["explanation"]


# ---------------- Severity ---------------- #
def test_severity_follows_configurable_thresholds():
    cfg = EngineConfig(medium_at=1.25, high_at=1.5)
    assert [severity_from_ratio(x, cfg) for x in (1.0, 1.3, 1.6)] == ["Low", "Medium", "High"]
    strict = EngineConfig(medium_at=2.0, high_at=3.0)
    assert severity_from_ratio(1.6, strict) == "Low"


def test_same_change_changes_severity_when_threshold_changes():
    panel = _panel([("2026-07", "A", 100), ("2026-08", "A", 80)])
    sev = lambda thr: run_engine(
        pd.DataFrame({"month": ["2026-07", "2026-08"], "district": ["A", "A"], "x": [100, 80]}),
        "month", "district", ["x"], EngineConfig(change_threshold=thr, entity_corr=False)
    ).insights.query("type == 'trend'").severity.iloc[0]
    assert sev(10) == "High" and sev(19) == "Low"


def test_correlation_severity_thirds():
    assert [severity_from_correlation(r, 0.7) for r in (0.72, 0.85, 0.99)] == ["Low", "Medium", "High"]


# ---------------- Part A: cleaning ---------------- #
def test_cleaning_merges_names_and_drops_exact_duplicates():
    df = pd.DataFrame({"month": ["2026-01"] * 4, "district": ["Surat", " surat ", "RAJKOT", "RAJKOT"],
                       "x": [1.0, 1.0, 2.0, 2.0]})
    clean, log = clean_dataset(df, "month", "district", ["x"])
    assert sorted(clean["_entity"].unique()) == ["Rajkot", "Surat"]
    assert log["exact_duplicates_removed"] == 2 and log["district_names_merged"] == 1


def test_validation_report_has_head_info_missing():
    rep = validation_report(pd.DataFrame({"a": [1, None], "b": ["x", "y"]}), ["a"])
    assert set(rep) >= {"head", "info", "missing"} and rep["missing"].loc["a", "missing_count"] == 1


# ---------------- End to end ---------------- #
def test_dataset_produces_core_insight_types(sample):
    df, d, di, ind = sample
    ins = run_engine(df, d, di, ind, EngineConfig(entity_corr=False)).insights
    assert {"trend", "outlier", "threshold_breach"} <= set(ins.type)


def test_insight_schema_and_ids(sample):
    df, d, di, ind = sample
    ins = run_engine(df, d, di, ind, EngineConfig(entity_corr=False)).insights
    assert list(ins.columns) == INSIGHT_COLUMNS
    assert ins.insight_id.tolist() == [f"INS-{i:04d}" for i in range(1, len(ins) + 1)]
    assert set(ins.type) <= set(INSIGHT_TYPES) and set(ins.severity) <= set(SEVERITY_ORDER)
    assert ins.explanation.str.len().min() > 20


def test_explanation_numbers_come_from_the_data(sample):
    df, d, di, ind = sample
    ins = run_engine(df, d, di, ind, EngineConfig(entity_corr=False)).insights
    for r in ins[ins.type == "trend"].itertuples():
        pct = re.search(r"by ([\d.]+)%", r.explanation).group(1)
        assert float(pct) == pytest.approx(abs((r.value - r.prev_value) / r.prev_value * 100), abs=0.06)


def test_engine_is_general_purpose_no_hardcoded_places():
    for f in [ROOT / "src" / "insight_engine.py", ROOT / "app.py", ROOT / "src" / "cli.py"]:
        text = f.read_text(encoding="utf-8")
        for name in ("Ahmedabad", "Mehsana", "Surat", "Vadodara", "Rajkot", "Bhavnagar"):
            assert name not in text, f"{name} hardcoded in {f.name}"


def test_works_on_arbitrary_schema():
    df = pd.DataFrame({"reporting_month": ["2025-01", "2025-02"] * 3, "location": list("AABBCC"),
                       "malaria_cases": [10, 30, 12, 11, 9, 10]})
    d, di, _, ind = infer_columns(df)
    assert (d, di, ind) == ("reporting_month", "location", ["malaria_cases"])
    ins = run_engine(df, d, di, ind, EngineConfig(entity_corr=False)).insights
    assert any(ins.type == "trend") and "Malaria Cases in A increased by 200.0%" in " ".join(ins.explanation)


def test_bundled_15k_dataset_cleaning():
    df = normalize_column_names(pd.read_csv(ROOT / "data" / "raw_district_healthcare_dataset_15k.csv"))
    d, di, st, ind = infer_columns(df)
    res = run_engine(df, d, di, ind, EngineConfig(entity_corr=False))
    assert res.clean_log["district_names_merged"] == 25 and res.clean_log["exact_duplicates_removed"] == 30
    assert res.panel["_entity"].nunique() == 450
