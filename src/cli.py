"""Headless runner: prints the Part A report and writes insights + correlation matrix.

    python -m src.cli --csv data/raw_district_healthcare_dataset_15k.csv --out outputs/demo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .insight_engine import (EngineConfig, infer_columns, normalize_column_names,
                             run_engine, validation_report, INSIGHT_COLUMNS)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Auto-Analytics Engine (headless)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--change-threshold", type=float, default=10.0)
    ap.add_argument("--outlier-method", choices=["IQR", "Z-score"], default="IQR")
    ap.add_argument("--outlier-threshold", type=float, default=None,
                    help="IQR multiplier (default 1.5) or |z| (default 3)")
    ap.add_argument("--outlier-scope", choices=["period", "pooled"], default="period")
    ap.add_argument("--min-peer-group", type=int, default=5)
    ap.add_argument("--correlation-threshold", type=float, default=0.70)
    ap.add_argument("--top", type=int, default=0, help="keep only the top-N insights (0 = all)")
    a = ap.parse_args(argv)

    raw = normalize_column_names(pd.read_csv(a.csv))
    date_col, district_col, state_col, indicators = infer_columns(raw)

    rep = validation_report(raw, indicators)
    print("=== head() ===");   print(rep["head"].to_string())
    print("\n=== info() ===");  print(rep["info"])
    print("=== missing values per column ==="); print(rep["missing"].to_string())

    thr = a.outlier_threshold if a.outlier_threshold is not None else (1.5 if a.outlier_method == "IQR" else 3.0)
    cfg = EngineConfig(change_threshold=a.change_threshold, outlier_method=a.outlier_method,
                       outlier_threshold=thr, outlier_scope=a.outlier_scope,
                       min_peer_group=a.min_peer_group, correlation_threshold=a.correlation_threshold)
    res = run_engine(raw, date_col, district_col, indicators, cfg)
    print("\n=== cleaning log ==="); print(res.clean_log)
    for w in res.warnings:
        print("WARNING:", w)

    out = (out / "insights.json").write_text(json.dumps(json.loads(ins.to_json(orient="records")), indent=2), encoding="utf-8")
    ins = res.insights.head(a.top) if a.top else res.insights
    ins[INSIGHT_COLUMNS].to_csv(out / "insights.csv", index=False)
    (out / "insights.json").write_text(json.dumps(json.loads(ins.to_json(orient="records")), indent=2))
    res.corr.to_csv(out / "correlation_matrix.csv")
    print(f"\n{len(res.insights)} insights ({len(ins)} written) -> {out}/")
    print(res.insights.groupby(["type", "severity"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
