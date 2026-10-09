# src/

* `insight_engine.py` - all analytics (cleaning, trends, outliers, correlations, threshold breaches,
  severity, insight generation). No UI code; import it from a notebook or script.
* `cli.py` - headless runner: `python -m src.cli --csv data/raw_district_healthcare_dataset_15k.csv --out outputs/demo`
