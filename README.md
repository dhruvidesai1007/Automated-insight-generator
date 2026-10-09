# Auto-Analytics Engine for District Healthcare

Assignment 4 - *Automated Insight Generation*. A general-purpose engine that ingests a district-level
healthcare CSV and automatically finds **trends, outliers, correlations and threshold breaches**, then
explains each one as a structured, human-readable insight. Python + pandas/SciPy backend, Streamlit UI.

> The bundled `data/raw_district_healthcare_dataset_15k.csv` is **synthetic** raw-style data for testing.
> It is not official HMIS data and must not be used for real healthcare or policy decisions.

## Run

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                                    # interactive UI
pytest -q                                               # 21 unit tests
python -m src.cli --csv data/raw_district_healthcare_dataset_15k.csv --out outputs/demo   # headless, no UI
```

Python 3.10+ (developed and tested on 3.12, pandas 3.0, streamlit 1.65).

## Project layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI: filters, sliders, insight list, severity chart, heatmap, line chart, validation tab |
| `src/insight_engine.py` | All analytics logic (pure functions, unit-tested) |
| `src/cli.py` | Headless run: prints head()/info()/missing report, writes CSV + JSON |
| `tests/test_engine.py` | 21 tests (formulas, thresholds, severity, schema, factuality, no hard-coded names) |
| `data/` | 15k synthetic dataset |
| `outputs/` | `insights.csv`, `insights.json`, `correlation_matrix.csv` for the 15k dataset |
| `screenshots/` | UI screenshots |

## How it maps to the assignment

| Part | Where | Notes |
|---|---|---|
| A - load + validate + filters | `validation_report`, *Data & validation* tab, sidebar filters | head(), info(), missing count per column, range checks, cleaning log. Live filters: district, month range, indicator (+ severity / type). |
| B - trends | `compute_trends`, `trend_insights` | `pct_change = (current − previous) / previous × 100`; `is_significant` when `|pct_change| ≥ threshold` (default 10 %, slider). Only the immediately preceding calendar month is used by default (toggle available). |
| C - outliers | `outlier_insights` | IQR (k slider, default 1.5) or Z-score (default 3). Evaluated **per indicator, per month** (cross-section), optionally within the same state. Listed with district, indicator, month, value. |
| D - correlation | `correlation_matrix`, `correlation_insights` | Pearson via `DataFrame.corr()`; pairs with `|r| ≥ 0.70` (slider) flagged; also per-district over time when ≥ N months exist. |
| E - insights | `finalize_insights` | Fields: `insight_id, type, indicator, entity, period, metric, value, prev_value, change_pct, change, severity, score, explanation`. Types: `trend, outlier, correlation, threshold_breach`. |
| F - UI | `app.py` | Insight list, severity-count bar chart, type chart, correlation heatmap, per-district line chart. Every threshold is a slider. |

## Insight generation (no hard-coded narratives)

Sentences are templates; every number, name and period is read from the data. Example output shape (see `outputs/synthetic_15k/insights.csv` for real rows):

* *High Risk Cases in District_430 increased by 2200.0% compared to the previous month (1 -> 23), exceeding the 10% significant-change threshold.* - **High**
* *High Risk Cases in District_352 was 1,723, 291.6% above the maximum limit of 440.* - **High**

For outliers `prev_value` holds the peer mean and `change_pct` the deviation from it; for breaches
`prev_value` is the limit; for correlations `value` is `r`.

### Severity is derived from the configurable thresholds

Severity is the number of **multiples of the threshold being tested** that an observation reaches:

| Insight | ratio |
|---|---|
| trend | `|pct_change| / change_threshold` |
| outlier (IQR) | distance beyond the nearest quartile ÷ IQR ÷ k (1.0 = exactly on the fence) |
| outlier (Z) | `|z| / z_threshold` |
| threshold breach | gap to limit (%) ÷ change_threshold |

`Low < 1.25×`, `Medium 1.25–1.5×`, `High ≥ 1.5×` - both cut-offs are sliders. Correlation severity splits the
range between the `|r|` threshold and 1.0 into thirds. Changing a threshold therefore changes severities;
there are no constants like `if change > 50`.

## Limitations (please read)

* **Correlations are fragile on small data.** With 2 months × 6 districts the matrix is built from 12 rows and
  `|r|` near 1 is easy to get by chance. The UI warns when there are < 3 months or < 10 districts, and every
  correlation insight states its sample size. Per-district correlation needs ≥ 12 months (slider); it is skipped
  on short datasets - the matrix is shown anyway, honestly.
* **Z-score cannot fire on tiny groups.** A population z-score is bounded by `(n−1)/√n` ≈ 2.04 for n = 6, so `|z| ≥ 3`
  is impossible; use IQR (default) for small peer groups. IQR on ~6 peers is also very sensitive - e.g. a 5.6 % deviation
  can be an "outlier" when peers are tightly clustered. Raise *Minimum districts in a comparison group* to suppress this.
* **Percent change on small counts explodes** (e.g. 1 → 23 = +2,200 %). Review the raw values in the explanation.
* Testing many districts × indicator pairs yields some chance findings; flags are prompts for review, not diagnoses.
* Correlation is not causation.
* Default threshold-breach limits are data-derived (5th-percentile tail; low tail for percentage-like indicators, high tail for
  counts) and editable in the sidebar - they are *not* clinical targets.
* The UI defaults to the latest month; widen the *Months* filter to review history (15k dataset: ~14.5k insights over 36 months).

## Data handling (15k dataset)

The engine cleans a copy and logs what it did (never modifies the upload): 25 district spellings merged
(whitespace/case, 475 → 450 districts), 30 exact duplicate rows removed, ~1 % missing values per indicator left as
missing (not imputed), `high_risk_cases` correctly not treated as a percentage.
