# AEnergy — Precision AI Energy Platform

AEnergy estimates what a home appliance actually adds to your electricity bill. It
pairs a machine-learning engine with Thailand's real **progressive residential
tariff**, wrapped in a holographic-glassmorphism interface that follows your
light/dark preference.

```bash
pip install -r requirements.txt
streamlit run main.py
```

---

## What it does

| Page | What you get |
| --- | --- |
| **⚡ Estimator** | Price one appliance: AI cost with an uncertainty band, consumption, effective ฿/kWh, daily/monthly/yearly horizons, CO₂, tariff-block breakdown, and costed savings advice. |
| **Household** | Build your whole home from presets, see every appliance stacked onto one tier-aware bill, and find your biggest line item. |
| **Insights** | Consumption patterns, honest held-out model diagnostics, feature importance, predicted-vs-actual, and the raw dataset. |
| **Control Center** | Record real bills, bulk-import CSV, edit the dataset in place, retrain, and watch accuracy trend across retrains. |

Highlights:

- **20 appliance presets** — from a 9 W LED bulb to a 7 kW EV charger — so you
  start from realistic numbers instead of a blank form.
- **Costed recommendations.** Every suggested saving is a real what-if run back
  through the engine, not a hard-coded percentage.
- **Uncertainty bands.** Predictions carry a 10th–90th percentile range taken
  across the forest's trees, so you can see where the engine is guessing.
- **CO₂ accounting** using Thailand's grid emission factor (0.4999 kg/kWh).

---

## The prediction engine

### Learning the rate, not the cost

A tree ensemble can only predict a weighted average of its training targets, so
a forest fitted directly on monthly cost **cannot** predict a value above the
most expensive row it has ever seen. Ask the old model about a 7 kW EV charger
when the dataset topped out around ฿1,200 and it confidently answers ฿1,200.

AEnergy learns the **effective tariff rate** in ฿/kWh and multiplies it back by
physically computed energy:

```
cost = kwh(watts, hours, days) × model(features)
```

The rate is bounded and scale-free, which trees handle well, while the kWh term
carries extrapolation exactly. The forest is left to learn only the part that
genuinely needs learning: how the efficiency label, duty cycle and tier position
bend the rate actually paid.

Measured on the bundled dataset, trained on the cheapest 75% of rows and asked
about a 7 kW charger:

| | Prediction | Ground truth |
| --- | --- | --- |
| AEnergy engine | **฿3,351** | ฿3,411 |
| Forest fitted on cost | ฿997 (capped at the training maximum) | ฿3,411 |

### Honest accuracy

The previous implementation scored R² on the very rows it had just fitted, which
is why it always reported ~99%. Every figure the app displays now comes from
k-fold cross-validation on data the fold's model never saw, reported in baht
alongside two baselines:

| Approach | R² | Mean error | Typical error |
| --- | --- | --- | --- |
| **AEnergy engine** | **0.996** | **฿30** | **7.8%** |
| Forest fitted on cost | 0.933 | ฿139 | 72.1% |
| Tariff formula only | 0.912 | ฿147 | 22.9% |

---

## Tariff model

Thai residential tariffs are block-progressive — your last kWh costs more than
your first — so an appliance's true cost depends on what the rest of the home
already draws. AEnergy prices appliances **at the margin**, on top of a
household baseline you set in the sidebar.

- MEA/PEA residential **1.1** (≤150 kWh/month) and **1.2** (>150 kWh/month)
- Ft fuel adjustment and 7% VAT, both configurable
- The fixed monthly service charge is excluded from appliance costs — the
  household pays it whether or not the appliance is plugged in
- A flat ฿4.50/kWh mode is retained for comparison with the legacy estimator

On the Household page, appliances are charged largest-first onto the baseline,
so each sits in the block it genuinely occupies instead of every device being
priced as the household's first kWh.

---

## Project structure

A monorepo-style split: `apps/backend` holds the costing and ML core,
`apps/frontend` holds the Streamlit layer.

```
main.py                       Entry point and navigation
pyproject.toml                Packaging + pytest configuration
apps/
  backend/
    config.py                 Paths, tariff definitions, appliance presets, constants
    tariff.py                 Progressive tariff maths (blocks, bills, marginal cost)
    dataset.py                Observations: load, validate, append, synthesise
    predictor.py              Train, cross-validate, persist, predict
    analysis.py               Emissions, cost horizons, recommendations, rollups
  frontend/
    theme.py                  CSS variables, glassmorphism shell, Plotly template
    state.py                  Cached resources and session state
    components.py             Hero card, stat tiles, recommendation rows
    pages/
      estimator.py            Price one appliance
      household.py            Stack a whole home into one bill
      insights.py             Dataset and model diagnostics
      control_center.py       Collect data, curate, retrain
data/                         Canonical dataset + generated model artefacts
tests/                        56 tests, one file per module
```

```python
from backend.tariff import marginal_cost
from frontend.pages import estimator
```

Each page module is named for the page it renders, matching the sidebar label,
and each test file matches the module it covers.

Two boundaries are enforced by this layout:

- **`backend` imports no Streamlit.** The costing and ML engine is plain Python,
  so it can be unit-tested and reused headlessly. All memoisation lives in
  `frontend/state.py`; the dependency points one way only, frontend into backend.
- **No working-directory paths.** `ROOT_DIR` walks up from the module file, so
  `data/` resolves identically whether you run from the repo root or anywhere
  else.

No install is required — `main.py` puts `apps/` on the path and `pyproject.toml`
does the same for pytest. An editable install is supported if you prefer it:

```bash
pip install -e ".[dev]"
```

---

## Tech stack

Python 3.9+ · Streamlit 1.49+ · scikit-learn · pandas · NumPy · Plotly · joblib

Optional: install `statsmodels` to enable the OLS trendline on the Insights
scatter plot.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

56 tests: tariff block arithmetic and additivity, schema validation and atomic
writes, model extrapolation and persistence, savings monotonicity, and
end-to-end render checks for all four pages.

---

## Notes on paths

Data and model artefacts live in `data/`, anchored to the repository root.
Earlier versions used bare relative filenames, so launching from different
directories silently read and wrote different datasets and different models. An
existing `collected_data.csv` is migrated into `data/` automatically on first
run, including one left behind in a pre-refactor `front-end/` directory.

Estimates are modelling output, not a billing statement.
