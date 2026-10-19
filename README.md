# PM Surya Ghar — Data Science Project

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install pandas openpyxl scikit-learn shap matplotlib jupyter ipykernel
```

Open this folder in VS Code, install the Jupyter extension if not already
installed, and open the notebooks in `notebooks/` in order.

## Project structure

```
pmsg-project/
├── data/
│   ├── raw/            10 verified Tier 1 source files — DO NOT EDIT these
│   └── processed/       Output of 01_panel_build.ipynb (generated, not manual)
├── notebooks/
│   ├── 01_panel_build.ipynb    Ingests raw/, builds the canonical panel
│   ├── 02_eda.ipynb            Exploratory analysis, correlation, charts
│   └── 03_modeling.ipynb       Model comparison (Linear vs Random Forest), SHAP
├── src/
│   └── state_mapping.py        Canonical 36-state list + name normalization
└── README.md
```

Run the notebooks **in order** — each depends on the previous one's output
in `data/processed/`.

## What's already verified (do not re-collect)

The original 10 scheme and covariate files in `data/raw/` were checked individually before being merged:
totals cross-verified against official sources, dates confirmed against
Rajya Sabha session records, and one real bug (a "Telengana" typo in the
irradiance file) was caught and fixed in `src/state_mapping.py`.

A supplemental MoHFW population estimate file is now included for denominator
normalization; it does not replace or modify the original 10 verified sources.

**Deliberately excluded, and why:** Net-metering regime (Tier 1 #7b) and
state-level subsidy top-up (#8) — no centralized source exists for either
(would require 28 separate SERC/nodal-agency lookups), and the structurally
similar ToD dataset showed only 1 of 35 states has genuine domestic-consumer
coverage, suggesting weak expected signal even if compiled.

## Key finding so far

A simple 2-feature linear model (tariff growth % + solar irradiance)
predicts `target_conversion_ratio` (installations ÷ applications) better
than a 6-feature Random Forest, evaluated with Leave-One-Out Cross-Validation:

| Model | LOO-CV MAE | vs. Naive baseline |
|---|---|---|
| Linear (tariff growth + irradiance) | 0.135 | **-20.7%** |
| Random Forest (6 features) | 0.161 | -5.9% |

Subsidy/tariff *level* and raw solar potential are consistently the
**weakest** predictors across every method tried — direct evidence against
"subsidy amount alone" explaining PM Surya Ghar adoption variation across
states, consistent with the literature's "subsidy paradox" finding.

**This is correlational, not causal**, and n=28-35 states is a small sample —
report accordingly.

## Coverage, population normalization, and validation

- `data/processed/dataset_register.csv` and `data/processed/data_dictionary.csv`
  are generated from the raw and processed tables by `src/build_data_catalog.py`.
- `data/processed/modeling_table_population_normalized.csv` adds applications
  and installations per lakh population using MoHFW's 1 July 2021 State/UT
  estimates. Scheme outcomes are from August 2026, so the five-year denominator
  gap is explicit; the merged Dadra and Nagar Haveli and Daman and Diu row has
  population but no matching scheme outcome and remains blank.
- `src/walk_forward_validation.py` evaluates naive, linear-trend, and log-linear
  forecasts over expanding-window one-step origins. With only six dated national
  observations there are three validation origins: linear trend MAPE is 13.75%,
  naive MAPE is 22.03%, and log-linear MAPE is 36.71%. These estimates are too
  uncertain for a confident model ranking. Run the script again when a new
  official national observation is added.

## Live source connections

- NASA POWER monthly solar radiation and temperature can be fetched by state
  capital point from the dashboard's **Data Sources & Live Data** tab or
  `/api/external/nasa-power/{state_name}`. A capital point is a location proxy,
  not a spatial average over the State/UT.
- The agent can retrieve a configured data.gov.in resource for latest/current
  PM Surya Ghar questions when `DATA_GOV_IN_API_KEY` and
  `PMSG_OGD_RESOURCE_ID` are set in the server environment. Check
  `/api/sources/status` to see which connectors are ready.
- NITI Aayog ICeD potential and the project's other covariates are retained as
  dated local source snapshots. No undocumented endpoint is treated as a live
  API. Net-metering and State subsidy top-ups remain unsupported.

## Forecasting (`04_forecasting.ipynb`)

Forecasts national installations 3/6/12 months ahead from 6 dated points
(Nov 2024 – Aug 2026). Compared naive, linear, and log-linear (exponential)
extrapolation — log-linear fit best (4.98% in-sample MAPE vs. 16.83% linear)
since cumulative adoption compounds during a scheme's ramp-up phase.

| Horizon | Point Forecast | 95% Interval |
|---|---|---|
| +3 months | ~5.24M | 4.62M – 5.95M |
| +6 months | ~6.84M | 6.03M – 7.75M |
| +12 months | ~11.61M | 10.24M – 13.17M |

**Read the caveats in the notebook before citing these** — 6 points over 18
months is a thin basis for any method, and the exponential assumption will
overstate long-horizon forecasts if the real-world growth rate slows
(policy changes, budget exhaustion, market saturation).

## Scenario Simulation (`05_scenario_simulation.ipynb`)

Addresses Step 7. Only 3 of the guideline's 5 scenarios are genuinely
supportable with the data actually collected — the notebook is explicit
about which and why, rather than faking results for scenarios the model
can't support:

| Scenario | Supported? |
|---|---|
| A — Base case | ✅ Yes |
| B — Subsidy +10% | ❌ No — Tier 1 #8 was deliberately scoped out, model has no subsidy feature |
| C — High-potential targeting | ✅ Yes (data-driven, found 4 states: Gujarat, Karnataka, Chhattisgarh, Andhra Pradesh) |
| D — Tariff increase | ✅ Yes — uses the fitted tariff_growth coefficient directly |
| E — Combined | ⚠️ Partial — tariff component only |

**Key honest finding:** because the model is linear with no interaction
terms, Scenario D's predicted lift is identical (~+16pp) for every state —
flagged in the notebook as a modeling limitation, not a real-world claim.

All scenario results are explicitly labeled *"predicted under model
assumptions,"* not causal, per guideline Section 7.

## RAG Decision-Support System (`rag-system/`)

A LangGraph decision-support system grounded in the State/UT empirical panel and 31 literature review papers:
- **LangChain orchestration:** a LangChain Expression Language prompt chain passes retrieved evidence or deterministic tool results to the local Ollama endpoint.
- **Local Inference:** Ollama serving `llama3.2:1b` (zero paid API keys required).
- **Retrieval:** Persistent ChromaDB vector store embedded with `sentence-transformers/all-MiniLM-L6-v2`.
- **Vectorless Retrieval:** BM25 lexical search is available through the API; chat requests can select dense, BM25, or reciprocal-rank hybrid retrieval.
- **Deterministic Tools:** Direct execution of econometric regressions and log-linear national forecasts to prevent mathematical hallucinations.
- **Safety Guardrails:** Automatic enforcement of causal disclaimers, subsidy exclusion checks, numerical verification, and out-of-scope filtering.
- **Interactive UI:** Streamlit web interface (`rag-system/app.py`).

### Evaluation Highlights (18-Query Benchmark)
- **Retrieval Accuracy (Top-3):** 76.9%
- **Weighted Correctness:** 77.8%
- **Wrong Answers / Hallucinations:** 0.0%
- **Guardrail Trigger Rate:** 75.0%
- **Mean Query Latency:** 19.02s (Local CPU)

For full details, see [`rag-system/README.md`](rag-system/README.md).

## Final Deliverables

- **`PMSG_Research_Report_Final.pdf`**: Publication-ready research paper (571 KB) incorporating Section 7 (RAG Decision-Support System), Table 3 (evaluation metrics), and Figure 6 (architecture).
- **`PMSG_Research_Report_Final.docx`**: Word document source with all updated sections, styles, and high-resolution figures.
- **`figure6_architecture.png`**: High-resolution 300 DPI system architecture diagram.
- **`rag-system/app.py`**: Streamlit application for interactive decision support and scenario exploration.
