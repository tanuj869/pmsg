# PM Surya Ghar (PMSG) RAG Decision-Support System

A fully local, self-hosted Decision-Support System and Retrieval-Augmented Generation (RAG) pipeline designed for policymakers, grid operators, and academic researchers analyzing the PM Surya Ghar: Muft Bijli Yojana rooftop solar scheme across 36 Indian States and Union Territories.

---

## 🏛️ System Architecture

The system is built as a state machine using **LangGraph** with local inference via **Ollama (`llama3.2:1b`)**, deterministic mathematical tools, and strict safety guardrails.

The answer stage uses a **LangChain Expression Language** prompt chain to pass retrieved corpus evidence or deterministic forecast results to the local Ollama REST endpoint. Retrieval can use dense ChromaDB search, vectorless BM25, or reciprocal-rank hybrid search. The API's `/api/chat` accepts `retriever_mode` as `dense`, `bm25`, or `hybrid` (default).

```
                    ┌─────────────────────────┐
                    │   User Input / Query    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      classify_node      │
                    │   (Keyword / Pattern)   │
                    └──────┬───────────┬──────┘
             "forecast"    │           │ "retrieve"
                           ▼           ▼
        ┌───────────────────────┐   ┌───────────────────────────┐
        │     forecast_node     │   │       retrieve_node       │
        │ (Fitted Linear Model, │   │    (ChromaDB Persistent,  │
        │ Log-Linear Fit, MAPE) │   │    all-MiniLM-L6-v2)      │
        └───────────┬───────────┘   └─────────────┬─────────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────┐
                    │       answer_node       │
                    │ (Local Ollama llama3.2, │
                    │ deterministic fallback) │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     guardrail_node      │
                    │ (4 Deterministic Checks)│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Final Verified Answer  │
                    │  & Grounded Citations   │
                    └─────────────────────────┘
```

### The 5 Nodes:
1. **`classify` (`classify_node`)**: Inspects user query with prioritized regex and keyword patterns. Mathematical / scenario queries route to `forecast`. Qualitative, empirical, or literature questions route to `retrieve`.
2. **`retrieve` (`retrieve_node`)**: Embeds query using `sentence-transformers/all-MiniLM-L6-v2` and queries ChromaDB vectorstore (`k=3`, persistence at `rag-system/chroma_db/`). Chunks derive from the 31-paper literature review and 17 subsections of the research paper (`corpus.jsonl`).
3. **`forecast` (`forecast_node`)**: Deterministic execution of mathematical models:
   - Conversion ratio prediction: $\widehat{\text{Ratio}} = 0.0805 + 1.6033 \times (\Delta\text{Tariff}) + 0.0210 \times \text{Irradiance}$.
   - National installation forecasts: log-linear extrapolation (+3mo: 5.24M, +6mo: 6.84M, +12mo: 11.61M; in-sample MAPE 4.98%).
   - Tariff shock scenario simulation: calculates baseline vs. shocked conversion ratio with exact delta.
4. **`answer` (`answer_node`)**: Prompts local `llama3.2:1b` (via `http://localhost:11434/api/generate`) with strict zero-shot grounded instructions and context chunks, backed by a deterministic fallback generator.
5. **`guardrail` (`guardrail_node`)**: Deterministic safety checks:
   - **Subsidy Exclusion (Scenario B)**: Blocks unsupported claims regarding state subsidy top-up.
   - **Causal Language Gate**: Enforces correlational disclaimers on causal claims.
   - **Numerical Grounding**: Verifies generated numbers against context or tool outputs.
   - **Out-of-Scope Gate**: Filters queries with low vector similarity (< 0.28).

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **OS**: Windows 10/11, Linux, or macOS.
- **Python**: Python 3.12 (recommended for PyTorch/ChromaDB compatibility).
- **Ollama**: Installed and serving locally (`http://localhost:11434`).

### 2. Environment Setup
```powershell
# Navigate to the project root
cd "C:\Users\shwet\Downloads\pmsg-project (3)\pmsg-project"

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Verify Ollama daemon is running with llama3.2:1b
ollama list
# If not running:
# Start-Process -NoNewWindow "ollama" "serve"
# ollama pull llama3.2:1b
```

Optional live Open Government Data retrieval requires `DATA_GOV_IN_API_KEY` and
`PMSG_OGD_RESOURCE_ID` in the server environment. NASA POWER requests are public
and do not require a key. The **Data Sources & Live Data** dashboard tab reports
connector status and fetches NASA POWER only when requested.

### 3. Launching the Interactive Streamlit UI
```powershell
streamlit run rag-system/app.py
```
This opens `http://localhost:8501` in your browser. The UI includes:
- Interactive query bar with example query selector.
- Execution routing indicator (`[forecast]` vs `[retrieve]`).
- Clean markdown answers with collapsible context chunks and citations.
- Interactive Parameter Explorer for live state-level tariff shock simulations.

---

## Evaluation and benchmarking

Current reproducible component checks are kept separate so they are not
mistaken for a full semantic evaluation of the language model:

| Evaluation | Current result | Scope |
|---|---:|---|
| Vectorless BM25 retrieval | 82.4% Top-5 hit rate (14/17 labeled queries) | Gold-chunk retrieval only |
| Deterministic guardrail audit | 5/5 cases passed | Grounded number, ungrounded number, out-of-scope, missing citation, causal wording |
| National forecast backtest | 3 expanding-window origins | Naive, linear trend, and log-linear; six observations total |

Run the component evaluations with `python rag-system/eval/benchmark_bm25_retrieval.py`,
`python rag-system/eval/run_guardrail_audit.py`, and
`python src/walk_forward_validation.py`. The older 18-query LLM benchmark remains
in `rag-system/eval/`, but its answer-scoring and hallucination fields are lexical
proxies and should not be presented as human-validated semantic accuracy or a
zero-hallucination guarantee. A full answer-quality evaluation requires a running
Ollama model and human-reviewed labels.

---

## 📂 Deliverables & Repository Structure

```
pmsg-project/
├── PMSG_Research_Report_Final.docx  # Full paper updated with Section 7 (RAG system), Table 3, Figure 6
├── PMSG_Research_Report_Final.pdf   # Publication-ready PDF export (571 KB)
├── figure6_architecture.png         # Vector architecture flowchart (300 DPI)
├── README.md                        # Master repository documentation
├── rag-system/
│   ├── app.py                       # Streamlit interactive application
│   ├── generate_figure6.py          # Script generating Figure 6 architecture diagram
│   ├── update_report.py             # Script injecting Section 7 into research paper docx
│   ├── chroma_db/                   # Persistent vector store (48 embedded chunks)
│   ├── corpus/
│   │   └── corpus.jsonl             # Ingested chunks from 31 papers + 17 report sections
│   ├── eval/
│   │   ├── eval_set.jsonl           # 18 benchmark test questions
│   │   ├── results.jsonl            # Raw evaluation logs with latency & scores
│   │   └── results_summary.csv      # Metric aggregates
│   ├── graph/
│   │   ├── state.py                 # PMSGState TypedDict definition
│   │   ├── nodes.py                 # 5 LangGraph workflow nodes & guardrails
│   │   └── build_graph.py           # Compiled StateGraph with MemorySaver
│   ├── ingestion/
│   │   ├── export_litreview.py      # Literature matrix extractor (31 papers)
│   │   ├── export_report.py         # Research report chunk extractor
│   │   ├── build_vectorstore.py     # ChromaDB indexing script
│   │   └── test_retrieval.py        # Vector retrieval unit tests
│   └── tools/
│       └── forecast_tool.py         # Conversion ratio & log-linear forecast functions
└── venv/                            # Python 3.12 virtual environment
```

---

## 🛡️ Responsible AI & Citations
- **Correlational Disclaimers**: All regression outputs explicitly indicate that findings represent correlational patterns across $n=28\text{--}35$ states, not verified causal mechanisms.
- **Subsidy Boundary**: The system deterministically blocks hallucinated recommendations for state subsidy increases, noting that state subsidy top-up (Tier 1 #8) was deliberately excluded from the underlying econometric specification.
- **Citations**: Citations in retrieved answers reference the primary sources extracted from the literature review matrices (e.g. Shrimali 2018, CEEW 2023, Bridge to India 2023).
