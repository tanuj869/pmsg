import os
import sys
from typing import Optional, List, Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
RAG_ROOT = os.path.join(PROJECT_ROOT, "rag-system")
if RAG_ROOT not in sys.path:
    sys.path.insert(0, RAG_ROOT)

from src.duckdb_engine import (
    get_top_states_by_conversion,
    get_state_metrics,
    get_national_summary,
    execute_sql
)
from tools.forecast_tool import (
    predict_conversion_ratio,
    forecast_installations,
    simulate_tariff_shock,
    OFFICIAL_FORECASTS
)
from graph.build_graph import app_graph
from vectorless_rag import get_vectorless_retriever
from src.external_data import fetch_nasa_power_state, fetch_ogd_resource, source_status

app = FastAPI(
    title="PM Surya Ghar — Policy Analytics & Decision Support API",
    description="High-performance REST API powered by FastAPI, DuckDB SQL, LangGraph, and Deterministic Econometric Tools.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    horizon_months: int = Field(6, ge=1, le=36, description="Forecast horizon in months ahead")

class ShockRequest(BaseModel):
    state: Optional[str] = Field("Gujarat", description="Target Indian State or UT")
    shock_percentage_points: float = Field(10.0, ge=1.0, le=50.0, description="Tariff growth shock in percentage points (e.g. 10.0 for +10pp)")

class ChatRequest(BaseModel):
    question: str = Field(..., description="Research, policy, or forecasting query")
    thread_id: Optional[str] = Field("api_session", description="Thread ID for checkpointer state persistence")
    retriever_mode: Literal["dense", "bm25", "hybrid"] = Field("hybrid", description="Dense ChromaDB search, vectorless BM25, or reciprocal-rank hybrid search")

class BM25SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    source_filter: Optional[str] = None

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "title": "PM Surya Ghar Policy Analytics API",
        "status": "online",
        "docs_url": "/docs",
        "version": "1.0.0",
        "engine": "FastAPI + DuckDB + LangGraph"
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "subsystems": {
            "duckdb": "operational",
            "langgraph": "operational",
            "vectorless_bm25": "operational",
            "chroma_vectorstore": "operational"
        }
    }

@app.get("/api/sources/status")
def external_source_status():
    """Reports which live external-data connectors are available/configured."""
    return source_status()

@app.get("/api/external/nasa-power/{state_name}")
def nasa_power_state(state_name: str, start_year: int = Query(2023, ge=1981, le=2100)):
    """Returns monthly NASA POWER solar-radiation and temperature data for a state-capital point."""
    try:
        return fetch_nasa_power_state(state_name, start_year=start_year)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NASA POWER request failed: {e}")

@app.get("/api/external/ogd")
def ogd_resource(limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    """Reads a configured data.gov.in resource (requires DATA_GOV_IN_API_KEY and PMSG_OGD_RESOURCE_ID)."""
    result = fetch_ogd_resource(limit=limit, offset=offset)
    if not result.get("configured"):
        raise HTTPException(status_code=503, detail=result["message"])
    return result

@app.get("/api/analytics/summary")
def get_summary():
    """Returns national high-level adoption and conversion KPIs via DuckDB."""
    try:
        return get_national_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/states")
def list_states(limit: int = Query(36, ge=1, le=50)):
    """Returns all states ranked by conversion ratio via DuckDB SQL."""
    try:
        df = get_top_states_by_conversion(limit=limit)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/states/{state_name}")
def state_profile(state_name: str):
    """Retrieves full covariate profile for a given state via DuckDB."""
    profile = get_state_metrics(state_name)
    if not profile:
        raise HTTPException(status_code=404, detail=f"State '{state_name}' not found.")
    return profile

@app.post("/api/forecast")
def generate_forecast(req: ForecastRequest):
    """Calculates point forecasts and 95% confidence intervals for cumulative national installations."""
    try:
        return forecast_installations(req.horizon_months)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulate-shock")
def simulate_shock(req: ShockRequest):
    """Simulates a counterfactual tariff growth shock on baseline adoption."""
    try:
        shock_decimal = req.shock_percentage_points / 100.0
        return simulate_tariff_shock(state=req.state, shock_pp=shock_decimal)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_agent(req: ChatRequest):
    """
    Submits query to the LangGraph Decision-Support Agent.
    Routes to mathematical tools or grounded corpus retrieval with safety guardrails.
    """
    try:
        config = {"configurable": {"thread_id": req.thread_id}}
        state_in = {
            "question": req.question.strip(),
            "retriever_mode": req.retriever_mode,
            "retrieved_docs": [],
            "citations": [],
            "caveats": []
        }
        res = app_graph.invoke(state_in, config=config)
        return {
            "question": req.question,
            "route": res.get("route"),
            "retriever_mode": req.retriever_mode,
            "answer": res.get("answer"),
            "answer_metadata": res.get("answer_metadata", {}),
            "citations": res.get("citations", []),
            "caveats": res.get("caveats", []),
            "safety_passed": res.get("safety_check_passed", True),
            "forecast_result": res.get("forecast_result"),
            "visualization_data": res.get("visualization_data", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search/bm25")
def search_bm25(req: BM25SearchRequest):
    """Executes Vectorless lexical search with BM25 and metadata filtering."""
    try:
        retriever = get_vectorless_retriever()
        filters = {"source": req.source_filter} if req.source_filter else None
        hits = retriever.search(req.query, top_k=req.top_k, metadata_filters=filters)
        return {"query": req.query, "total_hits": len(hits), "results": hits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/benchmarks")
def get_benchmarks():
    """Returns model benchmark comparison (Linear, Random Forest, XGBoost, LightGBM, SARIMAX)."""
    bench_file = os.path.join(PROJECT_ROOT, "data", "processed", "advanced_forecast_benchmarks.csv")
    if os.path.exists(bench_file):
        import pandas as pd
        df = pd.read_csv(bench_file)
        return df.to_dict(orient="records")
    return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
