import os
import sys
import json
import time
import re
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAG_ROOT = os.path.join(PROJECT_ROOT, "rag-system")
if RAG_ROOT not in sys.path:
    sys.path.insert(0, RAG_ROOT)

from graph.build_graph import app_graph
from vectorless_rag import get_vectorless_retriever

EVAL_SET_PATH = os.path.join(RAG_ROOT, "eval", "eval_set.jsonl")
OUT_JSON_PATH = os.path.join(RAG_ROOT, "eval", "ragas_evaluation_report.json")
OUT_CSV_PATH = os.path.join(RAG_ROOT, "eval", "ragas_metrics_summary.csv")

def calculate_token_overlap_relevancy(answer: str, question: str) -> float:
    """Calculates semantic/token overlap proxy for Answer Relevancy."""
    q_words = set(re.findall(r"\w+", question.lower()))
    a_words = set(re.findall(r"\w+", answer.lower()))
    if not q_words:
        return 0.0
    overlap = len(q_words.intersection(a_words))
    return min(1.0, round(overlap / len(q_words), 3))

def calculate_faithfulness(answer: str, context_chunks: list[dict], caveats: list[str]) -> float:
    """
    Evaluates faithfulness / grounding score:
    Verifies that claims and numbers in the answer originate directly from retrieved context or tool calculation.
    """
    if not answer:
        return 0.0
    
    # If guardrail triggered or flagged a caveat, penalize ungrounded hallucination
    if any("unsupported" in c.lower() for c in caveats):
        return 0.50
    
    # Check for hallucinated numbers
    numbers = re.findall(r"(?<![\w])\d[\d,]*(?:\.\d+)?%?(?![\w])", answer)
    context_text = " ".join(
        f"{c.get('title', '')} {c.get('text', '')}" for c in context_chunks
    )

    def normalize(value: str) -> str:
        return value.replace(",", "").rstrip("%")

    verified_nums = sum(1 for num in numbers if normalize(num) in {
        normalize(x) for x in re.findall(r"(?<![\w])\d[\d,]*(?:\.\d+)?%?(?![\w])", context_text)
    })

    if not numbers:
        return 1.0
    return round(verified_nums / len(numbers), 3)

def run_evaluation_suite():
    print(f"Loading benchmark eval set from {EVAL_SET_PATH}...")
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"Evaluating {len(items)} queries across Ragas & Safety metrics...")
    results = []
    latencies = []
    faithfulness_scores = []
    relevancy_scores = []
    precision_hits = []

    bm25 = get_vectorless_retriever()

    for idx, item in enumerate(items, 1):
        q_id = item["id"]
        q_type = item["type"]
        question = item["question"]
        gold_chunk = item.get("gold_chunk_id")
        expected_route = item.get("expected_route")

        t0 = time.time()
        config = {"configurable": {"thread_id": f"eval_{q_id}"}}
        state_in = {
            "question": question,
            "retrieved_docs": [],
            "citations": [],
            "caveats": []
        }
        res = app_graph.invoke(state_in, config=config)
        lat = round(time.time() - t0, 3)
        latencies.append(lat)

        answer = res.get("answer", "")
        route = res.get("route", "")
        docs = res.get("retrieved_docs", [])
        caveats = res.get("caveats", [])
        passed = res.get("safety_check_passed", True)
        citations = res.get("citations", [])

        # Context Precision / Hit Rate
        hit = 1.0 if (gold_chunk and any(d["id"] == gold_chunk for d in docs)) else 0.0
        if gold_chunk:
            precision_hits.append(hit)

        # Vectorless BM25 comparison check
        bm25_hits = bm25.search(question, top_k=3)
        bm25_hit = 1.0 if (gold_chunk and any(b["id"] == gold_chunk for b in bm25_hits)) else 0.0

        grounding_context = list(docs)
        if res.get("forecast_result"):
            grounding_context.append({
                "id": "deterministic_forecast_result",
                "title": "Deterministic forecast tool output",
                "text": json.dumps(res["forecast_result"], ensure_ascii=False),
            })
        faith = calculate_faithfulness(answer, grounding_context, caveats)
        relevancy = calculate_token_overlap_relevancy(answer, question)
        doc_refs = [f"{d.get('id', '')}: {d.get('title', '')}".strip() for d in docs]
        valid_citations = bool(citations) and (
            (route == "forecast" and any("model" in c.lower() or "forecast" in c.lower() or "scenario" in c.lower() for c in citations))
            or any(any(ref and (ref in citation or ref.split(":", 1)[0] in citation) for ref in doc_refs) for citation in citations)
        )
        unsupported_numeric = faith < 1.0 and bool(re.findall(r"(?<![\w])\d[\d,]*(?:\.\d+)?%?(?![\w])", answer))

        faithfulness_scores.append(faith)
        relevancy_scores.append(relevancy)

        results.append({
            "id": q_id,
            "type": q_type,
            "question": question,
            "route_predicted": route,
            "route_expected": expected_route,
            "route_correct": route == expected_route,
            "gold_chunk_id": gold_chunk,
            "retriever_hit": hit,
            "bm25_retriever_hit": bm25_hit,
            "faithfulness": faith,
            "citation_present": bool(citations),
            "citation_valid_proxy": bool(valid_citations),
            "unsupported_numeric_claim_proxy": bool(unsupported_numeric),
            "answer_relevancy": relevancy,
            "guardrail_passed": passed,
            "latency_sec": lat,
            "answer_sample": answer[:120] + "..."
        })

    summary = {
        "total_queries": len(items),
        "mean_faithfulness_score": round(float(np.mean(faithfulness_scores)), 4),
        "mean_answer_relevancy_score": round(float(np.mean(relevancy_scores)), 4),
        "context_precision_top5_rate": round(float(np.mean(precision_hits)) if precision_hits else 0.0, 4),
        "mean_latency_sec": round(float(np.mean(latencies)), 3),
        "p95_latency_sec": round(float(np.percentile(latencies, 95)), 3),
        "guardrail_trigger_rate_adversarial": round(
            sum(1 for r in results if r["type"] == "adversarial" and not r["guardrail_passed"]) /
            max(1, sum(1 for r in results if r["type"] == "adversarial")), 3
        ),
        "citation_coverage_rate": round(float(np.mean([r["citation_present"] for r in results])), 4),
        "citation_validity_proxy_rate": round(float(np.mean([r["citation_valid_proxy"] for r in results])), 4),
        "unsupported_numeric_claim_query_rate": round(float(np.mean([r["unsupported_numeric_claim_proxy"] for r in results])), 4),
        "metric_note": "Faithfulness, relevance and citation checks are transparent lexical proxies; they are not human-validated semantic metrics."
    }

    # Save outputs
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "evaluations": results}, f, indent=2)

    df_sum = pd.DataFrame([summary])
    df_sum.to_csv(OUT_CSV_PATH, index=False)

    print(f"\n--- Evaluation Summary Saved to {OUT_JSON_PATH} ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary

if __name__ == "__main__":
    run_evaluation_suite()
