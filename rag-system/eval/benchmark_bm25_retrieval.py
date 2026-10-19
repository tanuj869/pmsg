"""Measure vectorless retrieval against the labeled benchmark without LLM calls."""
import json
from pathlib import Path
import pandas as pd
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "eval" / "eval_set.jsonl"
OUT_PATH = ROOT / "eval" / "bm25_retrieval_benchmark.csv"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vectorless_rag import get_vectorless_retriever


def run() -> pd.DataFrame:
    items = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    retriever = get_vectorless_retriever()
    rows = []
    for item in items:
        expected = item.get("gold_chunk_id")
        hits = retriever.search(item["question"], top_k=5)
        hit_ids = [str(hit.get("id")) for hit in hits]
        rows.append({
            "id": item["id"],
            "type": item.get("type"),
            "question": item["question"],
            "gold_chunk_id": expected,
            "top5_hit": bool(expected and expected in hit_ids),
            "rank": hit_ids.index(expected) + 1 if expected in hit_ids else None,
            "retrieved_ids": ";".join(hit_ids),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT_PATH, index=False)
    gold = result[result["gold_chunk_id"].notna()]
    print(f"BM25 Top-5 hit rate: {gold['top5_hit'].mean():.1%} ({gold['top5_hit'].sum()}/{len(gold)})")
    print(f"Saved per-query results: {OUT_PATH}")
    return result


if __name__ == "__main__":
    run()
