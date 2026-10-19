import os
import sys
import json
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
RAG_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if RAG_ROOT not in sys.path:
    sys.path.insert(0, RAG_ROOT)

from graph.build_graph import app_graph

DAY4_QUESTIONS = [
    "Which two features best predict conversion ratio?",
    "What's the predicted installation volume in 6 months?",
    "Why does Random Forest underperform the linear model here?",
    "What would a 10pp tariff growth shock do to Gujarat's conversion ratio?",
    "What gaps did prior ML-adoption literature leave open?",
    "Is this conversion-ratio finding causal?"
]

OUTPUT_FILE = os.path.join(CURRENT_DIR, "day4_smoke_test.jsonl")

def run_smoke_test():
    print(f"Running Day 4 Smoke Test on {len(DAY4_QUESTIONS)} questions...")
    results = []

    for idx, q in enumerate(DAY4_QUESTIONS, 1):
        print(f"\n[{idx}/{len(DAY4_QUESTIONS)}] Processing: {q}")
        t0 = time.time()
        config = {"configurable": {"thread_id": f"smoke_{idx}"}}
        state_in = {
            "question": q,
            "retrieved_docs": [],
            "citations": [],
            "caveats": []
        }
        res = app_graph.invoke(state_in, config=config)
        lat = round(time.time() - t0, 3)

        entry = {
            "question": q,
            "route": res.get("route"),
            "answer": res.get("answer"),
            "citations": res.get("citations", []),
            "safety_check_passed": res.get("safety_check_passed", True),
            "caveats": res.get("caveats", []),
            "latency_seconds": lat
        }
        results.append(entry)

        print(f"  Route: {entry['route']}")
        print(f"  Latency: {lat}s")
        print(f"  Safety Passed: {entry['safety_check_passed']}")
        print(f"  Answer preview: {entry['answer'][:150]}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSmoke test complete! Saved {len(results)} records to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_smoke_test()
