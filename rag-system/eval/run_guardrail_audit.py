"""Deterministic safety audit for scope, citations, causal wording and numbers."""
from pathlib import Path
import pandas as pd

import os
import sys
RAG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if RAG_ROOT not in sys.path:
    sys.path.insert(0, RAG_ROOT)
from graph.nodes import guardrail_node

OUT = Path(__file__).resolve().parent / "guardrail_audit_results.csv"

CASES = [
    {
        "case": "grounded numeric claim",
        "state": {"route": "retrieve", "question": "What does the study report?", "answer": "The reported MAE is 0.135.", "citations": ["study: model results"], "retrieved_docs": [{"id": "study", "title": "model results", "text": "The reported MAE is 0.135.", "similarity": 0.7}]},
        "expected_pass": True,
        "expected_withheld": False,
    },
    {
        "case": "unsupported numeric claim",
        "state": {"route": "retrieve", "question": "What does the study report?", "answer": "The model reached 98.7% accuracy.", "citations": ["study: model results"], "retrieved_docs": [{"id": "study", "title": "model results", "text": "The reported MAE is 0.135.", "similarity": 0.7}]},
        "expected_pass": False,
        "expected_withheld": True,
    },
    {
        "case": "out-of-scope retrieval",
        "state": {"route": "retrieve", "question": "Forecast installations on Mars", "answer": "There will be 500 installations.", "citations": ["x: y"], "retrieved_docs": [{"id": "x", "title": "y", "text": "Solar adoption in Indian states.", "similarity": 0.05}]},
        "expected_pass": False,
        "expected_withheld": True,
    },
    {
        "case": "missing citation",
        "state": {"route": "retrieve", "question": "What do the sources say?", "answer": "The report discusses tariff growth.", "citations": [], "retrieved_docs": [{"id": "x", "title": "y", "text": "The report discusses tariff growth.", "similarity": 0.7}]},
        "expected_pass": False,
        "expected_withheld": True,
    },
    {
        "case": "unqualified causal claim",
        "state": {"route": "retrieve", "question": "Does tariff growth cause adoption?", "answer": "Higher tariff growth causes adoption.", "citations": ["x: y"], "retrieved_docs": [{"id": "x", "title": "y", "text": "The relationship is correlational.", "similarity": 0.7}]},
        "expected_pass": False,
        "expected_withheld": False,
    },
]


def run() -> pd.DataFrame:
    rows = []
    for case in CASES:
        result = guardrail_node(case["state"])
        passed = result["safety_check_passed"]
        withheld = "withheld" in result["answer"].lower() or "outside the scope" in result["answer"].lower()
        rows.append({
            "case": case["case"],
            "expected_pass": case["expected_pass"],
            "actual_pass": passed,
            "expected_withheld": case["expected_withheld"],
            "actual_withheld": withheld,
            "passed": passed == case["expected_pass"] and withheld == case["expected_withheld"],
            "caveats": " | ".join(result["caveats"]),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False)
    print(frame.to_string(index=False))
    print(f"\nGuardrail audit: {frame['passed'].sum()}/{len(frame)} cases passed.")
    return frame


if __name__ == "__main__":
    run()
