import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if RAG_ROOT not in sys.path:
    sys.path.insert(0, RAG_ROOT)

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import PMSGState
from graph.nodes import (
    classify_node,
    retrieve_node,
    forecast_node,
    scenario_node,
    answer_node,
    guardrail_node
)

def build_pmsg_graph(use_checkpointer: bool = True):
    """
    Constructs the decision workflow:
    classify -> (retrieve | forecast | scenario) -> answer -> guardrail -> END
    """
    builder = StateGraph(PMSGState)

    # Register nodes
    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("forecast", forecast_node)
    builder.add_node("scenario", scenario_node)
    builder.add_node("answer", answer_node)
    builder.add_node("guardrail", guardrail_node)

    # Set entry point
    builder.set_entry_point("classify")

    # Conditional routing after classification
    builder.add_conditional_edges(
        "classify",
        lambda state: state["route"],
        {
            "retrieve": "retrieve",
            "forecast": "forecast",
            "scenario": "scenario",
        }
    )

    # Both routes converge to answer node
    builder.add_edge("retrieve", "answer")
    builder.add_edge("forecast", "answer")
    builder.add_edge("scenario", "answer")

    # Answer passes to guardrail node
    builder.add_edge("answer", "guardrail")

    # Guardrail terminates
    builder.add_edge("guardrail", END)

    if use_checkpointer:
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)
    else:
        return builder.compile()

# Global compiled graph
app_graph = build_pmsg_graph(use_checkpointer=True)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test_smoke"}}
    
    # Test Day 3 Routing Query 1
    q1 = "What's the predicted installation volume in 6 months?"
    res1 = app_graph.invoke(
        {"question": q1, "retrieved_docs": [], "citations": [], "caveats": []},
        config=config
    )
    print("=" * 60)
    print("QUERY 1:", q1)
    print("ROUTE:", res1["route"])
    print("ANSWER:\n", res1["answer"])
    print("CITATIONS:", res1["citations"])
    print("SAFETY PASSED:", res1["safety_check_passed"])

    # Test Day 3 Routing Query 2
    q2 = "Why does tariff growth matter more than subsidy level?"
    res2 = app_graph.invoke(
        {"question": q2, "retrieved_docs": [], "citations": [], "caveats": []},
        config=config
    )
    print("=" * 60)
    print("QUERY 2:", q2)
    print("ROUTE:", res2["route"])
    print("ANSWER:\n", res2["answer"])
    print("CITATIONS:", res2["citations"])
    print("SAFETY PASSED:", res2["safety_check_passed"])
