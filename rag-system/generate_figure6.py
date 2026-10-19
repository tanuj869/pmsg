import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figure6_architecture.png"))

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Colors
    c_blue = "#1E40AF"
    c_light_blue = "#DBEAFE"
    c_green = "#065F46"
    c_light_green = "#D1FAE5"
    c_amber = "#92400E"
    c_light_amber = "#FEF3C7"
    c_purple = "#5B21B6"
    c_light_purple = "#EDE9FE"
    c_gray = "#1F2937"
    c_light_gray = "#F3F4F6"

    # Box styles
    box_props = dict(boxstyle="round,pad=0.5", facecolor=c_light_blue, edgecolor=c_blue, linewidth=2)
    box_retrieve = dict(boxstyle="round,pad=0.5", facecolor=c_light_green, edgecolor=c_green, linewidth=2)
    box_forecast = dict(boxstyle="round,pad=0.5", facecolor=c_light_amber, edgecolor=c_amber, linewidth=2)
    box_scenario = dict(boxstyle="round,pad=0.5", facecolor="#FFE4E6", edgecolor="#BE123C", linewidth=2)
    box_answer = dict(boxstyle="round,pad=0.5", facecolor=c_light_purple, edgecolor=c_purple, linewidth=2)
    box_guardrail = dict(boxstyle="round,pad=0.5", facecolor="#FEE2E2", edgecolor="#B91C1C", linewidth=2)

    # Node 1: Input Question
    ax.text(1.0, 3.5, "User Query\n(Natural Language)", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor="#6B7280", linewidth=1.5))

    # Node 2: Classify Node
    ax.text(3.2, 3.5, "1. Classify\n(Question Router)", ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=box_props)

    # Node 3A: Retrieve Node (Top branch)
    ax.text(6.4, 5.5, "2A. Retrieve\n• Dense, BM25 or hybrid search\n• Project corpus + live NASA POWER\n• Optional live data.gov.in records",
            ha="center", va="center", fontsize=8.5, fontweight="bold", bbox=box_retrieve)

    # Node 3B: Forecast Node (Bottom branch)
    ax.text(6.4, 3.5, "2B. Forecast\n• Deterministic model tools\n• National forecast and intervals\n• Tariff shock calculations",
            ha="center", va="center", fontsize=8.5, fontweight="bold", bbox=box_forecast)

    # Node 3: Scenario Node
    ax.text(6.4, 1.5, "2C. Scenario\n• Compare scenario to baseline\n• High-potential targeting\n• Unsupported subsidy scenario is blocked",
            ha="center", va="center", fontsize=8.5, fontweight="bold", bbox=box_scenario)

    # Node 4: Answer Node
    ax.text(9.8, 3.5, "3. Answer\n• LangChain prompt chain\n• Local Ollama synthesis\n• Sources and assumptions",
            ha="center", va="center", fontsize=8.5, fontweight="bold", bbox=box_answer)

    # Node 5: Guardrail Node
    ax.text(13.1, 3.5, "4. Guardrails\n• Scope and source check\n• Numeric claims fail closed\n• Causal disclaimer\n• Subsidy exclusion",
            ha="center", va="center", fontsize=8.2, fontweight="bold", bbox=box_guardrail)

    # Arrows
    arrow_kw = dict(arrowstyle="->", lw=2, color="#374151")
    
    # Input -> Classify
    ax.annotate("", xy=(2.1, 3.5), xytext=(1.7, 3.5), arrowprops=arrow_kw)
    
    # Classify -> Retrieve (up)
    ax.annotate("", xy=(5.0, 5.2), xytext=(3.9, 3.8), arrowprops=dict(arrowstyle="->", lw=2, color=c_green))
    ax.text(4.1, 4.8, "Evidence", fontsize=8, color=c_green, fontweight="bold")

    # Classify -> Forecast (down)
    ax.annotate("", xy=(5.0, 3.5), xytext=(4.3, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color=c_amber))
    ax.text(4.2, 3.8, "Forecast", fontsize=8, color=c_amber, fontweight="bold")

    # Classify -> Scenario (down)
    ax.annotate("", xy=(5.0, 1.8), xytext=(3.9, 3.2), arrowprops=dict(arrowstyle="->", lw=2, color="#BE123C"))
    ax.text(4.1, 2.2, "Intervention", fontsize=8, color="#BE123C", fontweight="bold")

    # Retrieve -> Answer
    ax.annotate("", xy=(8.7, 3.9), xytext=(7.7, 5.2), arrowprops=dict(arrowstyle="->", lw=2, color=c_green))

    # Forecast -> Answer
    ax.annotate("", xy=(8.7, 3.5), xytext=(7.8, 3.5), arrowprops=dict(arrowstyle="->", lw=2, color=c_amber))

    # Scenario -> Answer
    ax.annotate("", xy=(8.7, 3.1), xytext=(7.7, 1.8), arrowprops=dict(arrowstyle="->", lw=2, color="#BE123C"))

    # Answer -> Guardrail
    ax.annotate("", xy=(11.9, 3.5), xytext=(11.3, 3.5), arrowprops=arrow_kw)

    # MemorySaver Checkpointer note at top
    ax.text(7.4, 6.6, "Workflow Managed with LangGraph MemorySaver and a Session Checkpointer",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#1F2937",
            bbox=dict(boxstyle="square,pad=0.4", facecolor="#E5E7EB", edgecolor="none"))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Figure 6 architecture diagram saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_architecture_diagram()
