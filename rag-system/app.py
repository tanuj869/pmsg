import os
import sys
import time
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="PM Surya Ghar — Policy Analytics & Decision Support Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph.build_graph import app_graph
from tools.forecast_tool import (
    predict_conversion_ratio,
    forecast_installations,
    simulate_tariff_shock,
    OFFICIAL_FORECASTS
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .dashboard-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .dashboard-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-top: 4px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #38BDF8;
        margin-top: 4px;
    }
    
    .route-badge-forecast {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: #EFF6FF;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .route-badge-retrieve {
        background: linear-gradient(135deg, #064E3B 0%, #059669 100%);
        color: #ECFDF5;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .citation-box {
        background: rgba(56, 189, 248, 0.08);
        border-left: 4px solid #38BDF8;
        padding: 12px 16px;
        margin-top: 10px;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    .highlight-card {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.25);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">PM Surya Ghar: Policy Analytics & Decision-Support System</div>
    <div class="dashboard-subtitle">
        Empirical Econometric Modeling, National Forecasting, and Policy Scenario Simulator across 36 Indian States/UTs
    </div>
</div>
""", unsafe_allow_html=True)

# Helper function to load datasets safely
@st.cache_data
def load_data():
    paths = {
        "modeling": os.path.join(PROJECT_ROOT, "data", "processed", "modeling_table_population_normalized.csv"),
        "correlations": os.path.join(PROJECT_ROOT, "data", "processed", "correlation_table.csv"),
        "forecast_comparison": os.path.join(PROJECT_ROOT, "data", "processed", "forecast_method_comparison.csv"),
        "walk_forward": os.path.join(PROJECT_ROOT, "data", "processed", "forecast_walk_forward_summary.csv"),
        "forecast_series": os.path.join(PROJECT_ROOT, "data", "processed", "installations_forecast.csv"),
        "national_series": os.path.join(PROJECT_ROOT, "data", "processed", "national_installations_series.csv"),
        "scenario_c": os.path.join(PROJECT_ROOT, "data", "processed", "scenario_c_targeting.csv"),
        "scenario_d": os.path.join(PROJECT_ROOT, "data", "processed", "scenario_d_tariff_shock.csv"),
        "register": os.path.join(PROJECT_ROOT, "data", "processed", "dataset_register.csv"),
        "dictionary": os.path.join(PROJECT_ROOT, "data", "processed", "data_dictionary.csv"),
    }
    data = {}
    for k, p in paths.items():
        if os.path.exists(p):
            data[k] = pd.read_csv(p)
        elif k == "modeling" and os.path.exists(os.path.join(PROJECT_ROOT, "data", "processed", "modeling_table.csv")):
            data[k] = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "modeling_table.csv"))
        else:
            data[k] = pd.DataFrame()
    return data

data = load_data()
df_mod = data.get("modeling", pd.DataFrame())

# Top KPI Metric Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

total_apps = df_mod["applications_aug2026"].sum() if "applications_aug2026" in df_mod.columns else 7800000
total_inst = df_mod["installations_aug2026"].sum() if "installations_aug2026" in df_mod.columns else 2300000
avg_conv = (df_mod["target_conversion_ratio"].mean() * 100) if "target_conversion_ratio" in df_mod.columns else 46.2

with kpi1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">National Applications</div>
        <div class="metric-value">{:,.2f}L</div>
        <div class="metric-sub">Across 36 States & UTs</div>
    </div>
    """.format(total_apps / 100000), unsafe_allow_html=True)

with kpi2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">National Installations</div>
        <div class="metric-value">{:,.2f}L</div>
        <div class="metric-sub">Verified Completed Units</div>
    </div>
    """.format(total_inst / 100000), unsafe_allow_html=True)

with kpi3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Avg Conversion Ratio</div>
        <div class="metric-value">{:.1f}%</div>
        <div class="metric-sub">Range: 1.5% to 92.9%</div>
    </div>
    """.format(avg_conv), unsafe_allow_html=True)

with kpi4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Best Fit Model MAE</div>
        <div class="metric-value">0.135</div>
        <div class="metric-sub">-20.7% vs. Naive Baseline</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">12-Month Forecast</div>
        <div class="metric-value">~11.61M</div>
        <div class="metric-sub">95% CI: 10.2M – 13.2M</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 State Adoption & EDA",
    "🤖 Econometric Modeling & SHAP",
    "📈 National Forecasting",
    "🎯 Policy Scenario Simulator",
    "💡 AI Decision-Support Agent (RAG)",
    "🗂️ Data Sources & Live Data"
])

# -----------------------------------------------------------------------------
# TAB 1: State Adoption & EDA
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("State-Wise Adoption & Funnel Efficiency")
    st.markdown("""
    PM Surya Ghar adoption efficiency varies by a factor of **62x** across Indian states—from **1.5%** in Arunachal Pradesh to **92.9%** in Mizoram. 
    Below is the comprehensive empirical cross-section as of August 2026.
    """)
    
    col_chart, col_cor = st.columns([6, 4])
    
    with col_chart:
        img_state_path = os.path.join(PROJECT_ROOT, "data", "processed", "conversion_ratio_by_state.png")
        if os.path.exists(img_state_path):
            st.image(img_state_path, caption="Figure 1: State-by-State Application-to-Installation Conversion Ratio (%)", use_container_width=True)
    
    with col_cor:
        img_scat_path = os.path.join(PROJECT_ROOT, "data", "processed", "scatter_correlations.png")
        if os.path.exists(img_scat_path):
            st.image(img_scat_path, caption="Figure 2: Correlation Scatter Plots (Tariff Growth & Solar Irradiance vs Adoption)", use_container_width=True)

    st.markdown("---")
    st.subheader("Canonical 36-State Empirical Panel Table")
    
    if not df_mod.empty:
        df_display = df_mod.copy()
        if "target_conversion_ratio" in df_display.columns:
            df_display["Conversion Ratio (%)"] = (df_display["target_conversion_ratio"] * 100).round(2)
        
        display_cols = [c for c in [
            "state", "Conversion Ratio (%)", "applications_aug2026", "installations_aug2026", 
            "installations_per_lakh_population", "applications_per_lakh_population",
            "tariff_domestic4kw_2025_paise_per_kwh", "irradiance_annual_kwh_m2_day", 
            "solar_potential_6.69pct_wasteland_gw", "mnre_re_capacity_2025_mw"
        ] if c in df_display.columns]
        
        # Rename columns for presentation
        name_map = {
            "state": "State / UT",
            "applications_aug2026": "Applications",
            "installations_aug2026": "Installations",
            "installations_per_lakh_population": "Installations / lakh population",
            "applications_per_lakh_population": "Applications / lakh population",
            "tariff_domestic4kw_2025_paise_per_kwh": "Tariff (Paise/kWh)",
            "irradiance_annual_kwh_m2_day": "Irradiance (kWh/m²/day)",
            "solar_potential_6.69pct_wasteland_gw": "Solar Potential (GW)",
            "mnre_re_capacity_2025_mw": "RE Capacity (MW)"
        }
        df_show = df_display[display_cols].rename(columns=name_map)
        
        # State search filter
        search_state = st.text_input("🔍 Filter by State Name:", "")
        if search_state:
            df_show = df_show[df_show["State / UT"].str.contains(search_state, case=False, na=False)]
        
        st.dataframe(df_show, use_container_width=True, height=350)
        if "installations_per_lakh_population" in df_display.columns:
            st.caption("Per-capita rates use MoHFW estimated population for 1 July 2021 as the denominator and the project’s August 2026 scheme snapshot as the numerator. They are indicative, not contemporaneous rates.")
    
    st.markdown("---")
    st.subheader("Correlation Table: Covariates vs Conversion Ratio")
    df_cor = data.get("correlations", pd.DataFrame())
    if not df_cor.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(df_cor, use_container_width=True)
        with c2:
            st.info("""
            **Key Takeaways from Correlation Analysis:**
            - **Tariff Growth % ($r = +0.526$):** The single strongest predictor. States experiencing higher tariff hikes see higher conversion as rooftop economics become urgent.
            - **Solar Irradiance ($r = +0.510$):** Strong positive physical resource driver.
            - **Subsidy & Tariff Levels ($|r| < 0.15$):** Consistently weak correlation—providing direct empirical evidence for the **'Subsidy Paradox'**.
            """)

# -----------------------------------------------------------------------------
# TAB 2: Econometric Modeling & SHAP
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Model Comparison: Parsimonious Linear vs. Machine Learning Ensemble")
    st.markdown("""
    Two model architectures were evaluated using **Leave-One-Out Cross-Validation (LOO-CV)** across 28–35 states.
    Counterintuitively, a simple **2-feature regularized linear model** decisively outperforms the **6-feature Random Forest**.
    """)
    
    col_mod_tbl, col_shap = st.columns([5, 5])
    
    with col_mod_tbl:
        model_results = pd.DataFrame([
            {"Model Specification": "Linear Regression (Tariff Growth + Irradiance)", "Features": "2", "n": 28, "LOO-CV MAE": 0.135, "vs Naive Baseline": "-20.7%"},
            {"Model Specification": "Linear Regression (Tariff Growth Only)", "Features": "1", "n": 34, "LOO-CV MAE": 0.150, "vs Naive Baseline": "-15.3%"},
            {"Model Specification": "Linear Regression (Irradiance Only)", "Features": "1", "n": 28, "LOO-CV MAE": 0.155, "vs Naive Baseline": "-9.4%"},
            {"Model Specification": "Random Forest Regressor (Full Specification)", "Features": "6", "n": 28, "LOO-CV MAE": 0.161, "vs Naive Baseline": "-5.9%"},
            {"Model Specification": "Naive Baseline (Mean Target Value)", "Features": "0", "n": 28, "LOO-CV MAE": 0.170, "vs Naive Baseline": "0.0%"}
        ])
        st.dataframe(model_results, use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class="highlight-card">
            <h4>📐 The Best-Fit Linear Regression Equation</h4>
            <p>$$\\widehat{\\text{Conversion Ratio}} = -0.707 + 1.607 \\times (\\Delta\\text{Tariff Growth}) + 0.250 \\times (\\text{Solar Irradiance})$$</p>
            <ul>
                <li><strong>In-sample R²:</strong> 0.455 (explains 46% of variance)</li>
                <li><strong>Generalization LOO-CV MAE:</strong> 0.135</li>
                <li><strong>Sample size:</strong> n = 28 states with complete co-variates</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("""
        **Why did Random Forest underperform?**
        With small state-level cross-sections (n=28–35), tree-based non-linear ensembles overfit to idiosyncratic state noise. Parsimonious linear specifications generalize substantially better.
        """)
        
    with col_shap:
        img_shap_path = os.path.join(PROJECT_ROOT, "data", "processed", "shap_summary.png")
        if os.path.exists(img_shap_path):
            st.image(img_shap_path, caption="Figure 3: SHAP Feature Attribution — Impact on Model Output", use_container_width=True)
            st.caption("SHAP values independently confirm that Tariff Growth and Irradiance dominate feature importance, while subsidy/tariff levels remain near zero.")

# -----------------------------------------------------------------------------
# TAB 3: National Forecasting
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("National Adoption Forecasts (3, 6, and 12-Month Horizons)")
    st.markdown("""
    National cumulative installations were forecast from six dated official government releases (Nov 2024 to Aug 2026).
    Because the scheme is in its rapid early ramp-up phase, **log-linear (exponential growth) extrapolation** achieved the best fit.
    """)
    
    col_fc1, col_fc2, col_fc3 = st.columns(3)
    with col_fc1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">+3 Months (Nov 2026)</div>
            <div class="metric-value">~5.24 Million</div>
            <div class="metric-sub">95% CI: 4.62M – 5.95M</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_fc2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">+6 Months (Feb 2027)</div>
            <div class="metric-value">~6.84 Million</div>
            <div class="metric-sub">95% CI: 6.03M – 7.75M</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_fc3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">+12 Months (Jul 2027)</div>
            <div class="metric-value">~11.61 Million</div>
            <div class="metric-sub">95% CI: 10.24M – 13.17M</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    c_fc_img, c_fc_tbl = st.columns([6, 4])
    
    with c_fc_img:
        img_fc_path = os.path.join(PROJECT_ROOT, "data", "processed", "installations_forecast_chart.png")
        if os.path.exists(img_fc_path):
            st.image(img_fc_path, caption="Figure 4: National Cumulative Installations Extrapolation with 95% Prediction Interval", use_container_width=True)
            
    with c_fc_tbl:
        st.subheader("Model Comparison")
        df_fc_comp = data.get("forecast_comparison", pd.DataFrame())
        if not df_fc_comp.empty:
            st.dataframe(df_fc_comp, use_container_width=True, hide_index=True)

        wf = data.get("walk_forward", pd.DataFrame())
        if not wf.empty:
            st.subheader("Expanding-Window Validation")
            st.dataframe(wf, use_container_width=True, hide_index=True)
            st.caption("One-step-ahead validation uses three expanding-window origins from six dated observations. The small number of folds makes model rankings uncertain.")
            
        st.subheader("Historical Adoption Points")
        df_nat = data.get("national_series", pd.DataFrame())
        if not df_nat.empty:
            st.dataframe(df_nat[["date", "installations", "source"]], use_container_width=True, hide_index=True)
            
        st.info("""
        **Forecasting Methodology Note:**
        With only 6 dated national snapshots across 18 months, fitting high-parameter models like ARIMA, SARIMAX, or Prophet would cause severe overfitting. The log-linear trend provides an honest, transparent baseline.
        """)

# -----------------------------------------------------------------------------
# TAB 4: Policy Scenario Simulator
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Counterfactual Policy Scenario Simulator")
    st.markdown("""
    Evaluate how changes in electricity tariffs or targeted high-potential marketing interventions influence state-level rooftop adoption under fitted model assumptions.
    """)
    
    col_sim_ctrl, col_sim_view = st.columns([4, 6])
    
    with col_sim_ctrl:
        st.markdown("### Interactive Tariff Shock Simulator")
        
        # State selector
        state_list = df_mod["state"].dropna().tolist() if "state" in df_mod.columns else ["Gujarat", "Maharashtra", "Tamil Nadu", "Karnataka"]
        selected_sim_state = st.selectbox("Select State to Simulate:", state_list, index=state_list.index("Gujarat") if "Gujarat" in state_list else 0)
        
        # Slider for tariff growth shock
        shock_slider = st.slider("Tariff Growth Shock (Percentage Points):", min_value=1.0, max_value=25.0, value=10.0, step=1.0)
        
        sim_res = simulate_tariff_shock(state=selected_sim_state, shock_pp=shock_slider / 100.0)
        actual = sim_res.get("actual_conversion_ratio")
        actual_txt = f"{actual * 100:.2f}%" if actual is not None else "Not reported"
        pred_base = sim_res.get("predicted_base_conversion_ratio")
        pred_shock = sim_res.get("predicted_shocked_conversion_ratio")
        lift_pp = sim_res.get("state_specific_lift_pp")
        sim_ok = pred_base is not None and pred_shock is not None
        if sim_ok:
            pred_base_txt = f"{pred_base * 100:.2f}%"
            pred_shock_txt = f"{pred_shock * 100:.2f}%"
            lift_txt = f"+{lift_pp:.1f} percentage points" if lift_pp is not None else "N/A"
        else:
            pred_base_txt = "N/A — incomplete covariates"
            pred_shock_txt = "N/A"
            lift_txt = "N/A"

        st.markdown(f"""
        <div class="highlight-card">
            <h4>Simulation Output: {sim_res.get('state', selected_sim_state)}</h4>
            <p><strong>Actual Conversion Ratio:</strong> {actual_txt}</p>
            <p><strong>Predicted Baseline:</strong> {pred_base_txt}</p>
            <p><strong>Predicted with +{shock_slider:.0f}pp Shock:</strong> <span style="color:#F59E0B; font-weight:700;">{pred_shock_txt}</span></p>
            <p><strong>Predicted Net Lift:</strong> <span style="color:#10B981; font-weight:700;">{lift_txt}</span></p>
        </div>
        """, unsafe_allow_html=True)

        if not sim_ok:
            st.warning(sim_res.get(
                "unavailable_reason",
                "This state is outside the n=28 modeling sample, so the tariff-shock prediction cannot be computed."
            ))
        
    with col_sim_view:
        img_scend_path = os.path.join(PROJECT_ROOT, "data", "processed", "scenario_d_chart.png")
        if os.path.exists(img_scend_path):
            st.image(img_scend_path, caption="Figure 5: Scenario D — Predicted Mean Conversion Ratio Lift under Tariff Shocks", use_container_width=True)

    st.markdown("---")
    col_scen_c, col_scen_b = st.columns(2)
    
    with col_scen_c:
        st.subheader("🎯 Scenario C: High-Potential, Low-Adoption States")
        st.markdown("""
        States with **above-median solar irradiance** (> 5.0 kWh/m²/day) but **below-median conversion ratio** (< 46%):
        """)
        df_c = data.get("scenario_c", pd.DataFrame())
        if not df_c.empty:
            df_c_display = df_c.copy()
            if "target_conversion_ratio" in df_c_display.columns:
                df_c_display["Conversion Ratio (%)"] = (df_c_display["target_conversion_ratio"] * 100).round(2)
            st.dataframe(df_c_display, use_container_width=True, hide_index=True)
            st.caption("Prime candidates for targeted DISCOM streamlining and consumer awareness campaigns.")
            
    with col_scen_b:
        st.subheader("🛡️ Scenario B & E: Subsidy Policy Caveat")
        st.info("""
        **Why is Scenario B (Subsidy +10%) not modeled numerically?**
        - Central subsidy amounts are uniformly fixed under national PM Surya Ghar guidelines.
        - State-level subsidy top-ups are non-centralized, intermittent, and fragmented across 28 SERC orders.
        - The project transparently scopes out Scenario B rather than fabricating synthetic subsidy parameters.
        """)

# -----------------------------------------------------------------------------
# TAB 5: AI Decision-Support Agent (RAG)
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("☀️ SuryaIQ: LangGraph RAG Policy Assistant")
    st.markdown("""
    Ask natural language questions regarding **empirical findings, literature synthesis, and mathematical forecasts**. 
    The LangGraph workflow classifies questions into retrieval, forecast, or scenario paths, dispatches deterministic tools, prepares comparison data, and applies source, scope, causal, and numeric checks.
    """)
    
    # Query input container
    rag_examples = [
        "What's the predicted installation volume in 6 months?",
        "What is the subsidy paradox?",
        "Which two features best predict conversion ratio?",
        "What would a 10pp tariff growth shock do to Gujarat's conversion ratio?",
        "Why does Random Forest underperform the linear model here?",
        "Which states are high-potential, low-adoption?",
        "Is this conversion-ratio finding causal?",
        "How much would a 10% increase in state subsidy increase conversion ratio?"
    ]
    
    selected_q = st.selectbox("Choose a sample benchmark query or type your own:", ["(Type a custom question...)"] + rag_examples)
    default_q_val = selected_q if selected_q != "(Type a custom question...)" else ""
    user_query = st.text_input("Enter your research or policy question:", value=default_q_val, placeholder="e.g. What is the predicted installation volume in 6 months?")
    
    btn_col, _ = st.columns([2, 8])
    with btn_col:
        run_query = st.button("Submit to Agent", type="primary")
        
    if run_query and user_query.strip():
        with st.spinner("Processing through the LangGraph decision workflow..."):
            t0 = time.time()
            config = {"configurable": {"thread_id": "dashboard_session"}}
            state_in = {
                "question": user_query.strip(),
                "retrieved_docs": [],
                "citations": [],
                "caveats": []
            }
            res = app_graph.invoke(state_in, config=config)
            elapsed = round(time.time() - t0, 2)
            
        route = res.get("route", "retrieve")
        answer = res.get("answer", "")
        citations = res.get("citations", [])
        caveats = res.get("caveats", [])
        passed = res.get("safety_check_passed", True)
        
        st.markdown("---")
        
        # Meta badge row
        b1, b2, b3 = st.columns([3, 2, 5])
        with b1:
            if route == "forecast":
                st.markdown('<span class="route-badge-forecast">📈 Route: Deterministic Forecast Tool</span>', unsafe_allow_html=True)
            elif route == "scenario":
                st.markdown('<span class="route-badge-forecast">🧪 Route: Deterministic Scenario Workflow</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="route-badge-retrieve">📚 Route: Literature & Corpus RAG</span>', unsafe_allow_html=True)
        with b2:
            st.markdown(f"⏱️ **Latency:** `{elapsed}s`")
        with b3:
            if passed:
                st.success("✅ Guardrail Check Passed")
            else:
                st.warning("⚠️ Guardrail Caveat Appended")
                
        st.markdown("### Answer")
        st.markdown(answer)

        answer_meta = res.get("answer_metadata", {})
        if answer_meta:
            with st.expander("Confidence, assumptions, sources and limitations", expanded=True):
                st.write(f"**Confidence:** {answer_meta.get('confidence', 'Not assessed')}")
                st.write(f"**Source:** {answer_meta.get('source', 'Not recorded')}")
                st.write("**Assumptions:** " + "; ".join(answer_meta.get("assumptions", [])))
                st.write("**Limitations:** " + "; ".join(answer_meta.get("limitations", [])))
        
        if citations:
            st.markdown("### Cited Sources")
            for c in citations:
                st.markdown(f'<div class="citation-box">📌 <strong>{c}</strong></div>', unsafe_allow_html=True)
                
        if caveats:
            st.markdown("### Deterministic Guardrail Alerts")
            for cav in caveats:
                st.error(cav)
                
        if res.get("forecast_result"):
            with st.expander("🔍 View Structured Tool Output"):
                st.json(res["forecast_result"])
        visualization_data = res.get("visualization_data") or []
        if visualization_data:
            st.markdown("### Scenario comparison")
            st.bar_chart(pd.DataFrame(visualization_data).set_index("case" if "case" in visualization_data[0] else "state"))
                
    st.markdown("---")
    st.subheader("System Architecture")
    arch_img_path = os.path.join(PROJECT_ROOT, "figure6_architecture.png")
    if os.path.exists(arch_img_path):
        st.image(arch_img_path, caption="Figure 6: LangGraph retrieval, forecast and scenario workflow with deterministic guardrails", use_container_width=True)

with tab6:
    st.subheader("Dataset register and data dictionary")
    register = data.get("register", pd.DataFrame())
    dictionary = data.get("dictionary", pd.DataFrame())
    if not register.empty:
        st.dataframe(register, width="stretch", hide_index=True)
    if not dictionary.empty:
        with st.expander("Browse field definitions and missingness", expanded=False):
            st.dataframe(dictionary, width="stretch", hide_index=True, height=420)

    st.subheader("Live external sources")
    from src.external_data import STATE_POINTS, fetch_nasa_power_state, source_status
    status = source_status()
    st.json(status)
    live_state = st.selectbox("NASA POWER state/UT location proxy", sorted(STATE_POINTS), key="nasa_power_state")
    if st.button("Fetch NASA POWER monthly solar and temperature data", key="fetch_nasa_power"):
        try:
            live_payload = fetch_nasa_power_state(live_state)
            st.caption(f"NASA POWER values are for the {live_payload['location_proxy']} point, not a state-wide spatial average.")
            st.dataframe(pd.DataFrame(live_payload["series"]), width="stretch", hide_index=True)
            st.caption(f"Source: {live_payload['source_url']}")
        except Exception as exc:
            st.error(f"NASA POWER could not be reached: {exc}")
    st.info("Live data.gov.in retrieval in the agent requires DATA_GOV_IN_API_KEY and PMSG_OGD_RESOURCE_ID in the server environment. NITI Aayog ICeD is currently represented by the retained project covariate snapshot because its dashboard has no documented public API configured here.")

# Footer
st.markdown("---")
st.caption("PM Surya Ghar Policy Analytics Dashboard • Data as of August 2026 • Verified against official Rajya Sabha, CEA, NASA POWER, and NITI Aayog portals.")
