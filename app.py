"""
LaMagnetica — Sector-Based Cost & Profit Prediction
Scenario Engine Demo

Standalone Streamlit app. Loads demo_bundle.pkl (exported from the
training notebook) and reimplements what_if_safe() without any
dependency on the original notebook, master dataframe, or pipeline.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LaMagnet | Scenario Engine",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: #FAFAF8;
    }

    /* Hide default streamlit chrome we don't need */
    #MainMenu, footer { visibility: hidden; }

    /* ── Header banner ──────────────────────────────────── */
    .brand-header {
        background: linear-gradient(135deg, #161616 0%, #232323 60%, #161616 100%);
        border-radius: 20px;
        padding: 32px 40px;
        margin-bottom: 28px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        border: 1px solid #2E2E2E;
    }
    .brand-eyebrow {
        color: #F5C518;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .brand-title {
        color: white;
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        color: rgba(255,255,255,0.78);
        font-size: 15px;
        font-weight: 400;
        margin-top: 8px;
        max-width: 720px;
        line-height: 1.5;
    }

    /* ── Card containers ────────────────────────────────── */
    .card-title {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #161616;
        border-bottom: 2px solid #F5C518;
        padding-bottom: 8px;
        margin-bottom: 16px;
        display: inline-block;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
    }
    div[data-testid="stContainer"] {
        border-color: rgba(22,22,22,0.10) !important;
    }

    /* ── Cluster badge ───────────────────────────────────── */
    .cluster-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .badge-c1 { background: #1A1A1A; color: #F5C518; }
    .badge-c2 { background: #FEF3C7; color: #92400E; }
    .badge-c3 { background: #F0EEE6; color: #44403C; }

    .baseline-stats {
        font-size: 13px;
        color: #6B7280;
        margin-top: 8px;
    }
    .baseline-stats b { color: #374151; }

    /* ── Method pill ─────────────────────────────────────── */
    .method-pill {
        display: inline-block;
        background: #FFF8E1;
        color: #7A5C00;
        border: 1px solid #F5C518;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 16px;
    }

    /* ── Crisis banner ───────────────────────────────────── */
    .crisis-banner {
        background: linear-gradient(135deg, #1A1A1A 0%, #2A2A2A 100%);
        border: 1.5px solid #F5C518;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 20px;
    }
    .crisis-title {
        color: #F5C518;
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 6px;
    }
    .crisis-body {
        color: rgba(255,255,255,0.82);
        font-size: 13.5px;
        line-height: 1.5;
    }
    .crisis-body b { color: #F5C518; }

    /* ── Metric cards ────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: #FAFAF8;
        border-radius: 14px;
        padding: 16px 18px;
        border: 1px solid rgba(22,22,22,0.08);
        border-left: 3px solid #F5C518;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 12.5px;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
        color: #161616;
    }

    /* ── Buttons ──────────────────────────────────────────── */
    .stButton button {
        background: #F5C518 !important;
        color: #161616 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 12px !important;
        box-shadow: 0 4px 14px rgba(245, 197, 24, 0.35) !important;
        transition: transform 0.15s ease !important;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        background: #FFD23F !important;
        box-shadow: 0 6px 18px rgba(245, 197, 24, 0.5) !important;
    }
    .stButton button p { color: #161616 !important; }

    /* ── Section labels ──────────────────────────────────── */
    h3 {
        font-weight: 700 !important;
        color: #161616 !important;
        letter-spacing: -0.3px !important;
    }

    /* ── Dataframe ────────────────────────────────────────── */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD BUNDLE (cached — only loads once per session)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_bundle():
    bundle_path = os.path.join(os.path.dirname(__file__), "demo_bundle.pkl")
    return joblib.load(bundle_path)

bundle = load_bundle()

RF_MODEL_C2             = bundle["rf_model_c2"]
RF_MODEL_C3             = bundle["rf_model_c3"]
RF_TEST_R2_C2           = bundle["rf_test_r2_c2"]
RF_TEST_R2_C3           = bundle["rf_test_r2_c3"]
COMPANY_BASELINE        = bundle["company_baseline"]
CLUSTER_TRAIN_MEDIANS   = bundle["cluster_train_medians"]
CLUSTER_OPEX_RATIO      = bundle["cluster_opex_ratio"]
MACRO_BETAS             = bundle["macro_betas"]
C1_QUANTILES            = bundle["c1_quantiles"]
CRISIS_SEVERITY_PENALTY = bundle["crisis_severity_penalty"]
COMPANY_RATIOS_DICT     = bundle["company_ratios_dict"]
CLUSTER_RATIOS_DICT     = bundle["cluster_ratios_dict"]
GLOBAL_RATIOS_DICT      = bundle["global_ratios_dict"]
FCF_METHOD_DICT         = bundle["fcf_method_dict"]
FEAT_PRESENT            = bundle["feat_present"]
GP_TARGET               = bundle["gp_target"]
MASTER_SLIM             = bundle["master_slim"]
CLUSTER_LABELS          = bundle["cluster_labels"]

CLUSTER_MODELS = {2: RF_MODEL_C2, 3: RF_MODEL_C3}
CLUSTER_TEST_R2 = {2: RF_TEST_R2_C2, 3: RF_TEST_R2_C3}


# ─────────────────────────────────────────────────────────────
# CORE ENGINE LOGIC (ported from the notebook, no external deps)
# ─────────────────────────────────────────────────────────────
def get_kpi_ratio(ticker, ratio_name, cluster_id):
    if ticker in COMPANY_RATIOS_DICT:
        val = COMPANY_RATIOS_DICT[ticker].get(ratio_name)
        if val is not None and not pd.isna(val):
            return val
    if cluster_id in CLUSTER_RATIOS_DICT:
        val = CLUSTER_RATIOS_DICT[cluster_id].get(ratio_name)
        if val is not None and not pd.isna(val):
            return val
    return GLOBAL_RATIOS_DICT.get(ratio_name, 0.0)


def predict_fcf(net_profit, revenue, ticker, cluster_id):
    method = FCF_METHOD_DICT.get(cluster_id, "net_income")
    if method == "net_income":
        ratio = get_kpi_ratio(ticker, "fcf_conversion_co", cluster_id)
        return net_profit * ratio
    else:
        ratio = get_kpi_ratio(ticker, "fcf_to_revenue_co", cluster_id)
        return revenue * ratio


def _compute_kpi_chain(revenue, gp_margin, ticker, cid):
    cost_ratio   = 1 - (gp_margin / 100)
    gross_profit = revenue * (1 - cost_ratio)

    opex_ratio = CLUSTER_OPEX_RATIO.get(cid, 0.6)
    da_ratio   = get_kpi_ratio(ticker, "da_to_revenue_co", cid)
    int_ratio  = get_kpi_ratio(ticker, "interest_to_revenue", cid)
    tax_rate   = get_kpi_ratio(ticker, "effective_tax_rate", cid)

    da_est     = revenue * da_ratio
    opex_est   = gross_profit * opex_ratio
    ebitda     = gross_profit - opex_est
    ebit       = ebitda - da_est
    interest   = revenue * int_ratio
    ebt        = ebit - interest
    net_profit = ebt * (1 - tax_rate)

    fcf = predict_fcf(net_profit, revenue, ticker, cid)

    return {
        "gp_margin_pct":         round(gp_margin, 2),
        "gross_profit":          round(gross_profit, 2),
        "ebitda":                round(ebitda, 2),
        "ebitda_margin_pct":     round(ebitda / revenue * 100, 2) if revenue > 0 else 0,
        "ebt":                   round(ebt, 2),
        "net_profit":            round(net_profit, 2),
        "net_profit_margin_pct": round(net_profit / revenue * 100, 2) if revenue > 0 else 0,
        "fcf":                   round(fcf, 2),
        "fcf_margin_pct":        round(fcf / revenue * 100, 2) if revenue > 0 else 0,
        "ratios_used": {
            "opex_ratio": round(opex_ratio, 4),
            "da_ratio":   round(da_ratio, 4),
            "int_ratio":  round(int_ratio, 4),
            "tax_rate":   round(tax_rate, 4),
        }
    }


def what_if(ticker, revenue, inflation=None, rate=None, gdp=None):
    base       = COMPANY_BASELINE[ticker]
    cid        = int(base["cluster_auto"])
    last_rev   = base["last_revenue"]
    rev_4q_ago = base["revenue_4q_ago"]
    base_gm    = base["gross_margin_pct"]

    revenue_qoq_growth = (
        (revenue - last_rev) / abs(last_rev) * 100
        if pd.notna(last_rev) and last_rev != 0 else 0.0
    )
    revenue_yoy_growth = (
        (revenue - rev_4q_ago) / abs(rev_4q_ago) * 100
        if pd.notna(rev_4q_ago) and rev_4q_ago != 0
        else revenue_qoq_growth
    )

    quantile_range = None

    if cid == 1:
        pred_gp_margin     = base_gm
        prediction_method  = "Naive baseline (no predictive signal found)"
        quantile_range = {
            "pessimistic_gm": np.clip(base_gm + C1_QUANTILES["q10"] * 100, 0, 100),
            "base_gm":        np.clip(base_gm + C1_QUANTILES["q50"] * 100, 0, 100),
            "optimistic_gm":  np.clip(base_gm + C1_QUANTILES["q90"] * 100, 0, 100),
        }
    else:
        feats_c  = [f for f in FEAT_PRESENT if f in CLUSTER_TRAIN_MEDIANS[cid].index]
        feat_vec = CLUSTER_TRAIN_MEDIANS[cid].copy()
        feat_vec["revenue_qoq_growth"] = revenue_qoq_growth
        feat_vec["revenue_yoy_growth"] = revenue_yoy_growth

        if inflation is not None:
            baseline_infl_rf = float(feat_vec.get("inflation_lag1", 2.5))
            feat_vec["inflation_lag1"]  = inflation
            feat_vec["inflation_lag2"]  = inflation * 0.9
            feat_vec["inflation_delta"] = inflation - baseline_infl_rf
        if rate is not None:
            baseline_rate_rf = float(feat_vec.get("macro_fed_funds_rate", 4.5))
            feat_vec["macro_fed_funds_rate"] = rate
            feat_vec["rate_delta_1q"]        = rate - baseline_rate_rf
        if gdp is not None:
            feat_vec["gdp_growth_qoq"] = gdp

        feat_df  = pd.DataFrame([feat_vec])[feats_c]
        delta_gp = CLUSTER_MODELS[cid].predict(feat_df)[0]

        pred_gp_margin    = np.clip(base_gm + delta_gp * 100, 0.0, 100.0)
        prediction_method = f"Random Forest (Test R² = {CLUSTER_TEST_R2[cid]:.3f})"

    # ── Macro adjustment — override only ────────────────────
    MACRO_DAMPENING_NORMAL = 0.30
    MACRO_DAMPENING_CRISIS = 0.15
    MAX_MACRO_ADJ          = 0.05

    macro_delta     = 0.0
    crisis_mode     = False
    penalty_applied = 0.0

    if any(x is not None for x in [inflation, rate, gdp]):
        betas = MACRO_BETAS.get(cid, {})

        baseline_infl = float(CLUSTER_TRAIN_MEDIANS[cid].get("inflation_lag1", 2.5))
        baseline_rate = float(CLUSTER_TRAIN_MEDIANS[cid].get("macro_fed_funds_rate", 4.5))
        baseline_gdp  = float(CLUSTER_TRAIN_MEDIANS[cid].get("gdp_growth_qoq", 0.5))

        infl_shock = (inflation - baseline_infl) if inflation is not None else 0.0
        rate_shock = (rate      - baseline_rate) if rate      is not None else 0.0
        gdp_shock  = (gdp       - baseline_gdp)  if gdp       is not None else 0.0

        crisis_mode = (
            revenue_qoq_growth < -10.0 and
            infl_shock         >   5.0 and
            rate_shock         >   3.0
        )
        dampening = MACRO_DAMPENING_CRISIS if crisis_mode else MACRO_DAMPENING_NORMAL

        if inflation is not None:
            macro_delta += betas.get("inflation", 0.0) * infl_shock
        if rate is not None:
            rate_beta = betas.get("rate", 0.0)
            if crisis_mode:
                rate_beta = abs(rate_beta)
            macro_delta += rate_beta * rate_shock
        if gdp is not None:
            macro_delta += betas.get("gdp", 0.0) * gdp_shock

        macro_delta = np.clip(macro_delta * dampening, -MAX_MACRO_ADJ, MAX_MACRO_ADJ)
        pred_gp_margin = np.clip(pred_gp_margin + macro_delta * 100, 0.0, 100.0)

        if quantile_range is not None:
            for k in quantile_range:
                quantile_range[k] = np.clip(quantile_range[k] + macro_delta * 100, 0, 100)

        if crisis_mode:
            penalty_applied = CRISIS_SEVERITY_PENALTY.get(cid, 4.5)
            pred_gp_margin  = np.clip(pred_gp_margin - penalty_applied, 0.0, 100.0)
            if quantile_range is not None:
                for k in quantile_range:
                    quantile_range[k] = np.clip(quantile_range[k] - penalty_applied, 0, 100)

            ticker_hist  = MASTER_SLIM[MASTER_SLIM["ticker"] == ticker]["cost_to_revenue"].dropna()
            hist_worst   = float(ticker_hist.quantile(0.95)) if len(ticker_hist) else 0.7
            crisis_floor = min(hist_worst * 1.10, 1.0)
            pred_cost_ratio = float(np.clip(
                max(1 - pred_gp_margin/100, 1 - base_gm/100),
                0.0, crisis_floor
            ))
            pred_gp_margin = np.clip((1 - pred_cost_ratio) * 100, 0.0, 100.0)

    base_kpis = _compute_kpi_chain(revenue, pred_gp_margin, ticker, cid)

    result = {
        "ticker":              ticker,
        "sub_sector":          base["sub_sector"],
        "cluster":             base["cluster_label"],
        "prediction_method":   prediction_method,
        "crisis_mode":         crisis_mode,
        "crisis_penalty_pp":   round(penalty_applied, 2),
        "scenario_revenue":    round(revenue, 2),
        "revenue_qoq_growth":  round(revenue_qoq_growth, 2),
        "revenue_yoy_growth":  round(revenue_yoy_growth, 2),
        "macro_delta_applied": round(macro_delta * 100, 3),
        "base_case":           base_kpis,
    }

    if quantile_range is not None:
        result["pessimistic_case"] = _compute_kpi_chain(
            revenue, quantile_range["pessimistic_gm"], ticker, cid)
        result["optimistic_case"] = _compute_kpi_chain(
            revenue, quantile_range["optimistic_gm"], ticker, cid)

    return result


def what_if_safe(ticker, revenue, inflation=None, rate=None, gdp=None):
    errors = []
    ticker = str(ticker).strip().upper() if ticker else ""

    if not ticker:
        errors.append("Please select a ticker symbol.")
    elif ticker not in COMPANY_BASELINE:
        errors.append(f"'{ticker}' is not in the model universe.")

    try:
        revenue = float(revenue)
        if revenue <= 0:
            errors.append("Revenue must be a positive number.")
        elif ticker in COMPANY_BASELINE:
            last_rev = COMPANY_BASELINE[ticker]["last_revenue"]
            if pd.notna(last_rev) and last_rev > 0:
                ratio = revenue / last_rev
                if ratio > 5 or ratio < 0.2:
                    errors.append(
                        f"Revenue input is more than 5x different from {ticker}'s "
                        f"last known revenue (${last_rev:,.0f}). Please enter a "
                        f"more plausible value."
                    )
    except (ValueError, TypeError):
        errors.append("Revenue must be a valid number.")

    macro_bounds = {"inflation": (-5, 30), "rate": (-5, 25), "gdp": (-20, 20)}
    macro_inputs = {"inflation": inflation, "rate": rate, "gdp": gdp}
    for name, val in macro_inputs.items():
        if val is not None:
            lo, hi = macro_bounds[name]
            if not (lo <= val <= hi):
                errors.append(f"{name.capitalize()} input is outside the plausible range ({lo} to {hi}).")

    if errors:
        return {"success": False, "errors": errors}

    try:
        result = what_if(ticker, revenue, **macro_inputs)
        result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "errors": [f"Unexpected error: {str(e)}"]}


# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="brand-header">
    <div style="display:flex; align-items:flex-start; gap:20px;">
        <img src="https://s3-eu-west-1.amazonaws.com/sortlist-core-api/7HudFy6iHTEpM6ik5KjxfM7t"
             style="height:56px; width:auto; border-radius:8px; margin-top:2px;" />
        <div>
            <div class="brand-eyebrow">Scenario Engine</div>
            <p class="brand-title">LaMagnet</p>
            <p class="brand-subtitle">
                Enter a revenue assumption for a company to see how it propagates through
                the cost structure into Gross Profit, EBITDA, Net Profit, and Free Cash Flow.
                The model does not forecast revenue — it answers: given this revenue,
                what will the cost structure look like?
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    card1 = st.container(border=True)
    with card1:
        st.markdown('<div class="card-title">Scenario Inputs</div>', unsafe_allow_html=True)

        all_tickers = sorted(COMPANY_BASELINE.keys())
        ticker = st.selectbox("Company", all_tickers, index=all_tickers.index("MSFT") if "MSFT" in all_tickers else 0)

        baseline = COMPANY_BASELINE[ticker]
        last_rev = baseline["last_revenue"]
        cluster_label = baseline["cluster_label"]

        cluster_id = int(baseline["cluster_auto"])
        badge_class = {1: "badge-c1", 2: "badge-c2", 3: "badge-c3"}.get(cluster_id, "badge-c1")
        cluster_short = {
            "Cluster1_ProfitableHighMargin": "Profitable High-Margin",
            "Cluster2_HypergrowthInvestment": "Hypergrowth Investment",
            "Cluster3_PhysicalVariable": "Physical / Variable",
        }.get(cluster_label, cluster_label)

        st.markdown(f"""
        <div class="cluster-badge {badge_class}">● {cluster_short}</div>
        <div class="baseline-stats">
            Last known revenue: <b>${last_rev:,.0f}</b> &nbsp;|&nbsp;
            Last known gross margin: <b>{baseline['gross_margin_pct']:.1f}%</b>
        </div>
        """, unsafe_allow_html=True)

        revenue = st.number_input(
            "Revenue assumption ($)",
            min_value=0.0,
            value=float(last_rev),
            step=float(last_rev) * 0.01 if last_rev else 1_000_000.0,
            format="%.0f",
        )

        st.markdown("**Macro overrides** (optional — leave default to use the model's standard prediction)")

        use_macro = st.checkbox("Override macro conditions")

        inflation, rate, gdp = None, None, None
        if use_macro:
            m1, m2, m3 = st.columns(3)
            with m1:
                inflation = st.slider("Inflation (%)", -5.0, 30.0, 2.0, 0.5)
            with m2:
                rate = st.slider("Interest rate (%)", -5.0, 25.0, 4.5, 0.5)
            with m3:
                gdp = st.slider("GDP growth (%)", -20.0, 20.0, 1.0, 0.5)

        run = st.button("Run scenario", type="primary", use_container_width=True)

with col2:
    card2 = st.container(border=True)
    with card2:
        st.markdown('<div class="card-title">Results</div>', unsafe_allow_html=True)

        if run:
            result = what_if_safe(ticker, revenue, inflation, rate, gdp)

            if not result["success"]:
                for err in result["errors"]:
                    st.error(err)
            else:
                if result["crisis_mode"]:
                    st.markdown(f"""
                    <div class="crisis-banner">
                        <div class="crisis-title">⚠ Crisis regime detected</div>
                        <div class="crisis-body">
                            A severity penalty of <b>-{result['crisis_penalty_pp']:.2f}pp</b> has been
                            applied, anchored to historical worst-case quarterly margin compression
                            for this cluster. This is an explicit, disclosed assumption — not a model
                            prediction. Extreme macro inputs fall outside the training distribution
                            of the underlying models.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f'<div class="method-pill">{result["prediction_method"]}</div>', unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Revenue (QoQ)", f"{result['revenue_qoq_growth']:+.1f}%")
                with m2:
                    st.metric("Revenue (YoY)", f"{result['revenue_yoy_growth']:+.1f}%")

                if result["macro_delta_applied"] != 0:
                    st.caption(f"Macro adjustment applied: {result['macro_delta_applied']:+.3f}pp to gross margin")

                st.divider()

                if "pessimistic_case" in result:
                    import plotly.graph_objects as go

                    cases = ["Pessimistic", "Base case", "Optimistic"]
                    gms = [
                        result["pessimistic_case"]["gp_margin_pct"],
                        result["base_case"]["gp_margin_pct"],
                        result["optimistic_case"]["gp_margin_pct"],
                    ]
                    colors = ["#161616", "#F5C518", "#8A8A8A"]

                    fig = go.Figure(go.Bar(
                        x=cases, y=gms, marker_color=colors,
                        text=[f"{v:.1f}%" for v in gms], textposition="outside",
                    ))
                    fig.update_layout(
                        height=280, margin=dict(t=10, b=10, l=10, r=10),
                        yaxis_title="Gross margin (%)", showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    kpi_table = pd.DataFrame({
                        "Pessimistic": result["pessimistic_case"],
                        "Base case":   result["base_case"],
                        "Optimistic":  result["optimistic_case"],
                    }).T[["gross_profit", "ebitda", "net_profit", "fcf"]]
                    kpi_table.columns = ["Gross Profit", "EBITDA", "Net Profit", "FCF"]
                    st.dataframe(
                        kpi_table.style.format("${:,.0f}"),
                        use_container_width=True,
                    )

                else:
                    bk = result["base_case"]
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("Gross Margin", f"{bk['gp_margin_pct']:.1f}%")
                    k2.metric("Gross Profit", f"${bk['gross_profit']/1e9:.2f}B")
                    k3.metric("EBITDA", f"${bk['ebitda']/1e9:.2f}B", f"{bk['ebitda_margin_pct']:.1f}%")
                    k4.metric("Net Profit", f"${bk['net_profit']/1e9:.2f}B", f"{bk['net_profit_margin_pct']:.1f}%")
                    k5.metric("FCF", f"${bk['fcf']/1e9:.2f}B", f"{bk['fcf_margin_pct']:.1f}%")

                    with st.expander("Ratios used in this calculation"):
                        r = bk["ratios_used"]
                        st.write(f"OpEx/Revenue: {r['opex_ratio']*100:.1f}% (cluster median)")
                        st.write(f"D&A/Revenue: {r['da_ratio']*100:.2f}% (company or cluster median)")
                        st.write(f"Interest/Revenue: {r['int_ratio']*100:.2f}% (company or cluster median)")
                        st.write(f"Effective tax rate: {r['tax_rate']*100:.1f}% (company or cluster median)")
        else:
            st.info("Configure a scenario and click **Run scenario** to see results.")

st.markdown("""
<div style="text-align:center; color:#9CA3AF; font-size:13px; margin-top: 12px; padding: 16px;">
    Model notes: Cluster 1 (Profitable High-Margin) uses a naive baseline — no model
    outperformed predicting zero margin change in backtesting. Clusters 2 and 3 use
    Random Forest models on quarter-over-quarter gross margin change. All KPI assumptions
    (interest, tax, D&A, FCF) are derived from company or cluster historical medians,
    not flat constants.
</div>
""", unsafe_allow_html=True)
