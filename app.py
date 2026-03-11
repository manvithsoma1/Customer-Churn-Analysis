import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(__file__))
from src.pipeline import clean_data, engineer_features

# ─────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

RED   = "#e74c3c"
GREEN = "#2ecc71"
BLUE  = "#3498db"
GOLD  = "#f39c12"

DATASET_PATH = os.path.join(os.path.dirname(__file__), "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH  = os.path.join(os.path.dirname(__file__), "scaler.pkl")
KMEANS_PATH  = os.path.join(os.path.dirname(__file__), "kmeans.pkl")
FCOLS_PATH   = os.path.join(os.path.dirname(__file__), "feature_cols.json")

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0f1117; }
    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-left: 4px solid;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 8px;
    }
    .kpi-label { font-size: 0.82rem; color: #8b9cc7; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-size: 2rem; font-weight: 700; margin-top: 4px; }
    .risk-badge {
        display: inline-block;
        padding: 6px 22px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 8px;
    }
    .risk-low    { background: #1a4d2e; color: #2ecc71; }
    .risk-medium { background: #4d3a1a; color: #f39c12; }
    .risk-high   { background: #4d1a1a; color: #e74c3c; }
    div[data-testid="stSidebar"] { background: #161b2e; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
@st.cache_data
def load_raw_data(path=DATASET_PATH):
    return pd.read_csv(path)

@st.cache_data
def load_clean_data(path=DATASET_PATH):
    df = pd.read_csv(path)
    df = clean_data(df)
    return df

@st.cache_data
def load_feature_data(path=DATASET_PATH):
    df = pd.read_csv(path)
    df = clean_data(df)
    df = engineer_features(df)
    df_final = pd.get_dummies(df, drop_first=True)
    return df_final

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_resource
def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None

@st.cache_resource
def load_kmeans():
    if os.path.exists(KMEANS_PATH):
        return joblib.load(KMEANS_PATH)
    return None

@st.cache_data
def load_feature_cols():
    if os.path.exists(FCOLS_PATH):
        with open(FCOLS_PATH) as f:
            return json.load(f)
    return []


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.markdown("## 📡 Churn Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 EDA Explorer", "🤖 Churn Predictor",
     "👥 Customer Segments", "💰 Revenue Impact"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit · Scikit-Learn · SHAP · Plotly")


# ─────────────────────────────────────────────
# HELPER — KPI CARD
# ─────────────────────────────────────────────
def kpi_card(label, value, color=BLUE):
    st.markdown(f"""
    <div class="kpi-card" style="border-color:{color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🏠 Overview Dashboard")
    st.markdown("High-level KPIs and key visualisations for the Telco Churn dataset.")

    with st.spinner("Loading dataset …"):
        df = load_clean_data()

    total          = len(df)
    churn_rate     = df["Churn"].mean() * 100
    rev_at_risk    = df[df["Churn"] == 1]["MonthlyCharges"].sum() * 12
    model          = load_model()
    model_acc      = "88.04%" if model else "—"

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Customers",  f"{total:,}",         BLUE)
    with c2: kpi_card("Churn Rate",       f"{churn_rate:.1f}%", RED)
    with c3: kpi_card("Model Accuracy",   model_acc,            GREEN)
    with c4: kpi_card("Revenue at Risk",  f"${rev_at_risk:,.0f}", GOLD)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    # Pie chart
    with col_a:
        st.subheader("Overall Churn Distribution")
        counts = df["Churn"].value_counts().reset_index()
        counts.columns = ["Churn", "Count"]
        counts["Label"] = counts["Churn"].map({0: "Retained", 1: "Churned"})
        fig = px.pie(
            counts, values="Count", names="Label",
            color="Label",
            color_discrete_map={"Retained": GREEN, "Churned": RED},
            hole=0.45,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=20, b=20), legend=dict(font_color="white"))
        st.plotly_chart(fig, use_container_width=True)

    # Bar chart — churn by contract
    with col_b:
        st.subheader("Churn Rate by Contract Type")
        if "Contract" in df.columns:
            contract_churn = (
                df.groupby("Contract")["Churn"].mean() * 100
            ).reset_index()
            contract_churn.columns = ["Contract", "Churn Rate (%)"]
            fig2 = px.bar(
                contract_churn, x="Contract", y="Churn Rate (%)",
                color="Churn Rate (%)",
                color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                text_auto=".1f",
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white", coloraxis_showscale=False,
                               margin=dict(t=20, b=20))
            st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 2 — EDA EXPLORER
# ═════════════════════════════════════════════
elif page == "🔍 EDA Explorer":
    st.title("🔍 EDA Explorer")

    uploaded = st.file_uploader("Upload your own dataset CSV (or skip to use default)", type=["csv"])

    with st.spinner("Loading data …"):
        if uploaded:
            raw = pd.read_csv(uploaded)
            df = clean_data(raw)
        else:
            df = load_clean_data()
        df = engineer_features(df)

    # Sidebar filters
    st.sidebar.markdown("### Filters")
    contracts = df["Contract"].unique().tolist() if "Contract" in df.columns else []
    sel_contracts = st.sidebar.multiselect("Contract Type", contracts, default=contracts)

    if "tenure_group" in df.columns:
        tenure_groups = df["tenure_group"].unique().tolist()
        sel_tenure = st.sidebar.multiselect("Tenure Group", tenure_groups, default=tenure_groups)
        df = df[df["tenure_group"].isin(sel_tenure)]

    if contracts:
        df = df[df["Contract"].isin(sel_contracts)]

    st.subheader(f"Filtered Dataset ({len(df):,} rows)")
    st.dataframe(df.head(50), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Charges vs Tenure (by Churn)")
        fig = px.scatter(
            df, x="tenure", y="MonthlyCharges",
            color=df["Churn"].map({0: "Retained", 1: "Churned"}),
            color_discrete_map={"Retained": GREEN, "Churned": RED},
            opacity=0.6, size_max=6,
            labels={"color": "Status"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Feature Distribution")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        feat = st.selectbox("Select Feature", num_cols)
        fig2 = px.histogram(
            df, x=feat, color=df["Churn"].map({0: "Retained", 1: "Churned"}),
            barmode="overlay", opacity=0.7, nbins=40,
            color_discrete_map={"Retained": GREEN, "Churned": RED},
            labels={"color": "Status"},
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="white", margin=dict(t=30, b=20))
        st.plotly_chart(fig2, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 3 — CHURN PREDICTOR
# ═════════════════════════════════════════════
elif page == "🤖 Churn Predictor":
    st.title("🤖 Churn Predictor")
    st.markdown("Fill in a customer profile below and get an instant churn risk prediction.")

    model        = load_model()
    scaler       = load_scaler()
    feature_cols = load_feature_cols()

    if model is None or not feature_cols:
        st.error("⚠️ `model.pkl` or `feature_cols.json` not found. "
                 "Please run the notebook first to generate these artefacts.")
        st.stop()

    # ── Input form ──────────────────────────────
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
            total_charges   = monthly_charges * max(tenure, 1)
        with c2:
            contract        = st.selectbox("Contract Type",
                                           ["Month-to-month", "One year", "Two year"])
            internet        = st.selectbox("Internet Service",
                                           ["DSL", "Fiber optic", "No"])
            payment         = st.selectbox("Payment Method",
                                           ["Electronic check", "Mailed check",
                                            "Bank transfer (automatic)",
                                            "Credit card (automatic)"])
        with c3:
            paperless       = st.radio("Paperless Billing", ["Yes", "No"])
            phone_svc       = st.radio("Phone Service",    ["Yes", "No"])
            multiple_lines  = st.radio("Multiple Lines",   ["Yes", "No", "No phone service"])

        submitted = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True)

    if submitted:
        with st.spinner("Analysing customer profile …"):
            # ── Build raw input row matching training schema ──
            raw_input = {
                "tenure":           [tenure],
                "MonthlyCharges":   [monthly_charges],
                "TotalCharges":     [total_charges],
                "SeniorCitizen":    [0],
                "Partner":          ["No"],
                "Dependents":       ["No"],
                "PhoneService":     [phone_svc],
                "MultipleLines":    [multiple_lines],
                "InternetService":  [internet],
                "OnlineSecurity":   ["No"],
                "OnlineBackup":     ["No"],
                "DeviceProtection": ["No"],
                "TechSupport":      ["No"],
                "StreamingTV":      ["No"],
                "StreamingMovies":  ["No"],
                "Contract":         [contract],
                "PaperlessBilling": [paperless],
                "PaymentMethod":    [payment],
                "gender":           ["Male"],
                "Churn":            [0],               # placeholder — dropped below
            }
            df_raw = pd.DataFrame(raw_input)
            df_c   = clean_data(df_raw)
            df_fe  = engineer_features(df_c)
            df_enc = pd.get_dummies(df_fe, drop_first=True)

            # Align columns to training schema
            for col in feature_cols:
                if col not in df_enc.columns:
                    df_enc[col] = 0
            df_enc = df_enc[feature_cols]

            X_scaled = scaler.transform(df_enc)
            prob     = model.predict_proba(X_scaled)[0][1]        # churn probability

            # ── Risk tier ──────────────────────────────────────
            if prob < 0.35:
                tier, badge_cls, tier_color = "Low Risk",    "risk-low",    GREEN
            elif prob < 0.65:
                tier, badge_cls, tier_color = "Medium Risk", "risk-medium", GOLD
            else:
                tier, badge_cls, tier_color = "High Risk",  "risk-high",   RED

            # ── Layout ─────────────────────────────────────────
            st.markdown("---")
            left, right = st.columns([1, 1])

            with left:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode  = "gauge+number",
                    value = prob * 100,
                    title = {"text": "Churn Probability (%)", "font": {"color": "white"}},
                    gauge = {
                        "axis":      {"range": [0, 100], "tickcolor": "white"},
                        "bar":       {"color": tier_color},
                        "bgcolor":   "#1e2130",
                        "steps":     [
                            {"range": [0,  35], "color": "#1a4d2e"},
                            {"range": [35, 65], "color": "#4d3a1a"},
                            {"range": [65, 100],"color": "#4d1a1a"},
                        ],
                        "threshold": {
                            "line":  {"color": "white", "width": 3},
                            "thickness": 0.8,
                            "value": prob * 100,
                        },
                    },
                    number = {"suffix": "%", "font": {"color": tier_color, "size": 48}},
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                                  height=280, margin=dict(t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

            with right:
                st.markdown(f"""
                <div style="padding:20px">
                    <div class="kpi-label">Risk Assessment</div>
                    <div style="margin-top:10px">
                        <span class="risk-badge {badge_cls}">{tier}</span>
                    </div>
                    <br/>
                    <div class="kpi-label">Churn probability</div>
                    <div class="kpi-value" style="color:{tier_color}">{prob*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            # ── Top 3 reasons (SHAP) ───────────────────────────
            st.markdown("### 🔎 Top 3 Reasons Driving This Prediction")
            try:
                import shap
                explainer   = shap.Explainer(model.predict, pd.DataFrame(
                    scaler.transform(pd.DataFrame([np.zeros(len(feature_cols))],
                                     columns=feature_cols)),
                    columns=feature_cols))
                shap_vals   = explainer(pd.DataFrame(X_scaled, columns=feature_cols))
                sv          = shap_vals.values[0]
                top_idx     = np.argsort(np.abs(sv))[::-1][:3]
                for rank, idx in enumerate(top_idx, 1):
                    fname  = feature_cols[idx]
                    fval   = df_enc.iloc[0, idx]
                    impact = sv[idx]
                    direction = "↑ increases" if impact > 0 else "↓ decreases"
                    color     = RED if impact > 0 else GREEN
                    st.markdown(
                        f"**{rank}.** `{fname}` = `{fval:.2f}` &nbsp;—&nbsp; "
                        f"<span style='color:{color}'>{direction} churn risk</span>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                # Fallback: use feature importances
                if hasattr(model, "feature_importances_"):
                    imp    = model.feature_importances_
                    top3   = np.argsort(imp)[::-1][:3]
                    for rank, idx in enumerate(top3, 1):
                        st.markdown(f"**{rank}.** `{feature_cols[idx]}` — high importance feature")
                else:
                    st.info("SHAP explanation unavailable for this model type.")


# ═════════════════════════════════════════════
# PAGE 4 — CUSTOMER SEGMENTS
# ═════════════════════════════════════════════
elif page == "👥 Customer Segments":
    st.title("👥 Customer Segments")
    st.markdown("K-Means clustering (K=4) identifies distinct behavioural groups.")

    with st.spinner("Loading and segmenting customers …"):
        df_feat   = load_feature_data()
        scaler_m  = load_scaler()
        kmeans_m  = load_kmeans()
        feat_cols = load_feature_cols()

    if scaler_m is None or kmeans_m is None or not feat_cols:
        st.error("⚠️ Artefacts not found. Please execute the notebook first.")
        st.stop()

    df_feat  = df_feat.fillna(0)
    X_common = [c for c in feat_cols if c in df_feat.columns]
    missing  = [c for c in feat_cols if c not in df_feat.columns]
    for c in missing:
        df_feat[c] = 0
    X = df_feat[feat_cols].fillna(0)

    X_scaled  = scaler_m.transform(X)
    clusters  = kmeans_m.predict(X_scaled)
    df_feat["Cluster"] = clusters

    from sklearn.decomposition import PCA
    pca   = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame({"PCA 1": comps[:, 0], "PCA 2": comps[:, 1],
                            "Cluster": clusters.astype(str),
                            "Churn": df_feat["Churn"].values if "Churn" in df_feat.columns else 0})

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("PCA Scatter — Customer Clusters")
        fig = px.scatter(
            df_pca, x="PCA 1", y="PCA 2", color="Cluster",
            color_discrete_sequence=[BLUE, GREEN, RED, GOLD],
            opacity=0.55, size_max=5,
            labels={"Cluster": "Segment"},
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="white", margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn Rate by Segment")
        if "Churn" in df_feat.columns:
            seg_churn = (df_feat.groupby("Cluster")["Churn"].mean() * 100).reset_index()
            seg_churn.columns = ["Segment", "Churn Rate (%)"]
            seg_churn["Segment"] = seg_churn["Segment"].astype(str)
            fig2 = px.bar(
                seg_churn, x="Segment", y="Churn Rate (%)",
                color="Churn Rate (%)",
                color_continuous_scale=[GREEN, GOLD, RED],
                text_auto=".1f",
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="white", coloraxis_showscale=False,
                               margin=dict(t=30, b=20))
            st.plotly_chart(fig2, use_container_width=True)

    # Cluster profile table
    st.markdown("---")
    st.subheader("Cluster Profile Summary")
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                "num_services", "charges_per_month"]
    avail    = [c for c in num_cols if c in df_feat.columns] + (
        ["Churn"] if "Churn" in df_feat.columns else [])
    profile  = df_feat.groupby("Cluster")[avail].mean().round(2).reset_index()
    profile["Cluster"] = profile["Cluster"].astype(str)
    st.dataframe(profile, use_container_width=True)


# ═════════════════════════════════════════════
# PAGE 5 — REVENUE IMPACT
# ═════════════════════════════════════════════
elif page == "💰 Revenue Impact":
    st.title("💰 Revenue Impact Calculator")
    st.markdown("Estimate how much revenue can be protected with better retention.")

    with st.spinner("Loading data …"):
        df = load_clean_data()

    total_customers  = len(df)
    churned_count    = int(df["Churn"].sum())
    avg_monthly_base = float(df[df["Churn"] == 1]["MonthlyCharges"].mean())

    st.sidebar.markdown("### Revenue Inputs")
    retention_rate   = st.sidebar.slider("Retention Improvement (%)", 5, 50, 20)
    avg_charge       = st.sidebar.slider("Avg Monthly Charge ($)",
                                          int(avg_monthly_base * 0.5),
                                          int(avg_monthly_base * 2),
                                          int(avg_monthly_base))

    customers_saved  = int(churned_count * retention_rate / 100)
    annual_at_risk   = churned_count * avg_charge * 12
    revenue_saved    = customers_saved * avg_charge * 12

    col1, col2, col3 = st.columns(3)
    with col1: kpi_card("Churned Customers",  f"{churned_count:,}",         RED)
    with col2: kpi_card("Customers Saved",    f"{customers_saved:,}",       GREEN)
    with col3: kpi_card("Revenue Protected",  f"${revenue_saved:,.0f} /yr", GOLD)

    st.markdown("---")

    # Before / After bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Revenue at Risk (Before)",
        x=["Without Retention"],
        y=[annual_at_risk],
        marker_color=RED,
        text=[f"${annual_at_risk:,.0f}"],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Revenue Protected (After)",
        x=["With Retention Strategy"],
        y=[revenue_saved],
        marker_color=GREEN,
        text=[f"${revenue_saved:,.0f}"],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Annual Revenue — {retention_rate}% Retention Improvement",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        barmode="group",
        yaxis_title="Annual Revenue ($)",
        margin=dict(t=60, b=30),
        legend=dict(font_color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Impact summary
    st.markdown("---")
    st.markdown(f"""
    | Metric | Value |
    |---|---|
    | Total Customers | {total_customers:,} |
    | Churned Customers | {churned_count:,} ({churned_count/total_customers*100:.1f}%) |
    | Avg Monthly Charge (Churned) | ${avg_charge:,} |
    | Annual Revenue at Risk | ${annual_at_risk:,.0f} |
    | Customers Saved ({retention_rate}% improvement) | {customers_saved:,} |
    | **Annual Revenue Protected** | **${revenue_saved:,.0f}** |
    """)
