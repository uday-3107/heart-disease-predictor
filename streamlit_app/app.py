"""
Heart Disease Prediction - Streamlit Application
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import shap
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Typography & Base System ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background-color: #0d1117 !important;
        color: #f0f4f8 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    h1 { font-size: 1.75rem !important; font-weight: 800 !important; letter-spacing: -0.03em !important; color: #f0f4f8 !important; }
    h2.section-header { font-size: 1.2rem !important; font-weight: 700 !important; color: #f0f4f8 !important; }
    h3 { font-size: 1.0rem !important; font-weight: 600 !important; color: #f0f4f8 !important; }
    p, label, .stMarkdown { font-size: 0.875rem !important; line-height: 1.6 !important; color: #8899aa !important; }
    caption, .chart-title { font-size: 0.72rem !important; letter-spacing: 0.07em !important; text-transform: uppercase !important; color: #8899aa !important; }

    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
    }

    /* ── Header banner ── */
    .main-header {
        background: linear-gradient(135deg, #7b1e1e 0%, #c0392b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
        color: white;
        border: 1px solid #c0392b44;
    }
    .main-header h1 { font-size: 1.5rem !important; margin: 0; letter-spacing: -0.5px; color: white !important; }
    .main-header p  { opacity: 0.88; margin: 0.5rem 0 0 0; font-size: 0.9rem !important; color: white !important; }

    /* ── Compact Risk Result Strip ── */
    .risk-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .risk-strip.high { background: #1a0808; border-left: 4px solid #e74c3c; color: #f0f4f8; margin-top: 0.75rem; }
    .risk-strip.low { background: #081a0e; border-left: 4px solid #2ecc71; color: #f0f4f8; margin-top: 0.75rem; }
    .prob-badge {
        background: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 800;
        margin: 0 1rem;
    }

    /* ── Dark metric cards ── */
    .metric-card {
        background: #1c2130;
        border-radius: 12px;
        padding: 0.85rem 0.75rem;
        text-align: center;
        border: 1px solid #2a3550;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #4a6fa5; }
    .metric-card .icon { font-size: 1.3rem; margin-bottom: 0.25rem; }
    .metric-card .label {
        font-size: 0.68rem !important; color: #8899aa !important; letter-spacing: 0.08em !important;
        text-transform: uppercase; margin-bottom: 0.2rem;
    }
    .metric-card .value {
        font-size: 1.35rem !important; font-weight: 700 !important; color: #ffffff !important; line-height: 1.1 !important;
    }
    .metric-card .value.red   { color: #e74c3c !important; }
    .metric-card .value.green { color: #2ecc71 !important; }
    .metric-card .value.blue  { color: #3b82f6 !important; }

    /* ── Best model cards ── */
    .metric-card-best {
        background: #1c2333;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #3b82f655;
        box-shadow: 0 0 12px #3b82f611;
    }
    .metric-card-best .label {
        font-size: 0.68rem !important; color: #8899aa !important; letter-spacing: 0.09em !important;
        text-transform: uppercase; margin-bottom: 0.4rem;
    }
    .metric-card-best .value {
        font-size: 1.5rem !important; font-weight: 800 !important; color: #ffffff !important;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #1a1500; border: 1px solid #f9a82533;
        border-radius: 8px; padding: 0.75rem 1rem;
        font-size: 0.83rem; color: #c8a84b;
    }

    /* ── Analyse button ── */
    div[data-testid="stButton"] button {
        background: #e74c3c !important;
        color: white !important;
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.5rem !important;
        border-radius: 8px !important;
        border: none !important;
        letter-spacing: 0.02em !important;
        transition: background 0.15s ease, transform 0.1s ease !important;
        width: auto !important;
        min-width: 160px !important;
    }
    div[data-testid="stButton"] button:hover { background: #c0392b !important; transform: translateY(-1px) !important; }
    div[data-testid="stButton"] button:active { transform: translateY(0) !important; }

    /* ── Dataframe table text fix ── */
    [data-testid="stDataFrame"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stDataFrame"] th {
        background-color: #1c2333 !important;
        color: #aabbcc !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stDataFrame"] td {
        color: #e0e0e0 !important;
        font-size: 0.9rem !important;
    }

    /* ── Sidebar nav ── */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color: #8899aa !important; }

    /* ── Chart containers ── */
    .chart-box {
        background: #151c28;
        border-radius: 14px;
        border: 1px solid #2a3550;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }
    .chart-comparison-wrapper {
        max-width: 780px;
        max-height: 280px;
        margin: 0 auto;
        overflow: hidden;
    }
    .chart-title {
        font-size: 0.72rem !important; color: #8899aa !important;
        text-transform: uppercase !important; letter-spacing: 0.07em !important;
        margin-bottom: 0.5rem; padding-left: 0.2rem;
    }

    /* ── Selectbox + slider label ── */
    .stSelectbox label, .stSlider label {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #8899aa !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / 'models'
DATA_DIR   = BASE_DIR.parent / 'data'

@st.cache_resource
def load_artifacts():
    paths = {
        'model'   : MODELS_DIR / 'best_model.pkl',
        'all'     : MODELS_DIR / 'all_models.pkl',
        'features': MODELS_DIR / 'feature_columns.json',
        'name'    : MODELS_DIR / 'best_model_name.json',
        'metrics' : MODELS_DIR / 'metrics.csv',
    }
    if not paths['model'].exists():
        return None, None, None, None, None, None
    model        = joblib.load(paths['model'])
    all_models   = joblib.load(paths['all']) if paths['all'].exists() else {}
    with open(paths['features']) as f: feature_cols = json.load(f)
    with open(paths['name'])     as f: best_name    = json.load(f)['best_model']
    metrics = pd.read_csv(paths['metrics'], index_col=0) if paths['metrics'].exists() else None
    # Load a background sample for SHAP
    X_train_path = DATA_DIR / 'X_train.csv'
    bg = None
    if X_train_path.exists():
        X_train_full = pd.read_csv(X_train_path)
        X_train_full = X_train_full[feature_cols] if all(c in X_train_full.columns for c in feature_cols) else X_train_full
        bg = X_train_full.sample(min(50, len(X_train_full)), random_state=42)
    return model, all_models, feature_cols, best_name, metrics, bg

model, all_models, feature_cols, best_model_name, metrics_df, shap_bg = load_artifacts()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>❤️ Heart Disease Risk Predictor</h1>
    <p>Clinical Decision Support System &nbsp;·&nbsp; 10 ML Models &nbsp;·&nbsp; Cleveland Heart Disease Dataset</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗂️ Navigation")
    page = st.radio("", [
        "🔬 Patient Prediction",
        "📊 Model Performance",
        "🤖 Model Explorer",
        "ℹ️ About"
    ], label_visibility="collapsed")

    st.markdown("---")
    if model is not None:
        st.success(f"✅ Best model: **{best_model_name}**")
        if metrics_df is not None:
            auc = metrics_df.loc[best_model_name, 'roc_auc'] if best_model_name in metrics_df.index else "—"
            st.info(f"ROC-AUC: **{auc}**")
    else:
        st.error("⚠️ **Models not trained yet**")
        st.caption("Run `cd models && python train_model.py` to train all models.")

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Disclaimer:</b> Educational use only. Not for clinical diagnosis. Consult a qualified physician.
    </div>
    """, unsafe_allow_html=True)

def plotly_dark_layout(title="", height=400):
    return dict(
        title=dict(text=title, font=dict(color="#ffffff", size=14), x=0.01),
        paper_bgcolor="#151c28",
        plot_bgcolor="#151c28",
        font=dict(color="#8899aa", family="Inter, sans-serif", size=11),

        legend=dict(bgcolor="#1c2130", bordercolor="#2a3550", borderwidth=1,
                    font=dict(color="#e0e0e0")),
        margin=dict(l=40, r=20, t=50, b=40),
        height=height,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Patient Prediction
# ══════════════════════════════════════════════════════════════════════════════
if "Prediction" in page:
    st.markdown('<h2 class="section-header">Patient Clinical Data Input</h2>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ **Model artifacts not found.**")
        st.markdown("""
        The trained model files are missing. Please run the training script first:
        
        ```bash
        cd models && python train_model.py
        ```
        
        This will train all 10 models and save the required artifacts:
        - `best_model.pkl` (KNN)
        - `all_models.pkl` (all 10 models)
        - `feature_columns.json`
        - `best_model_name.json`
        - `metrics.csv`
        - All plots in `models/plots/`
        """)
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        age  = st.slider("Age (years)", 20, 80, 55)
        sex  = st.selectbox("Sex", ["Male", "Female"])
        sex_val = 1 if sex == "Male" else 0

        st.markdown("**🫀 Cardiac Tests**")
        thalach  = st.slider("Max Heart Rate (bpm)", 70, 210, 150)
        exang    = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        exang_val = 1 if exang == "Yes" else 0
        oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.0, step=0.1)

    with col2:
        st.markdown("**💉 Vitals**")
        trestbps = st.slider("Resting Blood Pressure (mmHg)", 90, 200, 130)
        chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
        fbs_sel  = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs      = 1 if fbs_sel == "Yes" else 0

        st.markdown("**📉 ECG**")
        restecg  = st.selectbox("Resting ECG", [
            "0 – Normal",
            "1 – ST-T Wave Abnormality",
            "2 – Left Ventricular Hypertrophy"])
        restecg_val = int(restecg[0])

    with col3:
        st.markdown("**🩻 Imaging**")
        ca    = st.slider("Major Vessels Coloured (0–3)", 0, 3, 0)
        slope = st.selectbox("ST Segment Slope", [
            "0 – Upsloping", "1 – Flat", "2 – Downsloping"])
        slope_val = int(slope[0])
        thal  = st.selectbox("Thalassemia", [
            "1 – Normal", "2 – Fixed Defect", "3 – Reversible Defect"])
        thal_val = int(thal[0])

        st.markdown("**💢 Chest Pain**")
        cp    = st.selectbox("Chest Pain Type", [
            "0 – Typical Angina", "1 – Atypical Angina",
            "2 – Non-Anginal Pain", "3 – Asymptomatic"])
        cp_val = int(cp[0])

    def build_input():
        row = {
            'age': age, 'sex': sex_val,
            'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'thalach': thalach,
            'exang': exang_val, 'oldpeak': oldpeak, 'ca': ca,
            'age_hr_ratio': round(age / thalach, 3),
            'high_chol': int(chol > 240),
            'bp_flag'  : int(trestbps > 130),
            'cp_1': int(cp_val==1), 'cp_2': int(cp_val==2), 'cp_3': int(cp_val==3),
            'restecg_1': int(restecg_val==1), 'restecg_2': int(restecg_val==2),
            'slope_1': int(slope_val==1), 'slope_2': int(slope_val==2),
            'thal_2': int(thal_val==2), 'thal_3': int(thal_val==3),
        }
        return pd.DataFrame([row])[feature_cols]

    st.markdown("---")
    model_choice = st.selectbox(
        "Predict using:",
        ["Best Model (" + best_model_name + ")"] + sorted(all_models.keys()),
        index=0
    )

    if st.button("🔍 Analyse Risk"):
        input_df = build_input()

        if model_choice.startswith("Best"):
            chosen_model = model
            chosen_name  = best_model_name
        else:
            chosen_model = all_models[model_choice]
            chosen_name  = model_choice

        prob = chosen_model.predict_proba(input_df)[0][1]
        pred = chosen_model.predict(input_df)[0]
        pct_val = prob * 100
        pct_str = f"{pct_val:.1f}"
        auc_val = metrics_df.loc[chosen_name, 'roc_auc'] if (metrics_df is not None and chosen_name in metrics_df.index) else "—"

        st.markdown(f"## Prediction — *{chosen_name}*")

        # Result Strip (Full Width)
        risk_type = "high" if pred == 1 else "low"
        risk_label = "🚨 High Risk — Heart Disease Likely" if pred == 1 else "✅ Low Risk — No Disease Indicated"
        risk_color_hex = "#e74c3c" if pred == 1 else "#2ecc71"
        
        st.markdown(f"""
        <div class="risk-strip {risk_type}">
            <span style="font-weight: 800; color: {risk_color_hex};">{risk_label}</span>
            <span class="prob-badge">{pct_str}%</span>
            <span style="opacity: 0.8; font-size: 0.85rem;">Model: {chosen_name} · AUC: {auc_val}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 5 metric cards — full width single row
        risk_color = "red" if pred == 1 else "green"
        cards = [
            ("🎯", "RISK", f"{pct_str}%", risk_color),
            ("📊", "AUC", f"{auc_val}", "blue"),
            ("🧬", "RATIO", f"{round(age/thalach,2)}", ""),
            ("💊", "CHOL", "Yes" if chol > 240 else "No", "red" if chol > 240 else "green"),
            ("💉", "BP", "High" if trestbps > 130 else "Normal", "red" if trestbps > 130 else "green"),
        ]
        cols = st.columns(5)
        for col_w, (icon, label, value, color) in zip(cols, cards):
            with col_w:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="icon">{icon}</div>
                    <div class="label">{label}</div>
                    <div class="value {color}">{value}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk bar — full width below cards
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[pct_val], y=["Risk"],
            orientation='h',
            marker_color='#e74c3c' if pred == 1 else '#2ecc71',
            width=0.4,
            hovertemplate=f"Risk Probability: {pct_val:.1f}%<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=[100 - pct_val], y=["Risk"],
            orientation='h',
            marker_color='#1e2d45',
            width=0.4,
            hoverinfo='skip'
        ))
        fig.add_vline(x=50, line_color="#8899aa", line_dash="dash", line_width=1)
        fig.update_layout(
            barmode='stack',
            xaxis=dict(range=[0, 100], title="Risk Probability (%)",
                       gridcolor="#1e2d45", linecolor="#1e2d45", tickfont=dict(color="#8899aa")),
            yaxis=dict(showticklabels=False, linecolor="#1e2d45"),
            showlegend=False,
            **plotly_dark_layout(f"Risk Score: {pct_val:.1f}%", height=130)
        )
        st.plotly_chart(fig, width="stretch")

        # SHAP explanation — per-patient interpretability
        with st.expander("🧠 Why did the model predict this?"):
            try:
                with st.spinner("Computing SHAP explanation..."):
                    bg = shap_bg if shap_bg is not None else input_df
                    try:
                        explainer = shap.Explainer(chosen_model, bg)
                        shap_values = explainer(input_df)
                    except Exception:
                        explainer = shap.PermutationExplainer(chosen_model.predict_proba, bg)
                        shap_values = explainer(input_df)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1]
                elif hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
                    shap_vals = shap_values.values[:, :, 1]
                elif hasattr(shap_values, 'values'):
                    shap_vals = shap_values.values
                else:
                    shap_vals = shap_values
                shap_vals = np.array(shap_vals)
                if shap_vals.ndim > 1 and shap_vals.shape[0] == 1:
                    shap_vals = shap_vals[0]
                feat_df = pd.DataFrame({'feature': feature_cols, 'shap': shap_vals})
                feat_df['abs'] = feat_df['shap'].abs()
                top_feats = feat_df.sort_values('abs', ascending=True).tail(10)
                colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in top_feats['shap']]
                fig_shap = go.Figure(go.Bar(
                    x=top_feats['shap'], y=top_feats['feature'],
                    orientation='h', marker_color=colors,
                    hovertemplate="%{y}: %{x:.4f}<extra></extra>"
                ))
                fig_shap.update_layout(
                    xaxis_title="SHAP Value (impact on prediction)",
                    yaxis=dict(autorange='reversed'),
                    **plotly_dark_layout("Top features driving this prediction", height=320)
                )
                fig_shap.add_vline(x=0, line_color="#8899aa", line_width=1)
                st.plotly_chart(fig_shap, width="stretch")
                st.caption("🔴 Red = pushes prediction toward Heart Disease &nbsp;·&nbsp; 🟢 Green = pushes toward Healthy")
            except Exception as e:
                st.caption(f"SHAP explanation unavailable for this model: {e}")

        st.markdown("### 📋 Input Summary")
        with st.expander("Show full input details"):
            summary = {
                "Age": age, "Sex": sex, "Chest Pain": cp,
                "Resting BP": f"{trestbps} mmHg", "Cholesterol": f"{chol} mg/dl",
                "Max HR": f"{thalach} bpm", "Exercise Angina": exang,
                "ST Depression": oldpeak, "Major Vessels": ca,
                "Age/HR Ratio": round(age/thalach, 3),
                "High Chol Flag": "Yes" if chol > 240 else "No",
                "High BP Flag": "Yes" if trestbps > 130 else "No",
            }
            st.dataframe(pd.DataFrame(summary.items(), columns=["Feature", "Value"]),
                         width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown('<h2 class="section-header">Model Performance Dashboard</h2>', unsafe_allow_html=True)

    if metrics_df is not None:
        # ── Metrics table ──────────────────────────────────────────────────────
        st.markdown("### 📋 All Models — Metrics Table")
        display_cols = [c for c in ['accuracy','precision','recall','f1',
                                     'roc_auc','cv_auc_mean','cv_auc_std']
                        if c in metrics_df.columns]
        styled = (metrics_df[display_cols]
                  .style.format("{:.4f}")
                  .set_table_styles([
                      {'selector': 'thead th',
                       'props': [('background-color','#1c2130'),
                                 ('color','#aabbcc'),
                                 ('font-size','0.8rem'),
                                 ('text-transform','uppercase'),
                                 ('letter-spacing','0.05em'),
                                 ('border-bottom','1px solid #2a3550')]},
                      {'selector': 'tbody td',
                       'props': [('background-color','#0f1117'),
                                 ('color','#e0e0e0'),
                                 ('font-size','0.88rem'),
                                 ('border-bottom','1px solid #1a2035')]},
                      {'selector': 'tbody tr:hover td',
                       'props': [('background-color','#1a2540 !important')]},
                  ])
                  .highlight_max(subset=['accuracy','recall','f1','roc_auc'],
                                 color='#0d3320')
                  .highlight_min(subset=['accuracy','recall','f1','roc_auc'],
                                 color='#3d0d0d'))
        st.dataframe(styled, width="stretch")
        st.caption(f"🏆 Best model: **{best_model_name}** (highest ROC-AUC)")

        # ── Best model metric cards ────────────────────────────────────────────
        st.markdown("### 🏆 Best Model Key Metrics")
        row = metrics_df.loc[best_model_name]
        card_meta = [
            ("🎯", "Accuracy",  'accuracy'),
            ("🔬", "Precision", 'precision'),
            ("📡", "Recall",    'recall'),
            ("⚖️", "F1-Score",  'f1'),
            ("📈", "ROC-AUC",   'roc_auc'),
        ]
        cols = st.columns(5)
        for col_w, (icon, label, metric) in zip(cols, card_meta):
            val = row[metric] if metric in row else "—"
            with col_w:
                st.markdown(f"""
                <div class="metric-card-best">
                    <div style="font-size:1.6rem;margin-bottom:0.3rem">{icon}</div>
                    <div class="label">{label}</div>
                    <div class="value">{val:.4f}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ **Model metrics not available**")
        st.markdown("""
        Please run the training script to generate metrics and plots:
        
        ```bash
        cd models && python train_model.py
        ```
        
        This will generate all 10 model metrics, confusion matrices, ROC curves, SHAP plots, and feature importance charts.
        """)

    # ── Charts grid ───────────────────────────────────────────────────────────
    st.markdown("---")

    # Chart 1: AUC Ranking
    st.markdown('<div class="chart-box"><div class="chart-title">🏅 Model Ranking by ROC-AUC</div>', unsafe_allow_html=True)
    auc_sorted = metrics_df['roc_auc'].sort_values()
    colors = ['#e74c3c' if n == best_model_name else '#2980b9' for n in auc_sorted.index]
    fig = go.Figure(go.Bar(
        x=auc_sorted.values, y=auc_sorted.index,
        orientation='h', marker_color=colors,
        text=[f"{v:.4f}" for v in auc_sorted.values],
        textposition='outside', textfont=dict(color='#aabbcc', size=10),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(xaxis=dict(range=[0.65, 1.02], gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
                      yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
                      **plotly_dark_layout("Model Ranking by ROC-AUC", height=380))
    st.plotly_chart(fig, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # Chart 2: CV AUC with error bars
    st.markdown('<div class="chart-box"><div class="chart-title">📉 Cross-Validation AUC (5-Fold)</div>', unsafe_allow_html=True)
    if 'cv_auc_mean' in metrics_df.columns and 'cv_auc_std' in metrics_df.columns:
        fig = go.Figure(go.Bar(
            x=metrics_df.index,
            y=metrics_df['cv_auc_mean'],
            error_y=dict(type='data', array=metrics_df['cv_auc_std'].values, visible=True,
                         color='#8899aa', thickness=1.5),
            marker_color='#16a085',
            hovertemplate="%{x}<br>CV AUC: %{y:.4f}<extra></extra>"
        ))
        fig.update_layout(xaxis_tickangle=-35, yaxis=dict(range=[0.65, 1.05]),
                          **plotly_dark_layout("5-Fold Cross-Validation AUC", height=360))
        st.plotly_chart(fig, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: ROC curves + All Metrics
    r2c1, r2c2 = st.columns(2)

    # Chart 3: ROC Curves
    with r2c1:
        st.markdown('<div class="chart-box"><div class="chart-title">📈 ROC Curves — All Models</div>', unsafe_allow_html=True)
        X_test_path = DATA_DIR / 'X_test.csv'
        y_test_path  = DATA_DIR / 'y_test.csv'
        if X_test_path.exists() and y_test_path.exists() and all_models:
            X_test_data = pd.read_csv(X_test_path)
            y_test_data = pd.read_csv(y_test_path).squeeze()
            fig = go.Figure()
            curve_colors = ['#2196F3','#4CAF50','#F44336','#FF9800','#9C27B0',
                            '#00BCD4','#795548','#607D8B','#E91E63','#3F51B5']
            for (name, mdl), color in zip(all_models.items(), curve_colors):
                fpr, tpr, _ = roc_curve(y_test_data, mdl.predict_proba(X_test_data)[:,1])
                auc_val_roc = roc_auc_score(y_test_data, mdl.predict_proba(X_test_data)[:,1])
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} ({auc_val_roc:.4f})",
                                         line=dict(color=color, width=1.8),
                                         hovertemplate=f"{name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                     line=dict(color='#4a5568', dash='dash', width=1),
                                     showlegend=False))
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                              xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
                              yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
                              **plotly_dark_layout("ROC Curves — All Models", height=420))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Test data not found. Run `train_model.py` to generate ROC curves.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 4: All Metrics Comparison
    with r2c2:
        st.markdown('<div class="chart-box"><div class="chart-title">📊 All Metrics Comparison</div>', unsafe_allow_html=True)
        metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metric_cols = [c for c in metric_cols if c in metrics_df.columns]
        fig = go.Figure()
        comp_colors = ['#2196F3','#4CAF50','#F44336','#FF9800','#9C27B0',
                       '#00BCD4','#795548','#607D8B','#E91E63','#3F51B5']
        for (name, row), color in zip(metrics_df[metric_cols].iterrows(), comp_colors):
            fig.add_trace(go.Bar(
                name=name, x=metric_cols, y=row.values,
                marker_color=color, opacity=0.88,
                hovertemplate=f"{name}<br>%{{x}}: %{{y:.4f}}<extra></extra>"
            ))
        fig.update_layout(barmode='group', yaxis=dict(range=[0.65, 1.05]),
                          **plotly_dark_layout("All Metrics Comparison", height=420))
        st.plotly_chart(fig, width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 5: Confusion Matrices
    st.markdown('<div class="chart-box"><div class="chart-title">🔲 Confusion Matrices — All Models</div>', unsafe_allow_html=True)
    if X_test_path.exists() and y_test_path.exists() and all_models:
        X_test_data = pd.read_csv(X_test_path)
        y_test_data = pd.read_csv(y_test_path).squeeze()
        model_names = list(all_models.keys())
        rows, cols_g = 2, 5
        fig = make_subplots(rows=rows, cols=cols_g,
                            subplot_titles=model_names,
                            vertical_spacing=0.18, horizontal_spacing=0.06)
        for i, (name, mdl) in enumerate(all_models.items()):
            r, c = divmod(i, cols_g)
            cm = confusion_matrix(y_test_data, mdl.predict(X_test_data))
            fig.add_trace(go.Heatmap(
                z=cm, x=['Healthy','Disease'], y=['Healthy','Disease'],
                colorscale='Blues', showscale=False,
                text=cm, texttemplate="%{text}",
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
            ), row=r+1, col=c+1)
        fig.update_layout(paper_bgcolor="#151c28", plot_bgcolor="#151c28",
                          font=dict(color="#8899aa"), height=520,
                          title=dict(text="Confusion Matrices — All Models",
                                     font=dict(color="#ffffff")))
        for ann in fig['layout']['annotations']:
            ann['font'] = dict(color='#aabbcc', size=10)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Test data not found. Run `train_model.py` to generate confusion matrices.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Chart 6: Feature Importance
    st.markdown('<div class="chart-box"><div class="chart-title">🌲 Feature Importances (Tree Models)</div>', unsafe_allow_html=True)
    tree_models = {k: v for k, v in all_models.items() if hasattr(v, 'feature_importances_')} if all_models else {}
    if tree_models and feature_cols:
        fig = go.Figure()
        fi_colors = ['#8e44ad','#2980b9','#16a085','#e67e22','#c0392b']
        for (name, mdl), color in zip(tree_models.items(), fi_colors):
            imp = pd.Series(mdl.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(12)
            fig.add_trace(go.Bar(
                x=imp.values, y=imp.index, orientation='h',
                name=name, marker_color=color, opacity=0.85,
                hovertemplate=f"{name}<br>%{{y}}: %{{x:.4f}}<extra></extra>"
            ))
        fig.update_layout(barmode='group',
                          xaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
                          yaxis=dict(gridcolor="#1e2d45", linecolor="#1e2d45", zerolinecolor="#1e2d45"),
                          **plotly_dark_layout("Feature Importances — Tree-Based Models", height=480))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No tree-based models found to extract feature importances.")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif "Explorer" in page:
    st.markdown('<h2 class="section-header">Model Explorer</h2>', unsafe_allow_html=True)
    st.markdown("Compare any two models side by side.")

    if metrics_df is None:
        st.warning("⚠️ **Model metrics not available**")
        st.markdown("""
        Please run the training script to generate metrics:
        
        ```bash
        cd models && python train_model.py
        ```
        """)
        st.stop()

    model_list = list(metrics_df.index)
    c1, c2 = st.columns(2)
    with c1:
        m1 = st.selectbox("Model A", model_list, index=0)
    with c2:
        m2 = st.selectbox("Model B", model_list, index=min(1, len(model_list)-1))

    if m1 == m2:
        st.warning("Select two different models.")
    else:
        metrics = ['accuracy','precision','recall','f1','roc_auc','cv_auc_mean']
        metrics = [m for m in metrics if m in metrics_df.columns]

        x_labels = metrics
        r1 = metrics_df.loc[m1, metrics].values.astype(float)
        r2 = metrics_df.loc[m2, metrics].values.astype(float)
        fig = go.Figure()
        fig.add_trace(go.Bar(name=m1, x=x_labels, y=r1, marker_color='#3498db',
                             text=[f"{v:.3f}" for v in r1], textposition='outside',
                             textfont=dict(color='#aabbcc', size=9),
                             hovertemplate=f"{m1}<br>%{{x}}: %{{y:.4f}}<extra></extra>"))
        fig.add_trace(go.Bar(name=m2, x=x_labels, y=r2, marker_color='#e74c3c',
                             text=[f"{v:.3f}" for v in r2], textposition='outside',
                             textfont=dict(color='#aabbcc', size=9),
                             hovertemplate=f"{m2}<br>%{{x}}: %{{y:.4f}}<extra></extra>"))
        fig.update_layout(barmode='group', yaxis=dict(range=[0.65, 1.08]),
                          **plotly_dark_layout(f"{m1}  vs  {m2}", height=340))
        st.plotly_chart(fig, width="stretch")

        diff = metrics_df.loc[m1, metrics] - metrics_df.loc[m2, metrics]
        diff_df = pd.DataFrame({
            'Metric': metrics,
            m1: metrics_df.loc[m1, metrics].values.round(4),
            m2: metrics_df.loc[m2, metrics].values.round(4),
            'Difference (A−B)': diff.values.round(4)
        })
        def color_diff(val):
            if isinstance(val, float):
                return 'color: #2ecc71' if val > 0 else ('color: #e74c3c' if val < 0 else '')
            return ''
        st.dataframe(diff_df.style.applymap(color_diff, subset=['Difference (A−B)']),
                     width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — About
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown('<h2 class="section-header">About This System</h2>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### 📦 Dataset
        - **Source**: `heart.csv` — Cleveland Heart Disease Dataset
        - **Samples**: 302 unique patients (after deduplication of Kaggle 1,025-row source)
        - **Train / Test**: 241 / 61 samples (80/20 stratified split)
        - **Features**: 21 (13 original + 3 engineered + 5 encoded)
        - **Target**: 1 = Heart disease, 0 = Healthy

        ### 🤖 Models Trained (10)
        | # | Model |
        |---|---|
        | 1 | Logistic Regression |
        | 2 | Random Forest |
        | 3 | XGBoost |
        | 4 | Gradient Boosting |
        | 5 | Extra Trees |
        | 6 | AdaBoost |
        | 7 | Support Vector Machine (SVM) |
        | 8 | K-Nearest Neighbours (KNN) |
        | 9 | Naive Bayes |
        | 10 | LightGBM |
        """)
    with c2:
        st.markdown("""
        ### ⚙️ Feature Engineering
        - `age_hr_ratio` = Age ÷ Max Heart Rate
        - `high_chol` = Cholesterol > 240 mg/dl
        - `bp_flag` = Resting BP > 130 mmHg
        - One-hot: `cp`, `restecg`, `slope`, `thal`

        ### 📐 Evaluation Metrics
        - **Accuracy** — Overall correctness
        - **Precision** — Positive predictive value
        - **Recall** — Sensitivity (clinically critical)
        - **F1-Score** — Harmonic mean
        - **ROC-AUC** — Overall discrimination
        - **5-Fold CV AUC** — Generalisation estimate
        """)

    st.markdown("---")
    st.markdown("""
    <div class="disclaimer">
    <b>⚠️ Medical Disclaimer:</b> This application is built for educational and academic purposes only.
    It is NOT intended for clinical diagnosis or medical decision-making.
    All outputs must be interpreted by qualified medical professionals.
    </div>""", unsafe_allow_html=True)
