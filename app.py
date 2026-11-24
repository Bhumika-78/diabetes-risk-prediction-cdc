"""
🏥 CDC Diabetes Risk AI Platform & End-to-End Machine Learning Lifecycle Explorer
Full interactive coverage of the 7 core stages:
Raw Data ➔ Data Cleaning ➔ Feature Engineering ➔ Model Training ➔ Model Evaluation ➔ Deployment ➔ Prediction
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Add workspace to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.inference import predict_patient_risk, EXPECTED_FEATURE_ORDER

# Page Config
st.set_page_config(
    page_title="CDC Diabetes Risk AI Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f766e 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
    }
    
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
    }
    
    .pipeline-bar {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.8rem;
    }
    
    .step-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 0.4rem;
    }
    
    .badge-done {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    
    .badge-active {
        background: #e0f2fe;
        color: #0369a1;
        border: 1px solid #7dd3fc;
    }
    
    .metric-card {
        background: white;
        padding: 1.4rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }
    
    .result-banner {
        padding: 1.8rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 20px -5px rgba(0,0,0,0.15);
    }
    
    .risk-driver-item {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        color: #991b1b;
        font-weight: 500;
        font-size: 0.92rem;
    }
    
    .recom-item {
        background: #f0fdf4;
        border-left: 4px solid #10b981;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        color: #065f46;
        font-weight: 500;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper Paths
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MODELS_DIR = BASE_DIR / "models"
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "cdc_diabetes_health_indicators.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Load Benchmark Results if available
benchmark_data = {}
benchmark_json_path = MODELS_DIR / "benchmark_results.json"
if benchmark_json_path.exists():
    with open(benchmark_json_path, "r") as f:
        benchmark_data = json.load(f)

# The 7 Defined Stages
STAGES = [
    "1. Raw Data",
    "2. Data Cleaning",
    "3. Feature Engineering",
    "4. Model Training",
    "5. Model Evaluation",
    "6. Deployment",
    "7. Prediction"
]

# Session State for step navigation
if "current_step_idx" not in st.session_state:
    st.session_state.current_step_idx = 0

def set_step(idx: int):
    st.session_state.current_step_idx = idx

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/caduceus.png", width=64)
    st.title("Lifecycle Navigator")
    st.markdown("**Machine Learning Pipeline:**")
    
    selected_stage_str = st.radio(
        "Jump to Stage:",
        STAGES,
        index=st.session_state.current_step_idx
    )
    
    # Sync radio with session state
    st.session_state.current_step_idx = STAGES.index(selected_stage_str)
    
    st.markdown("---")
    st.markdown("""
    **Project Metrics Summary**:
    - **Dataset**: 253,680 records (CDC BRFSS)
    - **Champion Model**: LightGBM (Balanced)
    - **Holdout Test PR-AUC**: **0.4249**
    - **Holdout Test ROC-AUC**: **0.8277**
    - **Clinical Recall**: **79.40%**
    - **Inference Latency**: **< 8 ms**
    """)

curr_idx = st.session_state.current_step_idx
curr_stage = STAGES[curr_idx]

# Top Visual Stepper
st.markdown("### 🔄 End-to-End Machine Learning Journey")
step_cols = st.columns(7)
for i, name in enumerate(STAGES):
    label = name.split(". ")[1]
    if i == curr_idx:
        step_cols[i].button(f"👉 {label}", key=f"top_btn_{i}", on_click=set_step, args=(i,), use_container_width=True)
    elif i < curr_idx:
        step_cols[i].button(f"✓ {label}", key=f"top_btn_{i}", on_click=set_step, args=(i,), use_container_width=True)
    else:
        step_cols[i].button(f"⏳ {label}", key=f"top_btn_{i}", on_click=set_step, args=(i,), use_container_width=True)

st.markdown("---")


# ==============================================================================
# 1. RAW DATA
# ==============================================================================
if curr_idx == 0:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">📁 Stage 1: Raw Data Ingestion & Codebook</div>
        <div class="sub-title">CDC Behavioral Risk Factor Surveillance System (BRFSS) survey dataset via UC Irvine ML Repository (ID: 891).</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Survey Respondents", "253,680")
    c2.metric("Features Collected", "21 Indicators")
    c3.metric("Target Variable", "Diabetes_binary (0 vs 1)")
    c4.metric("Completeness", "100% (0 Missing Values)")

    st.markdown("#### 🔍 Raw Dataset Sample (First 10 Rows)")
    if RAW_DATA_PATH.exists():
        df_sample = pd.read_csv(RAW_DATA_PATH, nrows=10)
        st.dataframe(df_sample, use_container_width=True)
    else:
        st.warning("Raw dataset file is downloading or located in cache.")

    st.markdown("#### 📊 Target Variable Class Distribution & Medical Imbalance")
    img_target = FIGURES_DIR / "01_target_distribution.png"
    if img_target.exists():
        st.image(str(img_target), caption="Figure 1: Target Variable Distribution (86.07% Non-Diabetic vs 13.93% Diabetic)", use_container_width=True)

    st.info("""
    **Clinical Significance**: 
    - 218,334 respondents (**86.07%**) report no diabetes diagnosis.
    - 35,346 respondents (**13.93%**) report prediabetes or diabetes.
    - Because the negative class heavily dominates, standard Accuracy is deeply deceptive: a dummy classifier predicting '0' gets 86.07% accuracy while missing **100% of diabetic patients**.
    - All downstream stages must optimize for **PR-AUC (Average Precision)** and **Recall (Sensitivity)**.
    """)


# ==============================================================================
# 2. DATA CLEANING
# ==============================================================================
elif curr_idx == 1:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🧼 Stage 2: Data Cleaning & Leakage-Free Splitting</div>
        <div class="sub-title">Stratified partitioning and outlier-robust scaling fitted strictly on training observations.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Stratified Training Partition", "202,944 samples (80.0%)")
    c2.metric("Holdout Test Partition", "50,736 samples (20.0%)")
    c3.metric("Scaling Strategy", "RobustScaler (Median / IQR)")

    st.markdown("#### 🛡️ Zero Data Leakage Guarantee")
    st.markdown("""
    To ensure clinical validity and eliminate optimistic bias:
    1. **Strict Stratified Split First**: Partitioning occurs **before** any scaling or transformation.
    2. **Independent Scaler Fitting**: `RobustScaler` parameters (medians and interquartile ranges) are computed exclusively on `X_train`. The test set is strictly transformed using training statistics.
    """)

    img_scale = FIGURES_DIR / "15_scaling_distribution_comparison.png"
    if img_scale.exists():
        st.image(str(img_scale), caption="Figure 15: Outlier-Robust Feature Distributions Before and After RobustScaler", use_container_width=True)

    st.markdown("#### 🩺 Skewness & Outlier Treatment")
    st.markdown("""
    - `BMI` values extend up to 98.0 kg/m² with severe right skewness.
    - `PhysHlth` and `MentHlth` range from 0 to 30 days with over 60% of respondents reporting 0 unhealthy days.
    - `RobustScaler` prevents extreme outliers from compressing normal variance compared to standard z-score normalization.
    """)


# ==============================================================================
# 3. FEATURE ENGINEERING
# ==============================================================================
elif curr_idx == 2:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🛠️ Stage 3: Domain-Driven Feature Engineering</div>
        <div class="sub-title">Formulating 7 clinically motivated interaction variables based on epidemiological insights.</div>
    </div>
    """, unsafe_allow_html=True)

    eng_features_data = [
        {"Engineered Variable": "Comorbidity_Score", "Clinical Rationale & Formula": "Sum of HighBP + HighChol + HeartDisease + Stroke (0 to 4)", "Prevalence Impact": "Escalates from 2.4% (score 0) to 48.8% (score 4)"},
        {"Engineered Variable": "Obesity_Flag", "Clinical Rationale & Formula": "Binary indicator for BMI ≥ 30.0 (WHO/CDC Class I+ Obesity)", "Prevalence Impact": "22.5% in obese cohort vs 9.1% non-obese"},
        {"Engineered Variable": "Severe_Obesity_Flag", "Clinical Rationale & Formula": "Binary indicator for BMI ≥ 35.0 (Class II+ Severe Obesity)", "Prevalence Impact": "Prevalence surges to 27.8%"},
        {"Engineered Variable": "Metabolic_Syndrome_Proxy", "Clinical Rationale & Formula": "Co-occurrence of BMI ≥ 30 + HighBP=1 + HighChol=1", "Prevalence Impact": "34.2% prevalence in triad-positive individuals"},
        {"Engineered Variable": "Unhealthy_Days_Total", "Clinical Rationale & Formula": "PhysHlth + MentHlth cumulative burden (0 to 60 days)", "Prevalence Impact": "Continuous multi-system disability burden"},
        {"Engineered Variable": "High_Risk_Age", "Clinical Rationale & Formula": "Age group ≥ 9 (Individuals aged 60 and older)", "Prevalence Impact": "20.8% in seniors vs 7.2% under 60"},
        {"Engineered Variable": "Socioeconomic_Stress", "Clinical Rationale & Formula": "Joint lower education (≤3) and lower income (≤3)", "Prevalence Impact": "24.1% prevalence in high vulnerability cohort"}
    ]

    st.dataframe(pd.DataFrame(eng_features_data), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        img_eng = FIGURES_DIR / "16_engineered_features_impact.png"
        if img_eng.exists():
            st.image(str(img_eng), caption="Figure 16: Risk Escalation of Engineered Features", use_container_width=True)
    with col_b:
        img_mi = FIGURES_DIR / "17_feature_importance_mutual_info.png"
        if img_mi.exists():
            st.image(str(img_mi), caption="Figure 17: Mutual Information Feature Importance Ranking", use_container_width=True)


# ==============================================================================
# 4. MODEL TRAINING
# ==============================================================================
elif curr_idx == 3:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🤖 Stage 4: Model Training & Cross-Validation</div>
        <div class="sub-title">Training 5 diverse model families with Stratified 5-Fold Cross-Validation and cost-sensitive class weights.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Class 0 Weight (Negative)", "0.5809")
    c2.metric("Class 1 Weight (Diabetic)", "3.5885")
    c3.metric("Cross-Validation Folds", "Stratified 5-Fold")
    c4.metric("Champion Architecture", "LightGBM")

    img_resample = FIGURES_DIR / "18_class_imbalance_resampling_check.png"
    if img_resample.exists():
        st.image(str(img_resample), caption="Figure 18: Cost-Sensitive Balancing Strategy vs SMOTE", use_container_width=True)

    st.markdown("""
    ### 🏗️ 5 Candidate Architectures Trained & Evaluated:
    1. **Logistic Regression (Balanced)**: Convex baseline with L2 penalty, providing interpretable log-odds coefficients.
    2. **Random Forest (Balanced)**: 100 decision trees with max depth 12, bagging bootstrap samples with balanced sub-weights.
    3. **Extra Trees (Balanced)**: 100 extremely randomized trees with lower variance and randomized split thresholds.
    4. **LightGBM (Balanced)**: Fast histogram-based gradient booster (150 trees, max depth 6, num leaves 31, balanced class weights).
    5. **XGBoost (Weighted)**: Exact second-order gradient boosted trees with `scale_pos_weight = 3.5885`.
    """)


# ==============================================================================
# 5. MODEL EVALUATION
# ==============================================================================
elif curr_idx == 4:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">📊 Stage 5: Rigorous Model Evaluation & Benchmarks</div>
        <div class="sub-title">Evaluated strictly on the 50,736 unseen holdout test respondents across clinical metrics.</div>
    </div>
    """, unsafe_allow_html=True)

    if benchmark_data and "all_models_holdout" in benchmark_data:
        bench_df = pd.DataFrame(benchmark_data["all_models_holdout"]).T
        display_cols = ["PR_AUC", "ROC_AUC", "Recall", "F1_Score", "Specificity", "Balanced_Accuracy", "TP", "FN"]
        st.dataframe(bench_df[display_cols].sort_values(by="PR_AUC", ascending=False), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        img_bench = FIGURES_DIR / "19_model_benchmark_comparison.png"
        if img_bench.exists():
            st.image(str(img_bench), caption="Figure 19: Model Benchmark Comparison (PR-AUC, ROC-AUC, Recall, F1)", use_container_width=True)

        img_roc = FIGURES_DIR / "21_roc_curves_comparison.png"
        if img_roc.exists():
            st.image(str(img_roc), caption="Figure 21: Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

        img_fi = FIGURES_DIR / "24_feature_importances_top_models.png"
        if img_fi.exists():
            st.image(str(img_fi), caption="Figure 24: Top 15 Most Predictive Features in Champion LightGBM", use_container_width=True)

    with col2:
        img_pr = FIGURES_DIR / "20_pr_curves_comparison.png"
        if img_pr.exists():
            st.image(str(img_pr), caption="Figure 20: Precision-Recall (PR) Curves vs No-Skill Baseline", use_container_width=True)

        img_cm = FIGURES_DIR / "22_confusion_matrices.png"
        if img_cm.exists():
            st.image(str(img_cm), caption="Figure 22: Normalized Confusion Matrices (Top 3 Performers)", use_container_width=True)

        img_thresh = FIGURES_DIR / "23_threshold_tuning_recall_precision.png"
        if img_thresh.exists():
            st.image(str(img_thresh), caption="Figure 23: Clinical Decision Threshold Optimization", use_container_width=True)


# ==============================================================================
# 6. DEPLOYMENT
# ==============================================================================
elif curr_idx == 5:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🚀 Stage 6: Production Deployment Architecture</div>
        <div class="sub-title">Dockerized orchestration, serialized model artifacts, and high-throughput prediction serving.</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Serving Latency", "< 8 ms / prediction")
    c2.metric("Serialized Model Size", "535 KB")
    c3.metric("Scaler Size", "871 bytes")
    c4.metric("Container Tech", "Docker & Compose")

    st.markdown("""
    ### 🏗️ Inference & Deployment Dataflow
    ```text
    ┌───────────────────────────┐
    │  Raw Survey Input Payload │ (21 Clinical/Demographic Variables via UI/REST)
    └──────────────┬────────────┘
                   │
                   ▼
    ┌───────────────────────────┐
    │  Dynamic Feature Engine   │ (Computes Comorbidity_Score, Obesity_Flag,
    │   (src/inference.py)      │  Metabolic_Syndrome_Proxy, Unhealthy_Days)
    └──────────────┬────────────┘
                   │
                   ▼
    ┌───────────────────────────┐
    │  Fitted RobustScaler      │ (Transforms BMI, PhysHlth, MentHlth, Unhealthy_Days
    │ (models/robust_scaler)    │  using frozen training medians & IQRs)
    └──────────────┬────────────┘
                   │
                   ▼
    ┌───────────────────────────┐
    │  Champion LightGBM Model  │ (predict_proba() yields calibrated risk percentage)
    └──────────────┬────────────┘
                   │
                   ▼
    ┌───────────────────────────┐
    │ Dual Decision Thresholds  │ (Screening: 0.50 Threshold -> 79.4% Sensitivity
    │   & Actionable Advice     │  Confirmation: 0.65 Threshold -> 0.4688 F1-Score)
    └───────────────────────────┘
    ```
    """)

    st.markdown("#### 🐳 Container Launch Commands")
    st.code("""
# 1. Build and run via Docker Compose
docker compose up --build

# 2. Direct local Python execution
streamlit run app.py --server.port=8501
    """, language="bash")


# ==============================================================================
# 7. PREDICTION (LIVE INFERENCE)
# ==============================================================================
elif curr_idx == 6:
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🩺 Stage 7: Real-Time Patient Risk Prediction</div>
        <div class="sub-title">Interactive clinical decision support tool evaluating individualized diabetes risk probabilities.</div>
    </div>
    """, unsafe_allow_html=True)

    # Preset Patient Profiles
    st.markdown("### ⚡ Quick-Load Clinical Scenarios")
    preset_cols = st.columns(3)
    preset_choice = preset_cols[0].button("👤 Healthy Young Adult", use_container_width=True)
    preset_choice_mod = preset_cols[1].button("⚠️ Middle-Aged with Obesity", use_container_width=True)
    preset_choice_high = preset_cols[2].button("🚨 Senior with Metabolic Triad", use_container_width=True)

    default_vals = {
        "HighBP": 0, "HighChol": 0, "CholCheck": 1, "BMI": 23.5, "Smoker": 0,
        "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
        "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
        "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 0,
        "Age": 3, "Education": 6, "Income": 8
    }

    if preset_choice_mod:
        default_vals = {
            "HighBP": 0, "HighChol": 1, "CholCheck": 1, "BMI": 33.2, "Smoker": 1,
            "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0, "Fruits": 0,
            "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
            "GenHlth": 3, "MentHlth": 4, "PhysHlth": 5, "DiffWalk": 0, "Sex": 1,
            "Age": 7, "Education": 4, "Income": 5
        }
    elif preset_choice_high:
        default_vals = {
            "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 36.8, "Smoker": 1,
            "Stroke": 0, "HeartDiseaseorAttack": 1, "PhysActivity": 0, "Fruits": 0,
            "Veggies": 0, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 1,
            "GenHlth": 4, "MentHlth": 8, "PhysHlth": 14, "DiffWalk": 1, "Sex": 1,
            "Age": 10, "Education": 3, "Income": 2
        }

    st.markdown("---")
    st.markdown("### 📋 Enter Patient Health Indicators")

    with st.form("live_patient_form"):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("#### 1. Demographics")
            age_options = {
                1: "18 - 24 years", 2: "25 - 29 years", 3: "30 - 34 years",
                4: "35 - 39 years", 5: "40 - 44 years", 6: "45 - 49 years",
                7: "50 - 54 years", 8: "55 - 59 years", 9: "60 - 64 years",
                10: "65 - 69 years", 11: "70 - 74 years", 12: "75 - 79 years",
                13: "80+ years"
            }
            age_val = st.selectbox("Age Bracket", options=list(age_options.keys()),
                                   format_func=lambda x: age_options[x],
                                   index=list(age_options.keys()).index(default_vals["Age"]))
            
            sex_val = st.radio("Biological Sex", options=[0, 1],
                               format_func=lambda x: "Female" if x == 0 else "Male",
                               index=default_vals["Sex"])
            
            edu_options = {
                1: "Never attended / Kindergarten", 2: "Elementary (Grades 1-8)",
                3: "Some High School (Grades 9-11)", 4: "High School Graduate",
                5: "Some College / Technical", 6: "College 4+ Years"
            }
            edu_val = st.selectbox("Education Level", options=list(edu_options.keys()),
                                   format_func=lambda x: edu_options[x],
                                   index=list(edu_options.keys()).index(default_vals["Education"]))
            
            inc_options = {
                1: "< $10,000", 2: "$10,000 - $15,000", 3: "$15,000 - $20,000",
                4: "$20,000 - $25,000", 5: "$25,000 - $35,000", 6: "$35,000 - $50,000",
                7: "$50,000 - $75,000", 8: "≥ $75,000"
            }
            inc_val = st.selectbox("Annual Household Income", options=list(inc_options.keys()),
                                   format_func=lambda x: inc_options[x],
                                   index=list(inc_options.keys()).index(default_vals["Income"]))

        with col2:
            st.markdown("#### 2. Clinical Biomarkers")
            bmi_val = st.slider("Body Mass Index (BMI kg/m²)", min_value=12.0, max_value=60.0,
                                value=float(default_vals["BMI"]), step=0.1)
            
            highbp_val = st.checkbox("Diagnosed High Blood Pressure", value=bool(default_vals["HighBP"]))
            highchol_val = st.checkbox("Diagnosed High Cholesterol", value=bool(default_vals["HighChol"]))
            cholcheck_val = st.checkbox("Cholesterol Checked in Last 5 Years", value=bool(default_vals["CholCheck"]))
            heart_val = st.checkbox("Heart Disease / Myocardial Infarction", value=bool(default_vals["HeartDiseaseorAttack"]))
            stroke_val = st.checkbox("History of Stroke", value=bool(default_vals["Stroke"]))

        with col3:
            st.markdown("#### 3. Lifestyle & Diet")
            smoker_val = st.checkbox("Smoked ≥ 100 Cigarettes Lifetime", value=bool(default_vals["Smoker"]))
            phys_act_val = st.checkbox("Engaged in Physical Activity (Past 30d)", value=bool(default_vals["PhysActivity"]))
            fruits_val = st.checkbox("Consumes Fruits ≥ 1 time/day", value=bool(default_vals["Fruits"]))
            veggies_val = st.checkbox("Consumes Vegetables ≥ 1 time/day", value=bool(default_vals["Veggies"]))
            alcohol_val = st.checkbox("Heavy Alcohol (Men ≥14/wk, Women ≥7/wk)", value=bool(default_vals["HvyAlcoholConsump"]))
            diffwalk_val = st.checkbox("Difficulty Walking / Climbing Stairs", value=bool(default_vals["DiffWalk"]))

        with col4:
            st.markdown("#### 4. General Well-Being")
            genhlth_options = {1: "1 - Excellent", 2: "2 - Very Good", 3: "3 - Good", 4: "4 - Fair", 5: "5 - Poor"}
            genhlth_val = st.selectbox("Self-Rated General Health", options=list(genhlth_options.keys()),
                                       format_func=lambda x: genhlth_options[x],
                                       index=list(genhlth_options.keys()).index(default_vals["GenHlth"]))
            
            physhlth_val = st.slider("Physically Unhealthy Days (Past 30d)", min_value=0, max_value=30,
                                     value=int(default_vals["PhysHlth"]))
            menthlth_val = st.slider("Mentally Unhealthy Days (Past 30d)", min_value=0, max_value=30,
                                     value=int(default_vals["MentHlth"]))
            
            healthcare_val = st.checkbox("Has Healthcare Coverage", value=bool(default_vals["AnyHealthcare"]))
            nodoc_val = st.checkbox("Could Not Afford Doctor (Past 12m)", value=bool(default_vals["NoDocbcCost"]))

        submit_btn = st.form_submit_button("⚡ Compute Calibrated Diabetes Risk Assessment", use_container_width=True)

    patient_payload = {
        "HighBP": int(highbp_val), "HighChol": int(highchol_val), "CholCheck": int(cholcheck_val),
        "BMI": float(bmi_val), "Smoker": int(smoker_val), "Stroke": int(stroke_val),
        "HeartDiseaseorAttack": int(heart_val), "PhysActivity": int(phys_act_val),
        "Fruits": int(fruits_val), "Veggies": int(veggies_val),
        "HvyAlcoholConsump": int(alcohol_val), "AnyHealthcare": int(healthcare_val),
        "NoDocbcCost": int(nodoc_val), "GenHlth": int(genhlth_val),
        "MentHlth": int(menthlth_val), "PhysHlth": int(physhlth_val),
        "DiffWalk": int(diffwalk_val), "Sex": int(sex_val),
        "Age": int(age_val), "Education": int(edu_val), "Income": int(inc_val)
    }

    result = predict_patient_risk(patient_payload)

    # Assessment Results
    st.markdown("---")
    st.markdown("### 📊 Comprehensive Assessment Report")

    res_col1, res_col2 = st.columns([1, 1.3])

    with res_col1:
        st.markdown(f"""
        <div class="result-banner" style="background: {result['risk_color']};">
            <div style="font-size: 1.1rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; opacity: 0.9;">
                Calculated Diabetes Probability
            </div>
            <div style="font-size: 3.8rem; font-weight: 800; line-height: 1.1; margin: 0.5rem 0;">
                {result['risk_percentage']}%
            </div>
            <div style="font-size: 1.35rem; font-weight: 700; background: rgba(0,0,0,0.2); display: inline-block; padding: 0.35rem 1.2rem; border-radius: 9999px;">
                {result['risk_level']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(result["risk_probability"])

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            status_screening = "🚨 POSITIVE (Flagged)" if result["screening_positive"] else "✅ NEGATIVE"
            st.metric("Clinical Screening (Threshold 0.50)", status_screening,
                      help="Optimized for high sensitivity (79.4% Recall) to catch potential cases early.")
        with t_col2:
            status_confirm = "🚨 POSITIVE (High F1)" if result["confirmatory_positive"] else "✅ NEGATIVE"
            st.metric("Diagnostic Confirmation (Threshold 0.65)", status_confirm,
                      help="Optimized for maximum F1-Score (0.4688) and higher precision (37.9%).")

    with res_col2:
        st.markdown("#### 🔍 Active Clinical Risk Drivers")
        if result["risk_factors"]:
            for driver in result["risk_factors"]:
                st.markdown(f"<div class='risk-driver-item'>• {driver}</div>", unsafe_allow_html=True)
        else:
            st.success("No critical high-risk indicators detected. Profile displays healthy baseline markers.")

        st.markdown("#### 💡 Clinical Preventive Guidance")
        for rec in result["recommendations"]:
            st.markdown(f"<div class='recom-item'>✓ {rec}</div>", unsafe_allow_html=True)

    with st.expander("🛠️ View Dynamically Engineered Features for This Patient", expanded=False):
        eng = result["engineered_summary"]
        e_col1, e_col2, e_col3, e_col4 = st.columns(4)
        e_col1.metric("Comorbidity Score (0-4)", eng["Comorbidity_Score"])
        e_col2.metric("Obesity Flag (BMI ≥30)", "Yes" if eng["Obesity_Flag"] else "No")
        e_col3.metric("Metabolic Syndrome Proxy", "Positive" if eng["Metabolic_Syndrome_Proxy"] else "Negative")
        e_col4.metric("Total Unhealthy Days", f"{eng['Unhealthy_Days_Total']} days")


# Bottom Stepper Navigation Buttons
st.markdown("---")
btn_prev_col, _, btn_next_col = st.columns([1, 4, 1])

if curr_idx > 0:
    btn_prev_col.button(f"⬅ Previous: {STAGES[curr_idx - 1].split('. ')[1]}", on_click=set_step, args=(curr_idx - 1,), use_container_width=True)

if curr_idx < len(STAGES) - 1:
    btn_next_col.button(f"Next: {STAGES[curr_idx + 1].split('. ')[1]} ➔", on_click=set_step, args=(curr_idx + 1,), use_container_width=True)
