"""
Inference and Real-Time Prediction Pipeline
for CDC Diabetes Health Indicators.

Loads serialized model, fits/loads RobustScaler, calculates engineered features,
and produces calibrated risk probability, risk category, and clinical recommendations.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_diabetes_model.joblib"
SCALER_PATH = MODELS_DIR / "robust_scaler.joblib"

CONTINUOUS_FEATURES = ["BMI", "PhysHlth", "MentHlth", "Unhealthy_Days_Total"]

EXPECTED_FEATURE_ORDER = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
    "Income", "Comorbidity_Score", "Obesity_Flag", "Severe_Obesity_Flag",
    "Metabolic_Syndrome_Proxy", "Unhealthy_Days_Total", "High_Risk_Age",
    "Socioeconomic_Stress"
]


def get_or_create_scaler() -> RobustScaler:
    """
    Load fitted RobustScaler from disk or fit on X_train.csv and persist.
    """
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    
    # Fit scaler strictly on unscaled training partition
    X_train_raw = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    scaler = RobustScaler()
    scaler.fit(X_train_raw[CONTINUOUS_FEATURES])
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def load_champion_model():
    """Load champion LightGBM model from models/ directory."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def engineer_single_patient(raw_features: Dict[str, Any]) -> pd.DataFrame:
    """
    Takes 21 raw survey indicators, derives 7 engineered features,
    and returns a 28-feature unscaled DataFrame.
    """
    df = pd.DataFrame([raw_features])
    
    # 1. Comorbidity Score (0 to 4)
    df["Comorbidity_Score"] = (
        int(df["HighBP"].iloc[0]) +
        int(df["HighChol"].iloc[0]) +
        int(df["HeartDiseaseorAttack"].iloc[0]) +
        int(df["Stroke"].iloc[0])
    )
    
    # 2. Obesity Flags
    bmi = float(df["BMI"].iloc[0])
    df["Obesity_Flag"] = int(bmi >= 30.0)
    df["Severe_Obesity_Flag"] = int(bmi >= 35.0)
    
    # 3. Metabolic Syndrome Proxy
    df["Metabolic_Syndrome_Proxy"] = int(
        (bmi >= 30.0) and
        (int(df["HighBP"].iloc[0]) == 1) and
        (int(df["HighChol"].iloc[0]) == 1)
    )
    
    # 4. Total Unhealthy Days
    df["Unhealthy_Days_Total"] = int(df["PhysHlth"].iloc[0]) + int(df["MentHlth"].iloc[0])
    
    # 5. Senior Risk Indicator (Age group >= 9, i.e., 60+)
    df["High_Risk_Age"] = int(int(df["Age"].iloc[0]) >= 9)
    
    # 6. Socioeconomic Stress (Education <= 3 and Income <= 3)
    df["Socioeconomic_Stress"] = int(
        (int(df["Education"].iloc[0]) <= 3) and
        (int(df["Income"].iloc[0]) <= 3)
    )
    
    # Ensure correct column ordering
    df = df[EXPECTED_FEATURE_ORDER]
    return df


def predict_patient_risk(raw_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete inference pipeline for a single patient profile.

    Parameters
    ----------
    raw_features : Dict[str, Any]
        Dictionary of 21 raw CDC health indicators.

    Returns
    -------
    Dict[str, Any]
        Probability, risk tier, classification flags, and clinical recommendations.
    """
    model = load_champion_model()
    scaler = get_or_create_scaler()
    
    # 1. Feature Engineering
    patient_df = engineer_single_patient(raw_features)
    
    # 2. Scaling continuous columns
    patient_df_scaled = patient_df.copy()
    patient_df_scaled[CONTINUOUS_FEATURES] = scaler.transform(patient_df[CONTINUOUS_FEATURES])
    
    # 3. Probability Inference
    risk_prob = float(model.predict_proba(patient_df_scaled)[0, 1])
    
    # 4. Threshold determinations
    screening_detected = bool(risk_prob >= 0.50)
    confirmatory_detected = bool(risk_prob >= 0.65)
    
    # 5. Clinical Risk Tier
    if risk_prob < 0.25:
        risk_level = "Low Risk"
        risk_color = "#10B981"  # Emerald Green
    elif risk_prob < 0.50:
        risk_level = "Moderate Risk"
        risk_color = "#F59E0B"  # Amber Orange
    elif risk_prob < 0.70:
        risk_level = "Elevated Risk"
        risk_color = "#EA580C"  # Dark Orange
    else:
        risk_level = "High / Critical Risk"
        risk_color = "#EF4444"  # Red
        
    # 6. Clinical Risk Drivers
    risk_factors = []
    if raw_features.get("HighBP", 0) == 1:
        risk_factors.append("Hypertension: Strongest vascular risk factor (RR = 4.5x).")
    if raw_features.get("HighChol", 0) == 1:
        risk_factors.append("High Cholesterol: Associated with 67% of diabetic individuals.")
    if float(raw_features.get("BMI", 25)) >= 30.0:
        risk_factors.append(f"Clinical Obesity (BMI {raw_features.get('BMI')} kg/m²): Significantly increases insulin resistance.")
    if int(raw_features.get("Age", 5)) >= 9:
        risk_factors.append("Age 60+: Age-related metabolic decline and reduced glucose tolerance.")
    if int(raw_features.get("GenHlth", 1)) >= 4:
        risk_factors.append("Fair/Poor Self-Rated Health: Correlates with elevated systemic health burden.")
    if int(raw_features.get("DiffWalk", 0)) == 1:
        risk_factors.append("Mobility Limitations: 3.5x higher prevalence among individuals with diabetes.")
    if int(raw_features.get("PhysActivity", 1)) == 0:
        risk_factors.append("Sedentary Lifestyle: Lack of physical activity contributes to metabolic risk.")

    # 7. Recommendations
    recommendations = []
    if risk_prob >= 0.50:
        recommendations.append("Schedule a comprehensive fasting plasma glucose (FPG) or HbA1c screening test.")
    if raw_features.get("HighBP", 0) == 1 or raw_features.get("HighChol", 0) == 1:
        recommendations.append("Prioritize cardiovascular monitoring and lipid panel management with your physician.")
    if float(raw_features.get("BMI", 25)) >= 25.0:
        recommendations.append("Incorporate nutritional adjustments to achieve a target 5-7% weight reduction.")
    if int(raw_features.get("PhysActivity", 1)) == 0:
        recommendations.append("Engage in moderate-intensity physical activity (e.g. brisk walking) at least 150 minutes per week.")
    if not recommendations:
        recommendations.append("Maintain routine annual health checkups, balanced diet, and active lifestyle habits.")

    return {
        "risk_probability": risk_prob,
        "risk_percentage": round(risk_prob * 100, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "screening_positive": screening_detected,
        "confirmatory_positive": confirmatory_detected,
        "risk_factors": risk_factors,
        "recommendations": recommendations,
        "engineered_summary": {
            "Comorbidity_Score": int(patient_df["Comorbidity_Score"].iloc[0]),
            "Obesity_Flag": int(patient_df["Obesity_Flag"].iloc[0]),
            "Severe_Obesity_Flag": int(patient_df["Severe_Obesity_Flag"].iloc[0]),
            "Metabolic_Syndrome_Proxy": int(patient_df["Metabolic_Syndrome_Proxy"].iloc[0]),
            "Unhealthy_Days_Total": int(patient_df["Unhealthy_Days_Total"].iloc[0]),
            "High_Risk_Age": int(patient_df["High_Risk_Age"].iloc[0]),
            "Socioeconomic_Stress": int(patient_df["Socioeconomic_Stress"].iloc[0]),
        }
    }
