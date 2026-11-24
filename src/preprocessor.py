"""
Data preprocessing, feature engineering, and stratified splitting pipeline
for CDC Diabetes Health Indicators dataset.
Prevents data leakage by ensuring all scalers and transformations fit strictly on training splits.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "cdc_diabetes_health_indicators.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Continuous and skewed features that benefit from scaling
CONTINUOUS_FEATURES: List[str] = [
    "BMI",
    "PhysHlth",
    "MentHlth",
    "Unhealthy_Days_Total",
]

# Engineered feature definitions and clinical logic
ENGINEERED_FEATURE_DEFINITIONS: Dict[str, str] = {
    "Comorbidity_Score": "Integer count (0-4) of co-occurring conditions: HighBP + HighChol + HeartDiseaseorAttack + Stroke.",
    "Obesity_Flag": "Binary indicator for Body Mass Index >= 30.0 (WHO/CDC Class I+ Obesity).",
    "Severe_Obesity_Flag": "Binary indicator for Body Mass Index >= 35.0 (Class II+ Severe Obesity).",
    "Metabolic_Syndrome_Proxy": "Co-occurrence of obesity (BMI >= 30), hypertension (HighBP=1), and hyperlipidemia (HighChol=1).",
    "Unhealthy_Days_Total": "Sum of physically unhealthy and mentally unhealthy days in past 30 days (0-60).",
    "High_Risk_Age": "Binary indicator for individuals aged 60 and older (Age category >= 9).",
    "Socioeconomic_Stress": "Joint presence of lower educational attainment (<= Grade 11) and lower household income (<$20,000).",
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct clinically and empirically validated interaction features
    derived from Phase 1 Exploratory Data Analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw BRFSS survey features.
        
    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with original + engineered features.
    """
    df_out = df.copy()
    
    # 1. Multi-comorbidity cardiovascular count (0 to 4)
    df_out["Comorbidity_Score"] = (
        df_out["HighBP"] +
        df_out["HighChol"] +
        df_out["HeartDiseaseorAttack"] +
        df_out["Stroke"]
    ).astype(int)
    
    # 2. Obesity clinical thresholds
    df_out["Obesity_Flag"] = (df_out["BMI"] >= 30.0).astype(int)
    df_out["Severe_Obesity_Flag"] = (df_out["BMI"] >= 35.0).astype(int)
    
    # 3. Metabolic syndrome clinical triad proxy
    df_out["Metabolic_Syndrome_Proxy"] = (
        (df_out["BMI"] >= 30.0) &
        (df_out["HighBP"] == 1) &
        (df_out["HighChol"] == 1)
    ).astype(int)
    
    # 4. Total unhealthy days burden (capped at 60)
    df_out["Unhealthy_Days_Total"] = (df_out["PhysHlth"] + df_out["MentHlth"]).astype(int)
    
    # 5. Senior demographic risk indicator (Age group >= 9, corresponding to 60+ years old)
    df_out["High_Risk_Age"] = (df_out["Age"] >= 9).astype(int)
    
    # 6. Socioeconomic vulnerability proxy (Education <= 3 and Income <= 3)
    df_out["Socioeconomic_Stress"] = (
        (df_out["Education"] <= 3) &
        (df_out["Income"] <= 3)
    ).astype(int)
    
    return df_out


def split_data(
    df: pd.DataFrame,
    target_col: str = "Diabetes_binary",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Partition dataset into training and test sets using stratified sampling
    to strictly preserve the 86.1% : 13.9% class distribution across both splits.
    
    Parameters
    ----------
    df : pd.DataFrame
        Complete raw dataset.
    target_col : str, default='Diabetes_binary'
        Name of binary target column.
    test_size : float, default=0.2
        Holdout proportion (default 20%).
    random_state : int, default=42
        Reproducibility seed.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Stratified Split Completed:")
    print(f"  Training Set : {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Holdout Test : {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
    print(f"  Train Target Distribution: 0={sum(y_train==0)/len(y_train)*100:.2f}%, 1={sum(y_train==1)/len(y_train)*100:.2f}%")
    print(f"  Test Target Distribution : 0={sum(y_test==0)/len(y_test)*100:.2f}%, 1={sum(y_test==1)/len(y_test)*100:.2f}%")
    
    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
    scaler_type: str = "robust"
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Fit scaler strictly on training split continuous features, then transform
    both train and test sets.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature set.
    X_test : pd.DataFrame
        Holdout test feature set.
    continuous_cols : Optional[List[str]]
        List of continuous feature names to scale.
    scaler_type : str, default='robust'
        'robust' (RobustScaler: median and IQR) or 'standard' (StandardScaler).
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Any]
        X_train_scaled, X_test_scaled, fitted scaler object
    """
    if continuous_cols is None:
        continuous_cols = [c for c in CONTINUOUS_FEATURES if c in X_train.columns]
        
    scaler = RobustScaler() if scaler_type.lower() == "robust" else StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit scaler on train only
    scaler.fit(X_train[continuous_cols])
    
    # Transform both splits
    X_train_scaled[continuous_cols] = scaler.transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    print(f"Applied {scaler.__class__.__name__} to features: {continuous_cols}")
    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing_pipeline(
    save_dir: Optional[Path] = None,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute full Phase 2 data cleaning, splitting, engineering, scaling,
    and serialization workflow.
    """
    target_dir = save_dir if save_dir else PROCESSED_DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CDC DIABETES PREDICTION - PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)
    
    # 1. Load raw dataset
    from src.data_loader import load_cdc_diabetes_data
    df_raw = load_cdc_diabetes_data(file_path=RAW_DATA_PATH)
    
    # 2. Stratified train-test split (BEFORE engineering/scaling to prevent any leakage)
    X_train, X_test, y_train, y_test = split_data(
        df_raw,
        target_col="Diabetes_binary",
        test_size=0.2,
        random_state=random_state
    )
    
    # 3. Feature engineering applied independently to splits
    print("\nApplying domain feature engineering...")
    X_train_eng = engineer_features(X_train)
    X_test_eng = engineer_features(X_test)
    print(f"Engineered {len(X_train_eng.columns) - len(X_train.columns)} new interaction variables.")
    print(f"Total features: {len(X_train_eng.columns)}")
    
    # 4. Feature scaling (RobustScaler fitted strictly on training partition)
    print("\nFitting RobustScaler on continuous features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_eng,
        X_test_eng,
        continuous_cols=CONTINUOUS_FEATURES,
        scaler_type="robust"
    )
    
    # 5. Serialize processed splits
    print(f"\nSerializing processed splits to: {target_dir}")
    X_train_eng.to_csv(target_dir / "X_train.csv", index=False)
    X_test_eng.to_csv(target_dir / "X_test.csv", index=False)
    X_train_scaled.to_csv(target_dir / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(target_dir / "X_test_scaled.csv", index=False)
    y_train.to_csv(target_dir / "y_train.csv", index=False)
    y_test.to_csv(target_dir / "y_test.csv", index=False)
    
    # 6. Metadata summary
    metadata = {
        "dataset_name": "CDC Diabetes Health Indicators BRFSS 2015",
        "random_state": random_state,
        "split_ratio": {"train": 0.8, "test": 0.2},
        "train_samples": int(len(X_train_eng)),
        "test_samples": int(len(X_test_eng)),
        "target_col": "Diabetes_binary",
        "target_class_distribution": {
            "train_0_neg": int((y_train == 0).sum()),
            "train_1_pos": int((y_train == 1).sum()),
            "test_0_neg": int((y_test == 0).sum()),
            "test_1_pos": int((y_test == 1).sum()),
        },
        "original_features": list(X_train.columns),
        "engineered_features": list(ENGINEERED_FEATURE_DEFINITIONS.keys()),
        "engineered_definitions": ENGINEERED_FEATURE_DEFINITIONS,
        "continuous_features_scaled": CONTINUOUS_FEATURES,
        "total_feature_count": int(len(X_train_eng.columns)),
        "scaler_used": "RobustScaler"
    }
    
    with open(target_dir / "preprocessor_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Preprocessing metadata saved to: {target_dir / 'preprocessor_metadata.json'}")
    print("=" * 70)
    print("STAGE 4 & 5 PIPELINE COMPLETE!")
    print("=" * 70)
    
    return {
        "X_train": X_train_eng,
        "X_test": X_test_eng,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "metadata": metadata
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()
