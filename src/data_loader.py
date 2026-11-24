"""
Data ingestion and caching module for CDC Diabetes Health Indicators dataset.
Retrieves data via UCI Machine Learning Repository (ID: 891) or loads from local cache.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd


# Default project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "cdc_diabetes_health_indicators.csv"

# Dataset feature codebook and metadata
FEATURE_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "Diabetes_binary": {
        "description": "0 = No diabetes, 1 = Prediabetes or Diabetes",
        "type": "Target (Binary)",
        "clinical_context": "Primary prediction target derived from CDC BRFSS question: '(Ever told) you have diabetes?'"
    },
    "HighBP": {
        "description": "0 = No high BP, 1 = High BP diagnosed",
        "type": "Binary",
        "clinical_context": "Adults who have been told by a doctor/nurse they have high blood pressure."
    },
    "HighChol": {
        "description": "0 = No high cholesterol, 1 = High cholesterol",
        "type": "Binary",
        "clinical_context": "Adults told they have high blood cholesterol."
    },
    "CholCheck": {
        "description": "0 = No cholesterol check in 5 years, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Cholesterol check performed within the past five years."
    },
    "BMI": {
        "description": "Body Mass Index (kg/m^2)",
        "type": "Continuous",
        "clinical_context": "Metric calculated from self-reported height and weight."
    },
    "Smoker": {
        "description": "0 = No, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Have you smoked at least 100 cigarettes in your entire life? (5 packs = 100 cigarettes)."
    },
    "Stroke": {
        "description": "0 = No, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Ever told by a doctor, nurse, or health professional that you had a stroke."
    },
    "HeartDiseaseorAttack": {
        "description": "0 = No, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Reported coronary heart disease (CHD) or myocardial infarction (heart attack)."
    },
    "PhysActivity": {
        "description": "0 = No, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Engaged in physical activity or exercise during past 30 days other than regular job."
    },
    "Fruits": {
        "description": "0 = No, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Consume fruit 1 or more times per day."
    },
    "Veggies": {
        "description": "0 = No, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Consume vegetables 1 or more times per day."
    },
    "HvyAlcoholConsump": {
        "description": "0 = No, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Heavy alcohol consumption: adult men >=14 drinks/week; adult women >=7 drinks/week."
    },
    "AnyHealthcare": {
        "description": "0 = No, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Any healthcare coverage including health insurance, prepaid plans (HMOs), Medicare."
    },
    "NoDocbcCost": {
        "description": "0 = No, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Was there a time in past 12 months needed to see doctor but could not due to cost?"
    },
    "GenHlth": {
        "description": "Scale 1-5 (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor)",
        "type": "Ordinal",
        "clinical_context": "Self-rated general health status."
    },
    "MentHlth": {
        "description": "0-30 days",
        "type": "Discrete / Continuous",
        "clinical_context": "Number of days mental health (stress, depression, problems with emotions) was not good in past 30 days."
    },
    "PhysHlth": {
        "description": "0-30 days",
        "type": "Discrete / Continuous",
        "clinical_context": "Number of days physical illness or injury was experienced in past 30 days."
    },
    "DiffWalk": {
        "description": "0 = No, 1 = Yes",
        "type": "Binary",
        "clinical_context": "Serious difficulty walking or climbing stairs."
    },
    "Sex": {
        "description": "0 = Female, 1 = Male",
        "type": "Binary",
        "clinical_context": "Biological sex reported by respondent."
    },
    "Age": {
        "description": "13-level category: 1=18-24, 2=25-29, 3=30-34, ..., 9=60-64, ..., 13=80+",
        "type": "Ordinal",
        "clinical_context": "Age groupings categorized in 5-year intervals."
    },
    "Education": {
        "description": "Scale 1-6 (1=None/KG, 2=Grades 1-8, 3=Grades 9-11, 4=High school grad, 5=Some college, 6=College 4+ yrs)",
        "type": "Ordinal",
        "clinical_context": "Highest grade or year of school completed."
    },
    "Income": {
        "description": "Scale 1-8 (1=<$10k, 2=$10k-15k, 3=$15k-20k, 4=$20k-25k, 5=$25k-35k, 6=$35k-50k, 7=$50k-75k, 8=>$75k)",
        "type": "Ordinal",
        "clinical_context": "Annual household income from all sources."
    }
}


def get_feature_definitions() -> Dict[str, Dict[str, str]]:
    """Return dictionary of dataset feature codebook and metadata."""
    return FEATURE_DEFINITIONS


def download_from_uci(save_path: Path) -> pd.DataFrame:
    """
    Download CDC Diabetes Health Indicators directly via the ucimlrepo API (ID: 891).
    """
    print("Connecting to UCI Machine Learning Repository (Dataset ID: 891)...")
    try:
        from ucimlrepo import fetch_ucirepo
        cdc_data = fetch_ucirepo(id=891)
        
        X = cdc_data.data.features
        y = cdc_data.data.targets
        
        # Ensure target column is named Diabetes_binary
        if isinstance(y, pd.DataFrame):
            df = pd.concat([y, X], axis=1)
        else:
            y_df = pd.DataFrame({"Diabetes_binary": y})
            df = pd.concat([y_df, X], axis=1)
            
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset successfully downloaded and cached at: {save_path}")
        print(f"Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Failed to fetch from ucimlrepo: {e}")
        raise RuntimeError(
            "Could not fetch dataset from UCI ML repository. "
            "Please ensure internet connectivity is active or check repository status."
        ) from e


def load_cdc_diabetes_data(force_download: bool = False, file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load CDC Diabetes Health Indicators dataset.
    Uses local cached CSV if available, otherwise downloads automatically from UCI.
    
    Parameters
    ----------
    force_download : bool, default=False
        If True, ignores cached file and downloads fresh copy.
    file_path : Optional[Path], default=None
        Custom file path to read/write dataset. Defaults to data/raw/cdc_diabetes_health_indicators.csv.
        
    Returns
    -------
    pd.DataFrame
        Complete DataFrame with 253,680 rows and 22 columns.
    """
    target_path = Path(file_path) if file_path else RAW_DATA_FILE
    
    if target_path.exists() and not force_download:
        print(f"Loading cached dataset from: {target_path}")
        df = pd.read_csv(target_path)
        print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
        return df
    
    return download_from_uci(target_path)


if __name__ == "__main__":
    print("=" * 70)
    print("CDC DIABETES HEALTH INDICATORS - DATA INGESTION")
    print("=" * 70)
    df = load_cdc_diabetes_data()
    print("\nFirst 3 rows:")
    print(df.head(3))
    print("\nClass distribution:")
    print(df["Diabetes_binary"].value_counts(normalize=True).round(4) * 100)
    print("=" * 70)
