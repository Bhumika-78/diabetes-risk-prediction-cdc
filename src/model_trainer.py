"""
Model Training, Cross-Validation, and Evaluation Pipeline
for CDC Diabetes Health Indicators dataset.

Implements model benchmarking across Linear Baselines, Tree Ensembles,
and Gradient Boosted Trees with cost-sensitive class weighting,
decision threshold tuning, and production artifact serialization.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import xgboost as xgb

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
MODELS_DIR = BASE_DIR / "models"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load preprocessed and scaled train/test splits.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train_scaled, X_test_scaled, y_train, y_test
    """
    X_train_scaled = pd.read_csv(PROCESSED_DATA_DIR / "X_train_scaled.csv")
    X_test_scaled = pd.read_csv(PROCESSED_DATA_DIR / "X_test_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").squeeze("columns")

    return X_train_scaled, X_test_scaled, y_train, y_test


def get_model_candidates(random_state: int = 42) -> Dict[str, Any]:
    """
    Construct dictionary of candidate classifiers equipped with cost-sensitive
    class weighting to counteract the 86:14 class imbalance.

    Parameters
    ----------
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary of instantiated model pipelines.
    """
    scale_pos_weight = 3.5885  # class 0 / class 1 ratio in training split

    models = {
        "Logistic Regression (Balanced)": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=-1
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "Extra Trees (Balanced)": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "LightGBM (Balanced)": lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            class_weight="balanced",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        ),
        "XGBoost (Weighted)": xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive clinical and statistical classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary target values.
    y_pred : np.ndarray
        Binary predicted labels.
    y_prob : np.ndarray
        Predicted probabilities for the positive class (Diabetes=1).

    Returns
    -------
    Dict[str, float]
        Dictionary of calculated evaluation metrics.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity),
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
        "PR_AUC": float(average_precision_score(y_true, y_prob)),
        "Brier_Score": float(brier_score_loss(y_true, y_prob)),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, pd.DataFrame]:
    """
    Scan classification thresholds between 0.05 and 0.95 to find the optimal
    decision boundary for clinical trade-offs.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_prob : np.ndarray
        Predicted probabilities for Class 1.
    metric : str
        Target optimization metric ('f1', 'balanced_accuracy', or 'recall_at_precision').

    Returns
    -------
    Tuple[float, pd.DataFrame]
        Optimal threshold value and sweep dataframe.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    records = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        records.append({
            "Threshold": thresh,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "Balanced_Accuracy": bal_acc
        })

    sweep_df = pd.DataFrame(records)
    if metric == "f1":
        best_idx = sweep_df["F1_Score"].idxmax()
    elif metric == "balanced_accuracy":
        best_idx = sweep_df["Balanced_Accuracy"].idxmax()
    else:
        best_idx = sweep_df["F1_Score"].idxmax()

    best_threshold = float(sweep_df.loc[best_idx, "Threshold"])
    return best_threshold, sweep_df
