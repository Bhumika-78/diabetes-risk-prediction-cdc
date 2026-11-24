"""
Exploratory Data Analysis (EDA) visualization and statistical utilities.
Standardizes aesthetics, label decoding, and figure export for publication-quality output.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Project base directories
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Curated categorical and ordinal mapping dictionaries
AGE_LABELS: Dict[int, str] = {
    1: "18-24",
    2: "25-29",
    3: "30-34",
    4: "35-39",
    5: "40-44",
    6: "45-49",
    7: "50-54",
    8: "55-59",
    9: "60-64",
    10: "65-69",
    11: "70-74",
    12: "75-79",
    13: "80+",
}

EDUCATION_LABELS: Dict[int, str] = {
    1: "Never/KG",
    2: "Elem (1-8)",
    3: "High (9-11)",
    4: "HS Grad/GED",
    5: "Some College",
    6: "College 4+ yrs",
}

INCOME_LABELS: Dict[int, str] = {
    1: "<$10k",
    2: "$10k-$15k",
    3: "$15k-$20k",
    4: "$20k-$25k",
    5: "$25k-$35k",
    6: "$35k-$50k",
    7: "$50k-$75k",
    8: "≥$75k",
}

GEN_HLTH_LABELS: Dict[int, str] = {
    1: "Excellent",
    2: "Very Good",
    3: "Good",
    4: "Fair",
    5: "Poor",
}

SEX_LABELS: Dict[int, str] = {
    0: "Female",
    1: "Male",
}

DIABETES_LABELS: Dict[int, str] = {
    0: "No Diabetes",
    1: "Prediabetes / Diabetes",
}

# Color palette constants
PALETTE_BINARY = ["#3498db", "#e74c3c"]  # Blue for Non-Diabetic, Red/Coral for Diabetic
PALETTE_GRADIENT = "coolwarm"


def setup_visualization_style() -> None:
    """Set uniform aesthetic parameters for matplotlib and seaborn."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "grid.color": "#ebebeb",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> Path:
    """
    Export matplotlib figure to reports/figures/ directory with high resolution.
    """
    output_path = FIGURES_DIR / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"[Figure Saved] -> {output_path}")
    return output_path


def compute_prevalence(df: pd.DataFrame, group_col: str, target_col: str = "Diabetes_binary") -> pd.DataFrame:
    """
    Calculate total count, positive count, and prevalence percentage of target grouped by a feature.
    """
    agg_df = df.groupby(group_col)[target_col].agg(
        Total="count",
        Positive="sum"
    ).reset_index()
    agg_df["Negative"] = agg_df["Total"] - agg_df["Positive"]
    agg_df["Prevalence_Pct"] = (agg_df["Positive"] / agg_df["Total"]) * 100
    return agg_df
