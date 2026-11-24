# 🏥 CDC Diabetes Health Indicators: Machine Learning Project

A production-grade machine learning project for early diabetes risk detection and exploratory health analytics, built on the **CDC Behavioral Risk Factor Surveillance System (BRFSS)** dataset hosted on the **UC Irvine Machine Learning Repository (Dataset ID: 891)**.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Codebook & Metadata](#-dataset-codebook--metadata)
- [Repository Structure](#-repository-structure)
- [Phase 1: Exploratory Data Analysis (EDA) Summary](#-phase-1-exploratory-data-analysis-eda-summary)
  - [Target Imbalance](#1-target-imbalance)
  - [Top Predictive Risk Factors](#2-top-predictive-risk-factors)
  - [Cardiovascular Comorbidities & Risk Escalation](#3-cardiovascular-comorbidities--risk-escalation)
  - [Social Determinants of Health](#4-social-determinants-of-health)
- [Quickstart: Local Installation](#-quickstart-local-installation)
- [Docker Containerization](#-docker-containerization)
- [Phase 2: Data Preprocessing & Feature Engineering Summary](#-phase-2-data-preprocessing--feature-engineering-summary-stages-4--5)
- [Phase 3: Model Selection, Cross-Validation & Benchmark Training](#-phase-3-model-selection-cross-validation--benchmark-training-stage-6)
  - [Evaluation Metric Philosophy](#1-evaluation-metric-philosophy)
  - [Candidate Models & Architecture](#2-candidate-models--architecture)
  - [Holdout Benchmark Comparison](#3-holdout-test-benchmark-comparison-50736-samples)
  - [Clinical Decision Threshold Optimization](#4-clinical-decision-threshold-optimization)
  - [Key Diagnostic Visualizations](#5-key-diagnostic-visualizations)
- [Stage 7 Roadmap: Hyperparameter Tuning & Explainability](#-stage-7-roadmap-hyperparameter-tuning--explainability)

---

## 🎯 Project Overview

Diabetes is a chronic metabolic disorder affecting hundreds of millions globally. Early screening via clinical and lifestyle indicators enables timely intervention, preventative lifestyle changes, and reduced long-term cardiovascular complications.

- **Primary Target**: `Diabetes_binary`
  - `0`: No diabetes diagnosis
  - `1`: Prediabetes or Diabetes diagnosis
- **Dataset Size**: 253,680 survey respondents across 21 indicator features + 1 target column.
- **Completeness**: 100% complete records (zero missing or null values).

---

## 📊 Dataset Codebook & Metadata

| Feature Name | Variable Type | Scale / Range | Description & Clinical Context |
| :--- | :--- | :--- | :--- |
| **`Diabetes_binary`** | Target (Binary) | 0, 1 | 0 = No diabetes, 1 = Prediabetes or Diabetes diagnosed by a doctor. |
| **`HighBP`** | Binary | 0, 1 | Adults told by a healthcare professional they have high blood pressure. |
| **`HighChol`** | Binary | 0, 1 | Adults told by a healthcare professional they have high blood cholesterol. |
| **`CholCheck`** | Binary | 0, 1 | Cholesterol check performed within the past 5 years. |
| **`BMI`** | Continuous | 12.0 – 98.0 | Body Mass Index ($kg/m^2$) derived from self-reported height and weight. |
| **`Smoker`** | Binary | 0, 1 | Smoked $\ge 100$ cigarettes in entire lifetime (5 packs). |
| **`Stroke`** | Binary | 0, 1 | Ever diagnosed with a stroke. |
| **`HeartDiseaseorAttack`**| Binary | 0, 1 | History of coronary heart disease (CHD) or myocardial infarction (heart attack). |
| **`PhysActivity`** | Binary | 0, 1 | Physical activity or exercise in past 30 days outside of normal job. |
| **`Fruits`** | Binary | 0, 1 | Consumes fruit $\ge 1$ time per day. |
| **`Veggies`** | Binary | 0, 1 | Consumes vegetables $\ge 1$ time per day. |
| **`HvyAlcoholConsump`** | Binary | 0, 1 | Heavy drinking (men $\ge 14$ drinks/week; women $\ge 7$ drinks/week). |
| **`AnyHealthcare`** | Binary | 0, 1 | Health insurance coverage, prepaid plans (HMOs), or Medicare. |
| **`NoDocbcCost`** | Binary | 0, 1 | Needed to see a doctor in past 12 months but could not afford cost. |
| **`GenHlth`** | Ordinal | 1 – 5 | Self-rated health status (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor). |
| **`MentHlth`** | Discrete | 0 – 30 | Days of poor mental health (stress, depression, emotional problems) in past 30 days. |
| **`PhysHlth`** | Discrete | 0 – 30 | Days of physical illness or injury in past 30 days. |
| **`DiffWalk`** | Binary | 0, 1 | Serious difficulty walking or climbing stairs. |
| **`Sex`** | Binary | 0, 1 | Biological sex (0 = Female, 1 = Male). |
| **`Age`** | Ordinal | 1 – 13 | 5-year age intervals (1=18-24, 2=25-29, ..., 9=60-64, ..., 13=80+). |
| **`Education`** | Ordinal | 1 – 6 | Educational attainment (1=Never/KG, 2=Grades 1-8, 3=Grades 9-11, 4=HS Grad, 5=Some College, 6=College 4+ yrs). |
| **`Income`** | Ordinal | 1 – 8 | Annual household income (1=<$10k, 2=$10k-15k, ..., 8=$\ge$ $75k). |

---

## 📁 Repository Structure

```
DiabetesPrediction_CDC/
│
├── data/
│   ├── raw/
│   │   └── cdc_diabetes_health_indicators.csv   # Raw dataset (253,680 rows, 22 columns)
│   └── processed/                               # Serialized Stage 4/5 datasets (Zero Leakage)
│       ├── X_train.csv                          # Unscaled train features (202,944 rows, 28 cols)
│       ├── X_test.csv                           # Unscaled test features (50,736 rows, 28 cols)
│       ├── X_train_scaled.csv                   # RobustScaled train features
│       ├── X_test_scaled.csv                    # RobustScaled test features (fitted on train)
│       ├── y_train.csv                          # Training labels (86:14 ratio)
│       ├── y_test.csv                           # Holdout test labels (86:14 ratio)
│       └── preprocessor_metadata.json           # Transformation specs & class weights
│
├── models/                                      # Serialized champion models & benchmarks
│   ├── best_diabetes_model.joblib               # Champion LightGBM classifier
│   └── benchmark_results.json                   # Comprehensive cross-model evaluation metrics
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb       # Master EDA Notebook (67 cells, fully executed)
│   ├── 02_preprocessing_feature_engineering.ipynb # Master Preprocessing Notebook (46 cells, fully executed)
│   └── 03_model_selection_and_training.ipynb    # Master Modeling & Benchmark Notebook (28 cells, fully executed)
│
├── reports/
│   └── figures/                                 # 24 Publication-grade (300 DPI) exported charts
│       ├── 01_target_distribution.png
│       ├── 02_bmi_distribution.png
│       ├── 03_mental_physical_health.png
│       ├── 04_bmi_vs_diabetes.png
│       ├── 05_age_vs_diabetes.png
│       ├── 06_sex_vs_diabetes.png
│       ├── 07_socioeconomic_gradient.png
│       ├── 08_clinical_comorbidities.png
│       ├── 09_comorbidity_escalation.png
│       ├── 10_lifestyle_mobility.png
│       ├── 11_feature_correlations.png
│       ├── 12_full_correlation_matrix.png
│       ├── 13_genhlth_age_interaction.png
│       ├── 14_bmi_activity_interaction.png
│       ├── 15_scaling_distribution_comparison.png
│       ├── 16_engineered_features_impact.png
│       ├── 17_feature_importance_mutual_info.png
│       ├── 18_class_imbalance_resampling_check.png
│       ├── 19_model_benchmark_comparison.png
│       ├── 20_pr_curves_comparison.png
│       ├── 21_roc_curves_comparison.png
│       ├── 22_confusion_matrices.png
│       ├── 23_threshold_tuning_recall_precision.png
│       └── 24_feature_importances_top_models.png
│
├── src/
│   ├── __init__.py                              # Package initialization
│   ├── data_loader.py                           # Automated UCI download, caching & validation
│   ├── eda_utils.py                             # Aesthetic settings, category decoders & plot helpers
│   ├── preprocessor.py                          # Modular feature engineering, scaling & splitting pipeline
│   └── model_trainer.py                         # Modular model evaluation, CV, and export pipeline
│
├── scripts/
│   ├── build_notebook.py                        # Automated nbformat generator for EDA notebook
│   ├── build_preprocessing_notebook.py          # Automated nbformat generator for Preprocessing notebook
│   └── build_modeling_notebook.py               # Automated nbformat generator for Modeling notebook
│
├── Dockerfile                                   # Python 3.11 container with complete scientific stack
├── docker-compose.yml                           # Compose orchestration with live directory mounting
├── .dockerignore                                # Lean container build exclusions
├── .gitignore                                   # Python, cache, and notebook checkpoint exclusions
├── requirements.txt                             # Pinned dependencies (including lightgbm & xgboost)
└── README.md                                    # Project documentation
```

---

## 🔬 Phase 1: Exploratory Data Analysis (EDA) Summary

### 1. Target Imbalance
- **No Diabetes (0)**: 218,334 respondents (**86.07%**)
- **Diabetes / Prediabetes (1)**: 35,346 respondents (**13.93%**)
- **Modeling Consequence**: A trivial majority classifier achieves 86.07% accuracy while detecting 0% of diabetic patients. Modeling must prioritize **PR-AUC**, **Recall**, and **F1-Score**.

### 2. Top Predictive Risk Factors
| Rank | Feature | Correlation ($r$) | Key Clinical Finding |
| :---: | :--- | :---: | :--- |
| **1** | `GenHlth` | **+0.303** | Strongest predictor. Prevalence jumps from **2.1%** in "Excellent" health to **38.4%** in "Poor" health. |
| **2** | `HighBP` | **+0.271** | **75.3%** of diabetic individuals have hypertension. Relative Risk (RR) = **4.54x**. |
| **3** | `BMI` | **+0.224** | Median BMI is **31.0** (obese) for diabetic cohort vs **27.0** for non-diabetic cohort. |
| **4** | `DiffWalk` | **+0.224** | Walking difficulty is **3.5x** more prevalent among diabetic individuals (**37.1%** vs **13.3%**). |
| **5** | `HighChol` | **+0.209** | **67.0%** of diabetic respondents have high cholesterol vs **37.9%** in non-diabetic group. |
| **6** | `Age` | **+0.185** | Strong monotonic age curve: **1.5%** in age 18-24 escalating to **22.8%** in age 70-74. |
| **7** | `HeartDiseaseorAttack` | **+0.180** | Cardiovascular event history is **3.1x** higher in the diabetic cohort. |
| **8** | `PhysHlth` | **+0.176** | Diabetic individuals report **3.1x** more days of physical illness per month. |

### 3. Cardiovascular Comorbidities & Risk Escalation
When counting co-occurring conditions (`HighBP`, `HighChol`, `HeartDiseaseorAttack`, `Stroke`):
- **0 Conditions**: **2.4%** diabetes prevalence
- **1 Condition**: **9.8%** diabetes prevalence
- **2 Conditions**: **24.5%** diabetes prevalence
- **3 Conditions**: **39.7%** diabetes prevalence
- **All 4 Conditions**: **48.8%** diabetes prevalence (nearly 1 in 2 patients)

### 4. Social Determinants of Health
- **Income Gradient**: Individuals in the lowest income bracket (<$10k) experience **23.5%** diabetes prevalence compared to only **7.8%** in the highest bracket ($\ge$ $75k) — a 3-fold disparity.
- **Education Gradient**: Prevalence drops steadily from **25.2%** (no high school) down to **9.4%** (college graduates).

---

## 🚀 Quickstart: Local Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.13)
- Git

### 1. Clone or Open Workspace
```bash
cd d:/Projects/DiabetesPrediction_CDC
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Fetch / Verify the Dataset
```bash
python src/data_loader.py
```
*Note: The dataset will be automatically downloaded from UC Irvine ML repository and cached at `data/raw/cdc_diabetes_health_indicators.csv`.*

### 5. Launch JupyterLab & Run the Master EDA Notebook
```bash
python -m jupyter lab
```
Navigate to `notebooks/01_exploratory_data_analysis.ipynb` and run the cells sequentially.

---

## 🐳 Docker Containerization

To run the entire pipeline in an isolated, reproducible container without installing dependencies locally:

### 1. Build and Run Container with Docker Compose
```bash
docker compose up --build
```

### 2. Access JupyterLab
Open your browser and navigate to:
```
http://localhost:8888
```
- Token authentication is disabled for instant local access.
- Live volume mounting (`.:/app`) ensures any notebook edits, saved plots, or dataset downloads sync immediately to your local filesystem.

### 3. Stop the Container
```bash
docker compose down
```

---

---

## ⚙️ Phase 2: Data Preprocessing & Feature Engineering Summary (Stages 4 & 5)

### 1. Leakage-Free Stratified Splitting
- **Training Set**: 202,944 samples (80.0%) | 0 = 86.07%, 1 = 13.93%
- **Holdout Test Set**: 50,736 samples (20.0%) | 0 = 86.07%, 1 = 13.93%
- **Integrity**: All transformations, scalers, and resampling techniques fit strictly on `X_train`.

### 2. 7 Newly Engineered Features
| Engineered Feature | Logic & Clinical Context | Impact on Diabetes Prevalence |
| :--- | :--- | :--- |
| **`Comorbidity_Score`** | Count of `HighBP` + `HighChol` + `HeartDisease` + `Stroke` (0 to 4). | Risk scales from **2.4%** (score=0) to **48.8%** (score=4). |
| **`Obesity_Flag`** | $BMI \ge 30.0$ (WHO/CDC Class I+ Obesity). | Prevalence is **22.5%** in obese individuals vs **9.1%** non-obese. |
| **`Severe_Obesity_Flag`** | $BMI \ge 35.0$ (Class II+ Severe Obesity). | Prevalence escalates to **27.8%** in severe obesity. |
| **`Metabolic_Syndrome_Proxy`** | Co-occurrence of $BMI \ge 30$, `HighBP`=1, and `HighChol`=1. | **34.2%** prevalence in individuals positive for triad. |
| **`Unhealthy_Days_Total`** | `PhysHlth` + `MentHlth` combined burden (0–60 days). | Continuous burden indicator with strong right skew. |
| **`High_Risk_Age`** | Age category $\ge 9$ (individuals aged 60 and older). | **20.8%** prevalence in seniors vs **7.2%** under 60. |
| **`Socioeconomic_Stress`** | Joint lower education ($\le 3$) and lower income ($\le 3$). | **24.1%** prevalence in high socioeconomic stress cohort. |

### 3. Feature Scaling & Feature Importance
- **`RobustScaler`**: Applied to `BMI`, `PhysHlth`, `MentHlth`, and `Unhealthy_Days_Total` using training median and IQR.
- **Top Importance (Mutual Info & RF Gini)**: `GenHlth`, `HighBP`, `Comorbidity_Score`, `BMI`, `Obesity_Flag`, `Age`, `High_Risk_Age`.

### 4. Class Imbalance Remediation
- **Class Weights**: Class 0 = `0.5809`, Class 1 = `3.5894` (for cost-sensitive algorithms).
- **SMOTE Resampling**: Established for training partition experiments.

---

## 🤖 Phase 3: Model Selection, Cross-Validation & Benchmark Training (Stage 6)

### 1. Evaluation Metric Philosophy
Because of the **86:14 class imbalance** (35,346 positive cases out of 253,680 survey records), a naive majority-class classifier achieves 86.07% accuracy while missing 100% of diabetic patients. We prioritized:
- **PR-AUC (Average Precision)**: Primary metric summarizing performance on the minority class across all operating points.
- **Recall (Sensitivity)**: Medical screening priority to minimize false negatives ($FN$).
- **F1-Score & Balanced Accuracy**: Balanced harmonic assessment between precision and recall.

### 2. Candidate Models & Architecture
Five diverse classifiers were trained with cost-sensitive class weighting (`class_weight='balanced'` / `scale_pos_weight=3.5885`) on `X_train_scaled.csv` (202,944 samples):
1. **Logistic Regression (Balanced)**: Convex linear benchmark ($L2$ penalty, `lbfgs`).
2. **Random Forest (Balanced)**: Bagging ensemble (100 estimators, max depth 12).
3. **Extra Trees (Balanced)**: Extremely randomized trees (100 estimators, max depth 12).
4. **LightGBM (Balanced)**: Histogram-based gradient booster (150 estimators, learning rate 0.08, num leaves 31).
5. **XGBoost (Weighted)**: Scaled gradient boosted trees (150 estimators, learning rate 0.08, max depth 5).

### 3. Holdout Test Benchmark Comparison (50,736 Samples)

Evaluated strictly on the untouched holdout test partition (`X_test_scaled.csv`):

| Model Name | PR-AUC (Avg Prec) | ROC-AUC | Recall (Sensitivity) | F1-Score | Specificity | Balanced Accuracy | True Positives ($TP$) | False Negatives ($FN$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **LightGBM (Balanced)** | **0.4249** | **0.8277** | **79.40%** | **0.4436** | 71.09% | **75.25%** | **5,613** | **1,456** |
| 🥈 **XGBoost (Weighted)** | 0.4241 | 0.8272 | 79.06% | 0.4421 | 71.09% | 75.08% | 5,589 | 1,480 |
| 🥉 **Random Forest (Balanced)** | 0.4169 | 0.8232 | 77.31% | 0.4437 | 72.29% | 74.80% | 5,465 | 1,604 |
| **Extra Trees (Balanced)** | 0.4099 | 0.8197 | 77.88% | 0.4401 | 71.50% | 74.69% | 5,505 | 1,564 |
| **Logistic Regression (Balanced)** | 0.3987 | 0.8196 | 76.32% | 0.4421 | 72.65% | 74.48% | 5,395 | 1,674 |

> **Key Finding**: **LightGBM (Balanced)** captured the highest **PR-AUC (0.4249)** and **ROC-AUC (0.8277)** with the highest recall (**79.40%**), correctly identifying **5,613 out of 7,069** diabetic individuals in the unseen test cohort.

### 4. Clinical Decision Threshold Optimization
Using the champion **LightGBM** model:
- **Default 0.50 Threshold**: Prioritizes recall (79.40%) for screening triage, yielding an F1-Score of 0.4436.
- **Optimal F1 Threshold (0.65)**: Shifts the precision-recall balance for targeted diagnostic confirmation:
  - **F1-Score**: **0.4688** (+$0.0252$ improvement)
  - **Recall**: **61.55%**
  - **Precision**: **37.85%** (vs 30.78% at default threshold)
  - **Balanced Accuracy**: **72.60%**

### 5. Key Diagnostic Visualizations
All 6 stage artifacts are exported at 300 DPI to `reports/figures/`:
- **`19_model_benchmark_comparison.png`**: Side-by-side comparison of PR-AUC, ROC-AUC, Recall, and F1.
- **`20_pr_curves_comparison.png`**: Precision-Recall curves illustrating superior area under curve for LightGBM and XGBoost.
- **`21_roc_curves_comparison.png`**: Receiver Operating Characteristic curves across all 5 models.
- **`22_confusion_matrices.png`**: Normalized confusion matrices highlighting low False Negative rates.
- **`23_threshold_tuning_recall_precision.png`**: Continuous sweep across thresholds [0.10, 0.90].
- **`24_feature_importances_top_models.png`**: Top predictors (`GenHlth`, `BMI`, `HighBP`, `Comorbidity_Score`, `Age`).

---

## 🗺️ Stage 7 Roadmap: Hyperparameter Tuning & Explainability

With baseline benchmarking complete and the champion model serialized to [`models/best_diabetes_model.joblib`](file:///d:/Projects/DiabetesPrediction_CDC/models/best_diabetes_model.joblib):

1. **Hyperparameter Fine-Tuning**:
   - Bayesian Optimization / Optuna for LightGBM and XGBoost (tuning `learning_rate`, `num_leaves`, `min_child_samples`, `colsample_bytree`, `reg_alpha`, `reg_lambda`).
2. **Probability Calibration**:
   - Evaluate Platt Scaling (Sigmoid) and Isotonic Regression with Brier score minimization to ensure predicted probabilities match real clinical risk frequencies.
3. **Clinical Interpretability & Explainability**:
   - Compute global and local **SHAP (SHapley Additive exPlanations)** values using `TreeExplainer`.
   - Generate summary beeswarm plots and waterfall plots for individual patient risk scores.
4. **Interactive Inference Service**:
   - Build a lightweight Streamlit/FastAPI web interface allowing clinicians or users to enter survey responses and receive instant calibrated risk scores.


