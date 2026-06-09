# Heart Disease Prediction System
## Project Documentation Report

**Project**: Heart Disease Risk Prediction using Machine Learning  
**Dataset**: Cleveland Heart Disease Dataset (UCI / Kaggle)  
**Models**: 10 ML algorithms compared  
**Tools**: Python 3.x, Pandas, Scikit-learn, XGBoost, LightGBM, Streamlit  

---

## 1. Project Overview

This project builds a clinical decision support system to predict the presence of heart disease in patients using structured clinical data. Ten machine learning models are trained, compared, and evaluated on the Cleveland Heart Disease dataset. The best-performing model is deployed through a professional multi-page Streamlit web application.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | Cleveland Heart Disease Dataset (UCI / Kaggle) |
| Original rows | ~1,025 (includes duplicates) |
| After cleaning | 302 unique patient records |
| Features (final) | 21 |
| Target | 1 = Heart disease present, 0 = Healthy |
| Class balance | 54.3% disease / 45.7% healthy |

### Clinical Features

| Feature | Description |
|---|---|
| age | Patient age (years) |
| sex | 1 = Male, 0 = Female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved (bpm) |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Major vessels coloured (0–3) |
| thal | Thalassemia type (1–3) |

---

## 3. Day 1 — Data Preparation & EDA

### 3.1 Cleaning
- Removed 723 duplicate rows
- Fixed `thal = 0` (invalid) → replaced with mode
- Fixed `ca = 4` (out of range) → replaced with mode
- No missing values found

### 3.2 Key EDA Findings
- Disease patients have significantly lower max heart rate (~139 vs 158 bpm)
- Disease patients show higher ST depression (oldpeak: 1.58 vs 0.57)
- `ca`, `exang`, `oldpeak`, `thalach` are top predictors
- Dataset is nearly balanced — no oversampling required

### 3.3 Outlier Handling
IQR-based capping (not deletion) applied to: `chol`, `trestbps`, `oldpeak`, `thalach`

### 3.4 Feature Engineering
| Feature | Formula | Rationale |
|---|---|---|
| age_hr_ratio | age / thalach | Age-adjusted cardiac strain |
| high_chol | chol > 240 → 1 | Clinical threshold (ACC/AHA) |
| bp_flag | trestbps > 130 → 1 | Stage 1 hypertension threshold |

### 3.5 Preprocessing
- One-hot encoding: `cp`, `restecg`, `slope`, `thal` (drop_first=True)
- Z-score scaling: StandardScaler, fitted on train set only
- Train/Test split: 80/20, stratified, random_state=42

---

## 4. Day 2 — Model Building & Evaluation

### 4.1 Models Trained (10)

| # | Model | Type |
|---|---|---|
| 1 | Logistic Regression | Linear |
| 2 | Random Forest | Ensemble — Bagging |
| 3 | XGBoost | Ensemble — Boosting |
| 4 | Gradient Boosting | Ensemble — Boosting |
| 5 | Extra Trees | Ensemble — Bagging |
| 6 | AdaBoost | Ensemble — Boosting |
| 7 | SVM (RBF kernel) | Kernel method |
| 8 | K-Nearest Neighbours | Instance-based |
| 9 | Naive Bayes | Probabilistic |
| 10 | LightGBM | Ensemble — Boosting |

### 4.2 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV AUC |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8525 | 0.9032 | 0.8182 | 0.8571 | 0.9015 | 0.9201 |
| Random Forest | 0.8197 | 0.8235 | 0.8485 | 0.8358 | 0.8885 | 0.8997 |
| XGBoost | 0.8197 | 0.8235 | 0.8485 | 0.8358 | 0.8582 | 0.8995 |
| Gradient Boosting | 0.8033 | 0.8485 | 0.7879 | 0.8125 | 0.8701 | 0.8996 |
| Extra Trees | 0.8033 | 0.8438 | 0.7879 | 0.8148 | 0.8896 | 0.9045 |
| AdaBoost | 0.8361 | 0.8485 | 0.8485 | 0.8485 | 0.8869 | 0.8900 |
| SVM | 0.8361 | 0.8485 | 0.8485 | 0.8485 | 0.8734 | 0.9017 |
| KNN | 0.8525 | 0.8750 | 0.8485 | 0.8615 | 0.9021 | 0.9012 |
| Naive Bayes | 0.8361 | 0.8485 | 0.8485 | 0.8485 | 0.8983 | 0.8859 |
| LightGBM | 0.8525 | 0.8710 | 0.8485 | 0.8615 | 0.8723 | 0.9085 |

**Best model: KNN** — ROC-AUC = 0.9021 on test set (tuned to 0.9064)

> ℹ️ **Pipeline note:** The training pipeline (`train_model.py`) mirrors the data preparation steps in `day1_preprocessing.ipynb`. The dataset is first deduplicated (723 duplicates → 302 unique records), invalid values fixed, IQR-capped for outliers, feature-engineered, scaled with `StandardScaler` (fit on train only), and split 80/20 stratified. The resulting metrics reflect generalisation on genuinely independent test samples. These values — KNN at ~0.90 AUC, Logistic Regression at ~0.90, tree models at ~0.86–0.89 — are realistic for the Cleveland dataset's limited size (302 unique patients) and are consistent with published benchmarks.

### 4.4 Hyperparameter Tuning

The best model (KNN) was further optimized using **GridSearchCV** with 5-fold stratified cross-validation:

| Model | Parameter Grid | Best Parameters | Tuned ROC-AUC |
|---|---|---|---|
| KNN | `n_neighbors: [5, 7, 9]` | `n_neighbors=9` | **0.9064** |

All other models were evaluated with their default hyperparameters (as specified in `train_model.py`). The tuned KNN model improves ROC-AUC from **0.9021 → 0.9064** (+0.43%), with the tuned parameters saved to `best_params.json` and tuned metrics to `tuned_metrics.csv`.

### 4.3 Metric Priority
For a medical classifier, **Recall** is the most critical metric. A false negative means a patient with heart disease is sent home undetected. The model selection uses ROC-AUC as the primary criterion since it captures performance across all thresholds.

---

## 5. Streamlit Application

### Pages
1. **Patient Prediction** — Enter clinical values, select any of 10 models, get instant risk score with probability bar
2. **Model Performance** — Full metrics table, ROC curves, confusion matrices, feature importances, AUC ranking
3. **Model Explorer** — Side-by-side comparison of any two models with difference table
4. **About** — Dataset info, model list, methodology, disclaimer

### How to Run

#### Prerequisites
- Python 3.9+ (tested on 3.13)
- Virtual environment recommended

#### Quick Start (with pre-trained models)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app (pre-trained models included)
cd streamlit_app && streamlit run app.py
```
The app opens at **http://localhost:8501**

#### Full Retrain (optional)
If you want to re-train all models from scratch:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all 10 models (generates all artifacts)
cd models && python train_model.py

# 3. Run the Streamlit app
cd ../streamlit_app && streamlit run app.py
```

#### Expected Training Output
```
KNN  Acc=0.8525  Recall=0.8485  F1=0.8615  AUC=0.9021
Best model: KNN
```

#### Troubleshooting
| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: best_model.pkl` | Run `cd models && python train_model.py` first |
| `FileNotFoundError: X_train.csv` | Run training script to generate data splits |
| `ModuleNotFoundError: _loss` (LightGBM) | Re-run `pip install lightgbm` in your environment |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |
| App shows "Models not trained" | Run `cd models && python train_model.py` |

---

## 6. Achievements

| Achievement | Detail |
|---|---|
| ✅ 10 models trained and compared | Linear, ensemble, kernel, instance, probabilistic |
| ✅ End-to-end pipeline | Raw CSV → cleaned → engineered → modelled → deployed |
| ✅ No data leakage | Scaler fit on train set only |
| ✅ Clinical insight | Recall prioritised, IQR capping, clinical thresholds used |
| ✅ Professional Streamlit UI | 4-page app with custom CSS, model selector, side-by-side explorer |
| ✅ Cross-validation | 5-fold StratifiedKFold for reliable generalisation estimates |
| ✅ Reproducible | random_state=42 throughout, all artifacts saved |

---

## 7. File Inventory

| File | Purpose |
|---|---|---|
| day1_preprocessing.ipynb | Data cleaning, EDA, feature engineering, train/test split (Day 1 notebook) |
| data/X_train.csv, X_test.csv | Scaled features |
| data/y_train.csv, y_test.csv | Labels |
| models/train_model.py | Training script (10 models) |
| models/best_model.pkl | Best saved model |
| models/all_models.pkl | All 10 models |
| models/metrics.csv | Full evaluation table |
| models/plots/ | 9 performance charts (ROC, confusion, feature importance, AUC ranking, CV AUC, model comparison, SHAP summary, SHAP bar, SHAP waterfall) |
| streamlit_app/app.py | 4-page web application |
| requirements.txt | Python dependencies |
| README.md | Setup and run guide |

---

*Documentation — Heart Disease Prediction ML Project*
