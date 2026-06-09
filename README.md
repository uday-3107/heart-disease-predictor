# Heart Disease Prediction System

A machine learning system that predicts heart disease risk using the Cleveland Heart Disease dataset, with a professional Streamlit web interface.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Re-train models — pre-trained models already included
cd models && python train_model.py

# 3. Run the Streamlit app
cd ../streamlit_app && streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Project Structure

```
heart_disease_project/
├── data/
│   ├── X_train.csv          # Scaled training features (241 × 21)
│   ├── X_test.csv           # Scaled test features (61 × 21)
│   ├── y_train.csv          # Training labels (241)
│   └── y_test.csv           # Test labels (61)
├── models/
│   ├── train_model.py       # Model training script (10 models)
│   ├── best_model.pkl       # Best model (KNN, pre-trained)
│   ├── all_models.pkl       # All 10 trained models (pre-trained)
│   ├── feature_columns.json # Feature names (21)
│   ├── metrics.csv          # Evaluation results (pre-computed)
│   ├── best_params.json     # Best model hyperparameters
│   ├── tuned_metrics.csv    # Tuned model metrics
│   ├── clinical_report.txt  # SHAP-based clinical interpretation
│   └── plots/               # Performance charts (9 PNGs)
├── streamlit_app/
│   └── app.py               # 4-tab Streamlit web application
├── docs/
│   └── documentation.md     # Full project report
├── day1_preprocessing.ipynb # Day 1 EDA & preprocessing notebook
├── requirements.txt
└── README.md
```

---

## Models (Pre-trained)

| # | Model | Type | ROC-AUC |
|---|---|---|---|
| 1 | KNN | Instance-based | **0.9021** (best) |
| 2 | Logistic Regression | Linear | 0.9015 |
| 3 | Naive Bayes | Probabilistic | 0.8983 |
| 4 | Extra Trees | Ensemble — Bagging | 0.8896 |
| 5 | Random Forest | Ensemble — Bagging | 0.8885 |
| 6 | AdaBoost | Ensemble — Boosting | 0.8869 |
| 7 | SVM (RBF) | Kernel method | 0.8734 |
| 8 | LightGBM | Ensemble — Boosting | 0.8723 |
| 9 | Gradient Boosting | Ensemble — Boosting | 0.8701 |
| 10 | XGBoost | Ensemble — Boosting | 0.8582 |

**Best model: KNN** (tuned n_neighbors=9, ROC-AUC=0.9064)

---

## Key Results (Best Model — KNN)

| Metric | Value |
|---|---|
| Accuracy | 0.8525 |
| Recall (Disease) | 0.8485 |
| F1-Score | 0.8615 |
| ROC-AUC | 0.9021 (tuned: 0.9064) |

*Metrics on 302 unique records after deduplication. 241 train / 61 test split.*

---

## Using the App

### Patient Prediction tab
1. Enter patient clinical values using the sliders and dropdowns
2. Click **Analyse Risk**
3. See the risk score, probability bar, and input summary

### Model Performance tab
- View metrics table (accuracy, recall, F1, ROC-AUC for all 10 models)
- View confusion matrices, ROC curves, feature importance charts

### About tab
- Dataset details, feature engineering summary, disclaimer

---

## Models

| # | Model | Type | Description |
|---|---|---|---|
| 1 | Logistic Regression | Linear | `C=0.1`, balanced class weight |
| 2 | Random Forest | Ensemble — Bagging | 200 trees, max depth 8, balanced |
| 3 | XGBoost | Ensemble — Boosting | 200 estimators, lr=0.05, depth 4 |
| 4 | Gradient Boosting | Ensemble — Boosting | 100 estimators, lr=0.05, depth 3 |
| 5 | Extra Trees | Ensemble — Bagging | 200 trees, max depth 8 |
| 6 | AdaBoost | Ensemble — Boosting | 100 estimators, lr=0.1 |
| 7 | SVM (RBF) | Kernel method | `C=1.0`, balanced class weight |
| 8 | KNN | Instance-based | 9 neighbours (tuned), minkowski metric |
| 9 | Naive Bayes | Probabilistic | GaussianNB |
| 10 | LightGBM | Ensemble — Boosting | 100 estimators, lr=0.05, depth 3 |

**Best model selection**: Based on highest ROC-AUC on test set. Recall is the clinically critical metric (false negatives = missed disease).

---

## Key Results (best model — KNN)

| Metric | Value |
|---|---|
| Accuracy | 0.8525 |
| Recall (Disease) | 0.8485 |
| F1-Score | 0.8615 |
| ROC-AUC | 0.9021 |

Metrics reflect 302 unique patient records after deduplication. See `docs/documentation.md` for full results and pipeline details.

---

## Troubleshooting

**`ModuleNotFoundError`** → Run `pip install -r requirements.txt`

**`FileNotFoundError: best_model.pkl`** → Run `python models/train_model.py` first

**Streamlit app shows "Model not found"** → Same as above — train first

**Port already in use** → `streamlit run app.py --server.port 8502`
