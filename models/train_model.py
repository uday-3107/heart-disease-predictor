"""
Heart Disease Prediction - Model Training Script
Loads heart.csv, engineers features, splits data, trains 10 models, saves best.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from xgboost import XGBClassifier
import shap

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("LightGBM not available, skipping.")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR.parent / 'data'
MODELS_DIR = BASE_DIR
PLOT_DIR   = MODELS_DIR / 'plots'
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load & engineer features from heart.csv ────────────────────────────────────
print("=" * 60)
print("  HEART DISEASE PREDICTION — MODEL TRAINING")
print("=" * 60)

csv_path = DATA_DIR / 'heart.csv'
print(f"\nLoading dataset: {csv_path}")
df = pd.read_csv(csv_path)
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts().to_dict()}")

# ── Cleaning (aligned with day1_preprocessing.ipynb) ──────────────────────────
before = len(df)
df = df.drop_duplicates()
dupes = before - len(df)
print(f"Duplicates removed: {dupes}  (remaining: {len(df)})")

# Fix invalid values
thal_mode = df['thal'].mode()[0]
df['thal'] = df['thal'].replace(0, thal_mode)
ca_mode = df['ca'].mode()[0]
df['ca'] = df['ca'].replace(4, ca_mode)

# IQR capping
for col in ['chol', 'trestbps', 'oldpeak', 'thalach']:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = Q3 - Q1
    lo, hi = Q1 - 1.5*iqr, Q3 + 1.5*iqr
    df[col] = df[col].clip(lo, hi)

# ── Feature engineering ──────────────────────────────────────────────────────
df['age_hr_ratio'] = (df['age'] / df['thalach']).round(3)
df['high_chol']    = (df['chol'] > 240).astype(int)
df['bp_flag']      = (df['trestbps'] > 130).astype(int)

# One-hot encode cp, restecg, slope, thal (drop first category)
for col, cats in [('cp', [1,2,3]), ('restecg', [1,2]), ('slope', [1,2]), ('thal', [2,3])]:
    for c in cats:
        df[f'{col}_{c}'] = (df[col] == c).astype(int)

feature_cols = [
    'age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca',
    'age_hr_ratio', 'high_chol', 'bp_flag',
    'cp_1', 'cp_2', 'cp_3',
    'restecg_1', 'restecg_2',
    'slope_1', 'slope_2',
    'thal_2', 'thal_3',
]

X = df[feature_cols]
y = df['target']

# Train/test split (80/20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits for reference
# ── Scale (fit on train only — no data leakage) ───────────────────────────────
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols, index=X_test.index)

X_train.to_csv(DATA_DIR / 'X_train.csv', index=False)
X_test.to_csv(DATA_DIR / 'X_test.csv', index=False)
y_train.to_csv(DATA_DIR / 'y_train.csv', index=False)
y_test.to_csv(DATA_DIR / 'y_test.csv', index=False)

print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Train target: {y_train.value_counts().to_dict()}")
print(f"Test  target: {y_test.value_counts().to_dict()}")

# ── Define 10 models ───────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        C=0.1, max_iter=1000, class_weight='balanced', random_state=42),

    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=2,
        class_weight='balanced', random_state=42),

    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss', verbosity=0),

    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.7, random_state=42),

    'Extra Trees': ExtraTreesClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced',
        random_state=42),

    'AdaBoost': AdaBoostClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42),

    'SVM': SVC(
        C=1.0, kernel='rbf', class_weight='balanced',
        probability=True, random_state=42),

    'KNN': KNeighborsClassifier(n_neighbors=7, metric='minkowski'),

    'Naive Bayes': GaussianNB(),
}

if HAS_LGBM:
    models['LightGBM'] = LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        num_leaves=15, class_weight='balanced',
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1)

# ── Train & evaluate ───────────────────────────────────────────────────────────
print(f"\nTraining {len(models)} models...\n")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

    results[name] = {
        'accuracy'   : round(accuracy_score(y_test, y_pred),  4),
        'precision'  : round(precision_score(y_test, y_pred), 4),
        'recall'     : round(recall_score(y_test, y_pred),    4),
        'f1'         : round(f1_score(y_test, y_pred),        4),
        'roc_auc'    : round(roc_auc_score(y_test, y_prob),   4),
        'cv_auc_mean': round(cv_auc.mean(), 4),
        'cv_auc_std' : round(cv_auc.std(),  4),
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob
    }
    r = results[name]
    print(f"  {name:<22}  Acc={r['accuracy']}  Recall={r['recall']}  "
          f"F1={r['f1']}  AUC={r['roc_auc']}  CV={r['cv_auc_mean']}±{r['cv_auc_std']}")

# ── Pick best (ROC-AUC) ────────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]['roc_auc'])
best_model = results[best_name]['model']
print(f"\n  Best model → {best_name}  (ROC-AUC = {results[best_name]['roc_auc']})")

# ── Hyperparameter Tuning ─────────────────────────────────────────────────────────
print(f"\n  Tuning hyperparameters for {best_name}...")
param_grids = {
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10]},
    'Random Forest': {'n_estimators': [100, 200], 'max_depth': [3, 5, 8], 'min_samples_leaf': [2, 4, 6]},
    'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 4], 'learning_rate': [0.01, 0.05], 'reg_lambda': [1, 10]},
    'Gradient Boosting': {'n_estimators': [50, 100], 'max_depth': [3, 4], 'learning_rate': [0.01, 0.05], 'subsample': [0.7, 0.8]},
    'Extra Trees': {'n_estimators': [100, 200], 'max_depth': [3, 5, 8]},
    'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
    'SVM': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
    'KNN': {'n_neighbors': [5, 7, 9]},
    'Naive Bayes': {}, # No hyperparameters to tune
    'LightGBM': {'n_estimators': [50, 100], 'max_depth': [3, 4], 'learning_rate': [0.01, 0.05], 'reg_alpha': [0.1, 1.0], 'reg_lambda': [0.1, 1.0]}
}

grid = param_grids.get(best_name, {})
if grid:
    gs = GridSearchCV(best_model, grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    print(f"  Tuned Params: {gs.best_params_}")
    
    # Save Tuning Artifacts
    with open(MODELS_DIR / 'best_params.json', 'w') as f:
        json.dump(gs.best_params_, f)
    
    # Update test metrics for the tuned model
    y_pred_tuned = best_model.predict(X_test)
    y_prob_tuned = best_model.predict_proba(X_test)[:, 1]
    
    tuned_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_tuned),
        'precision': precision_score(y_test, y_pred_tuned),
        'recall': recall_score(y_test, y_pred_tuned),
        'f1': f1_score(y_test, y_pred_tuned),
        'roc_auc': roc_auc_score(y_test, y_prob_tuned)
    }
    pd.DataFrame([tuned_metrics]).to_csv(MODELS_DIR / 'tuned_metrics.csv', index=False)
    
    print(f"  Tuned ROC-AUC: {tuned_metrics['roc_auc']:.4f}")
    print("  best_params.json and tuned_metrics.csv saved.")
else:
    print("  No hyperparameters to tune for this model.")

# ── Explainable AI (SHAP) ─────────────────────────────────────────────────────────
print("\n  Generating SHAP interpretations...")
try:
    # Use TreeExplainer for tree-based models, otherwise KernelExplainer
    if hasattr(best_model, 'feature_importances_'):
        explainer = shap.TreeExplainer(best_model)
    else:
        # Use a small sample for KernelExplainer as it is slow
        explainer = shap.KernelExplainer(best_model.predict_proba, shap.sample(X_train, 50))
    
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, shap_values can be a list [negative_class, positive_class]
    # We care about the positive class (Heart Disease)
    if isinstance(shap_values, list):
        shap_vals_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3: # some versions of shap
        shap_vals_to_plot = shap_values[:, :, 1]
    else:
        shap_vals_to_plot = shap_values

    # 1. Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, X_test, show=False)
    plt.title(f"SHAP Summary Plot - {best_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'shap_summary.png', dpi=150)
    plt.close()
    
    # 2. Bar Plot (Global Importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_to_plot, X_test, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance Bar Plot - {best_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'shap_bar.png', dpi=150)
    plt.close()
    
    # 3. Waterfall Plot (Local Interpretation for first test sample)
    # Waterfall plots require an Explanation object
    if hasattr(explainer, 'explain_instance'): # KernelExplainer
        # KernelExplainer's explain_instance returns a SHAP values object
        exp_val = explainer.expected_value
        # If binary, expected_value might be a list
        if isinstance(exp_val, (list, np.ndarray)) and len(exp_val) == 2:
            exp_val = exp_val[1]
        
        # We need an Explanation object for waterfall. 
        # We can construct it manually for a single sample.
        sample_idx = 0
        sample_shap = shap_vals_to_plot[sample_idx]
        
        plt.figure(figsize=(10, 6))
        shap.plots._waterfall.waterfall_plot(
            shap.Explanation(
                values=sample_shap,
                base_values=exp_val,
                data=X_test.iloc[sample_idx].values,
                feature_names=feature_cols
            ),
            show=False
        )
        plt.title(f"SHAP Waterfall Plot - Sample {sample_idx}", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'shap_waterfall.png', dpi=150)
        plt.close()
    else: # TreeExplainer
        # For TreeExplainer, we can use explainer(X) to get an Explanation object
        # Note: explainer(X) can be slow, so we use a single sample
        explainer_obj = explainer(X_test.iloc[[0]])
        # Again, binary classification might have an extra dimension
        if len(explainer_obj.shape) == 3:
            explainer_obj = explainer_obj[:, :, 1]
            
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explainer_obj[0], show=False)
        plt.title(f"SHAP Waterfall Plot - Sample 0", fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'shap_waterfall.png', dpi=150)
        plt.close()
        
    print("  SHAP plots (summary, bar, waterfall) saved.")
except Exception as e:
    print(f"  SHAP analysis failed: {e}")
    import traceback
    traceback.print_exc()

# ── Clinical Interpretability Report ───────────────────────────────────────────────
print("\n  Generating Clinical Interpretability Report...")
try:
    # Calculate mean absolute SHAP values for global importance
    if isinstance(shap_values, list):
        vals = np.abs(shap_values[1]).mean(axis=0)
    elif len(shap_values.shape) == 3:
        vals = np.abs(shap_values[:, :, 1]).mean(axis=0)
    else:
        vals = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': vals})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    # Medical descriptions for features
    medical_map = {
        'age': 'Age of the patient',
        'sex': 'Gender (1=Male, 0=Female)',
        'trestbps': 'Resting blood pressure',
        'chol': 'Serum cholesterol',
        'fbs': 'Fasting blood sugar',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina',
        'oldpeak': 'ST depression induced by exercise',
        'ca': 'Number of major vessels colored by flourosopy',
        'age_hr_ratio': 'Ratio of age to max heart rate',
        'high_chol': 'Presence of high cholesterol (>240 mg/dl)',
        'bp_flag': 'Presence of high resting blood pressure (>130 mmHg)',
        'cp_1': 'Chest pain type 1', 'cp_2': 'Chest pain type 2', 'cp_3': 'Chest pain type 3',
        'restecg_1': 'Resting ECG results 1', 'restecg_2': 'Resting ECG results 2',
        'slope_1': 'Slope of peak exercise ST segment 1', 'slope_2': 'Slope of peak exercise ST segment 2',
        'thal_2': 'Thalassemia 2', 'thal_3': 'Thalassemia 3'
    }
    
    report_content = [
        "CLINICAL INTERPRETABILITY REPORT",
        "=================================",
        f"Model Used: {best_name}",
        f"Performance (ROC-AUC): {roc_auc_score(y_test, y_prob_tuned if 'y_prob_tuned' in locals() else y_prob):.4f}",
        "\nOVERVIEW:",
        "This report identifies the key physiological and clinical markers that the AI model",
        "uses to predict the presence of heart disease. These markers indicate the most",
        "influential factors in the model's decision-making process.",
        "\nKEY CLINICAL DRIVERS (Top Features):",
    ]
    
    for i, row in importance_df.head(10).iterrows():
        feat = row['feature']
        imp = row['importance']
        desc = medical_map.get(feat, feat)
        report_content.append(f"- {desc} ({feat}): Importance Score = {imp:.4f}")
    
    report_content.append("\nCLINICAL INTERPRETATION:")
    report_content.append("High importance scores suggest that the model heavily weighs these features.")
    report_content.append("For instance, markers related to chest pain (cp), maximum heart rate (thalach),")
    report_content.append("and ST depression (oldpeak) are typically strong predictors of cardiac events.")
    report_content.append("Physicians should prioritize these factors when interpreting model predictions.")
    report_content.append("\nDISCLAIMER:")
    report_content.append("This AI report is for clinical decision support and should not replace")
    report_content.append("professional medical judgment.")

    with open(MODELS_DIR / 'clinical_report.txt', 'w') as f:
        f.write("\n".join(report_content))
    print("  clinical_report.txt saved.")
except Exception as e:
    print(f"  Clinical report generation failed: {e}")

# ── Save artifacts ─────────────────────────────────────────────────────────────
joblib.dump(best_model, MODELS_DIR / 'best_model.pkl')
joblib.dump({k: v['model'] for k, v in results.items()},
            MODELS_DIR / 'all_models.pkl')
with open(MODELS_DIR / 'feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)
with open(MODELS_DIR / 'best_model_name.json', 'w') as f:
    json.dump({'best_model': best_name}, f)

metrics_df = pd.DataFrame({
    k: {m: v for m, v in v.items() if m not in ('model','y_pred','y_prob')}
    for k, v in results.items()
}).T
metrics_df.to_csv(MODELS_DIR / 'metrics.csv')
print("\n  metrics.csv saved")

# ── PLOTS ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#151c28',
    'axes.facecolor':   '#151c28',
    'axes.edgecolor':   '#1e2d45',
    'axes.labelcolor':  '#8899aa',
    'xtick.color':      '#8899aa',
    'ytick.color':      '#8899aa',
    'text.color':       '#e0e0e0',
    'grid.color':       '#1e2d45',
    'legend.facecolor': '#1c2130',
    'legend.edgecolor': '#2a3550',
    'savefig.facecolor':'#151c28',
})

COLORS = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6',
          '#1abc9c','#e67e22','#7f8c8d','#e91e63','#00bcd4']

# 1. Confusion matrices
n = len(results)
cols_g = 5
rows_g = (n + cols_g - 1) // cols_g
fig, axes = plt.subplots(rows_g, cols_g, figsize=(cols_g*3.5, rows_g*3.5))
axes = axes.flatten()
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[i],
                xticklabels=['Healthy','Disease'],
                yticklabels=['Healthy','Disease'], cbar=False)
    axes[i].set_title(f"{name}\nACC={res['accuracy']}", fontsize=9)
    axes[i].set_xlabel('Predicted', fontsize=8)
    axes[i].set_ylabel('Actual', fontsize=8)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Confusion Matrices — All Models', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'confusion_matrices.png', dpi=150)
plt.close()

# 2. ROC curves
plt.figure(figsize=(9, 7))
for (name, res), color in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, color=color, lw=1.8,
             label=f"{name} ({res['roc_auc']})")
plt.plot([0,1],[0,1],'k--', alpha=0.4)
plt.xlabel('False Positive Rate', fontsize=11)
plt.ylabel('True Positive Rate', fontsize=11)
plt.title('ROC Curves — All Models', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'roc_curves.png', dpi=150)
plt.close()

# 3. Grouped bar — all metrics
metric_names = ['accuracy','precision','recall','f1','roc_auc']
model_names  = list(results.keys())
x = np.arange(len(metric_names))
width = 0.8 / len(model_names)
fig, ax = plt.subplots(figsize=(13, 6))
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metric_names]
    ax.bar(x + i*width - (len(model_names)-1)*width/2, vals, width,
           label=name, color=COLORS[i % len(COLORS)])
ax.set_xticks(x)
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_title('Model Performance Comparison — All Metrics', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.axhline(0.85, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'model_comparison.png', dpi=150)
plt.close()

# 4. ROC-AUC ranked bar
auc_series = pd.Series(
    {k: v['roc_auc'] for k, v in results.items()}
).sort_values()
colors_bar = ['#e74c3c' if n == best_name else '#3498db' for n in auc_series.index]
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(auc_series.index, auc_series.values, color=colors_bar)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel('ROC-AUC Score', fontsize=11)
ax.set_title('Model Ranking by ROC-AUC', fontsize=13, fontweight='bold')
for bar, val in zip(bars, auc_series.values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.axvline(0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'auc_ranking.png', dpi=150)
plt.close()

# 5. CV AUC comparison
cv_means = {k: v['cv_auc_mean'] for k, v in results.items()}
cv_stds  = {k: v['cv_auc_std']  for k, v in results.items()}
sorted_names = sorted(cv_means, key=cv_means.get)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(sorted_names,
        [cv_means[n] for n in sorted_names],
        xerr=[cv_stds[n] for n in sorted_names],
        color='#1abc9c', capsize=4, alpha=0.9)
ax.set_xlim(0.5, 1.05)
ax.set_xlabel('5-Fold CV ROC-AUC', fontsize=11)
ax.set_title('Cross-Validation AUC with Std Dev', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'cv_auc_comparison.png', dpi=150)
plt.close()

# 6. Feature importance (tree models)
tree_models = {k: v['model'] for k, v in results.items()
               if hasattr(v['model'], 'feature_importances_')}
if tree_models:
    fig, axes = plt.subplots(1, len(tree_models), figsize=(5*len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]
    for ax, (name, mdl) in zip(axes, tree_models.items()):
        imp = pd.Series(mdl.feature_importances_, index=feature_cols)
        top = imp.sort_values(ascending=True).tail(15)
        top.plot(kind='barh', ax=ax, color='#9b59b6')
        ax.set_title(f'{name}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Importance')
    fig.suptitle('Feature Importances — Tree-Based Models', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'feature_importance.png', dpi=150)
    plt.close()

print("  All plots saved to models/plots/")
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
