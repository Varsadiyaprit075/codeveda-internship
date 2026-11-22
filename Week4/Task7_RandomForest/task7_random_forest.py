# ============================================================
# Codveda Technology - Machine Learning Internship
# Week 4 | Level 3 - Task 1: Random Forest Classifier
# Dataset : Customer Churn Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)

print("=" * 60)
print("  WEEK 4 | TASK 7 : RANDOM FOREST CLASSIFIER")
print("=" * 60)

# ── 1. Load & Preprocess ─────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df.drop(columns=['State','Area code'], inplace=True, errors='ignore')
    df['International plan'] = (df['International plan'] == 'Yes').astype(int)
    df['Voice mail plan']    = (df['Voice mail plan']    == 'Yes').astype(int)
    df['Churn'] = df['Churn'].map({'True':1, True:1, 'False':0, False:0})
    return df

train_df = preprocess(pd.read_csv("churn_train.csv"))
test_df  = preprocess(pd.read_csv("churn_test.csv"))

X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']
X_test  = test_df.drop('Churn', axis=1)
y_test  = test_df['Churn']

print(f"\n[1] Train: {X_train.shape} | Test: {X_test.shape}")

# ── 2. Baseline Model ────────────────────────────────────────
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)
print(f"\n[2] Baseline RF (n_estimators=100):")
print(f"    Accuracy : {accuracy_score(y_test, y_pred_base):.4f}")
print(f"    F1-Score : {f1_score(y_test, y_pred_base):.4f}")

# ── 3. Hyperparameter Tuning ─────────────────────────────────
print("\n[3] Hyperparameter Tuning (n_estimators × max_depth):")
n_trees_opts = [50, 100, 200]
max_depth_opts = [None, 5, 10, 15]
best_score, best_params = 0, {}

for n in n_trees_opts:
    for d in max_depth_opts:
        rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42, n_jobs=-1)
        score = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1').mean()
        print(f"    n_estimators={n:3d}, max_depth={str(d):4s} → CV F1 = {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = {'n_estimators': n, 'max_depth': d}

print(f"\n    Best params: {best_params} → CV F1 = {best_score:.4f}")

# ── 4. Final Model ───────────────────────────────────────────
rf_best = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
rf_best.fit(X_train, y_train)
y_pred  = rf_best.predict(X_test)
y_proba = rf_best.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"\n[4] Final Model Evaluation:")
print(f"    Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"    Precision : {precision_score(y_test, y_pred):.4f}")
print(f"    Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"    F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"    ROC-AUC   : {roc_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")

# Cross-validation final
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_best, X_train, y_train, cv=cv, scoring='accuracy')
print(f"[5] 5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 5. Feature Importance ────────────────────────────────────
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_best.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n[6] Top 10 Feature Importances:")
print(importance_df.head(10).to_string(index=False))

# ── 6. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Random Forest — Customer Churn\nIntern: Kenil Golakiya | Codveda Technology',
             fontsize=12, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
axes[0].imshow(cm, cmap='Blues')
axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
axes[0].set_xticklabels(['No Churn','Churn']); axes[0].set_yticklabels(['No Churn','Churn'])
for i in range(2):
    for j in range(2):
        axes[0].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16,fontweight='bold',
                     color='white' if cm[i,j]>cm.max()/2 else 'black')
axes[0].set_title('Confusion Matrix'); axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC={roc_auc:.3f}')
axes[1].plot([0,1],[0,1],'r--',lw=1); axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title('ROC Curve'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

top10 = importance_df.head(10)
axes[2].barh(top10['Feature'][::-1], top10['Importance'][::-1], color='steelblue', edgecolor='white')
axes[2].set_xlabel('Importance'); axes[2].set_title('Top 10 Feature Importances')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('random_forest_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[7] Plot saved → random_forest_results.png")
print("\n" + "=" * 60)
print("  TASK 7 COMPLETE ✓")
print("=" * 60)
