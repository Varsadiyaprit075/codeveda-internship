# ============================================================
# Codveda Technology - Machine Learning Internship
# Week 2 | Level 2 - Task 1: Logistic Regression
# Dataset : Customer Churn Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)

print("=" * 60)
print("  WEEK 2 | TASK 4 : LOGISTIC REGRESSION (CHURN)")
print("=" * 60)

# ── 1. Load ──────────────────────────────────────────────────
train_df = pd.read_csv("churn_train.csv")
test_df  = pd.read_csv("churn_test.csv")
print(f"\n[1] Train: {train_df.shape} | Test: {test_df.shape}")

# ── 2. Preprocess ────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    df.drop(columns=['State', 'Area code'], inplace=True, errors='ignore')
    df['International plan'] = (df['International plan'] == 'Yes').astype(int)
    df['Voice mail plan']    = (df['Voice mail plan']    == 'Yes').astype(int)
    df['Churn'] = df['Churn'].map({'True': 1, True: 1, 'False': 0, False: 0})
    return df

train_df = preprocess(train_df)
test_df  = preprocess(test_df)

X_train = train_df.drop('Churn', axis=1)
y_train = train_df['Churn']
X_test  = test_df.drop('Churn', axis=1)
y_test  = test_df['Churn']

print(f"\n[2] Preprocessing done.")
print(f"    Churn rate (train): {y_train.mean()*100:.1f}%")
print(f"    Churn rate (test) : {y_test.mean()*100:.1f}%")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 3. Train ─────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)
print("\n[3] Logistic Regression model trained.")

# ── 4. Coefficients & Odds Ratios ────────────────────────────
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0])
}).sort_values('Odds Ratio', ascending=False)

print("\n[4] Top 10 Features by Odds Ratio:")
print(coef_df.head(10).to_string(index=False))
print("    → Odds Ratio > 1 increases churn risk")
print("    → Odds Ratio < 1 decreases churn risk")

# ── 5. Evaluate ──────────────────────────────────────────────
y_pred  = model.predict(X_test_sc)
y_proba = model.predict_proba(X_test_sc)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

print(f"\n[5] Evaluation Metrics:")
print(f"    Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"    Precision : {precision_score(y_test, y_pred):.4f}")
print(f"    Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"    F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"    ROC-AUC   : {roc_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")

# ── 6. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Logistic Regression — Customer Churn\nIntern: Kenil Golakiya | Codveda Technology',
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

axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {roc_auc:.3f}')
axes[1].plot([0,1],[0,1],'r--',lw=1); axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].set_title('ROC Curve'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

top10 = coef_df.head(10)
colors = ['green' if o>1 else 'red' for o in top10['Odds Ratio']]
axes[2].barh(top10['Feature'], top10['Odds Ratio'], color=colors, edgecolor='white')
axes[2].axvline(1, color='black', lw=1); axes[2].set_xlabel('Odds Ratio')
axes[2].set_title('Top 10 Feature Odds Ratios'); axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('logistic_regression_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[6] Plot saved → logistic_regression_results.png")
print("\n" + "=" * 60)
print("  TASK 4 COMPLETE ✓")
print("=" * 60)
