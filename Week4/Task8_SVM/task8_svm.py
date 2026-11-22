# ============================================================
# Codveda Technology - Machine Learning Internship
# Week 4 | Level 3 - Task 2: SVM for Classification
# Dataset : Iris Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)

print("=" * 60)
print("  WEEK 4 | TASK 8 : SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)

# ── 1. Load & Preprocess ─────────────────────────────────────
df = pd.read_csv("iris.csv")
le = LabelEncoder()
df['label'] = le.fit_transform(df['species'])

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]; y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[1] Train: {len(X_train)} | Test: {len(X_test)}")

# ── 2. Compare Kernels ───────────────────────────────────────
print("\n[2] Comparing SVM Kernels:")
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
kernel_results = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42, probability=True)
    cv_score = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy').mean()
    svm.fit(X_train, y_train)
    y_pred_k = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred_k)
    f1  = f1_score(y_test, y_pred_k, average='weighted')
    kernel_results[kernel] = {'cv_acc': cv_score, 'test_acc': acc, 'f1': f1}
    print(f"    {kernel:8s} → CV Acc: {cv_score:.4f} | Test Acc: {acc:.4f} | F1: {f1:.4f}")

best_kernel = max(kernel_results, key=lambda k: kernel_results[k]['f1'])
print(f"\n    Best kernel: {best_kernel}")

# ── 3. Final Model ───────────────────────────────────────────
svm_best = SVC(kernel=best_kernel, probability=True, random_state=42)
svm_best.fit(X_train, y_train)
y_pred = svm_best.predict(X_test)

print(f"\n[3] Final SVM ({best_kernel} kernel) Evaluation:")
print(f"    Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"    Precision : {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"    Recall    : {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"    F1-Score  : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

# ── 4. Decision Boundary (2 features for 2D plot) ────────────
svm_2d_linear = SVC(kernel='linear', random_state=42)
svm_2d_rbf    = SVC(kernel='rbf',    random_state=42)
feat_2d = ['petal_length', 'petal_width']
X2 = scaler.fit_transform(df[feat_2d])
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.2, random_state=42, stratify=y)
svm_2d_linear.fit(X2_train, y2_train)
svm_2d_rbf.fit(X2_train, y2_train)

def plot_decision_boundary(model, X, y, ax, title, classes):
    h = 0.02
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='Set1')
    colors = ['red','green','blue']
    for i, cls in enumerate(np.unique(y)):
        mask = y == cls
        ax.scatter(X[mask,0], X[mask,1], c=colors[i], s=60,
                   label=classes[i], edgecolors='white', zorder=3)
    ax.set_xlabel(feat_2d[0]); ax.set_ylabel(feat_2d[1])
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# ── 5. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('SVM Classifier — Iris Dataset\nIntern: Kenil Golakiya | Codveda Technology',
             fontsize=12, fontweight='bold')

plot_decision_boundary(svm_2d_linear, X2, y.values, axes[0],
                       'Linear Kernel — Decision Boundary', le.classes_)
plot_decision_boundary(svm_2d_rbf, X2, y.values, axes[1],
                       'RBF Kernel — Decision Boundary', le.classes_)

cm = confusion_matrix(y_test, y_pred)
axes[2].imshow(cm, cmap='Purples')
axes[2].set_xticks([0,1,2]); axes[2].set_yticks([0,1,2])
axes[2].set_xticklabels(le.classes_, rotation=15); axes[2].set_yticklabels(le.classes_)
for i in range(3):
    for j in range(3):
        axes[2].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=14,fontweight='bold',
                     color='white' if cm[i,j]>cm.max()/2 else 'black')
axes[2].set_title(f'Confusion Matrix ({best_kernel} kernel)')
axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('svm_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[4] Plot saved → svm_results.png")
print("\n" + "=" * 60)
print("  TASK 8 COMPLETE ✓")
print("=" * 60)
