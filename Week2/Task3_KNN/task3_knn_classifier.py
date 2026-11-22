# ============================================================
# Codveda Technology - Machine Learning Internship
# Week 2 | Level 1 - Task 3: KNN Classifier
# Dataset : Iris Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

print("=" * 60)
print("  WEEK 2 | TASK 3 : K-NEAREST NEIGHBORS (KNN)")
print("=" * 60)

# ── 1. Load & Preprocess ─────────────────────────────────────
df = pd.read_csv("iris.csv")
le = LabelEncoder()
df['label'] = le.fit_transform(df['species'])

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[1] Data ready → Train: {len(X_train)} | Test: {len(X_test)}")
print(f"    Classes: {list(le.classes_)}")

# ── 2. Try Different K Values ────────────────────────────────
print("\n[2] Evaluating different K values:")
k_values = range(1, 21)
accuracies, precisions, recalls, f1s = [], [], [], []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recalls.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1s.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    print(f"    K={k:2d} → Accuracy: {accuracies[-1]:.4f} | Precision: {precisions[-1]:.4f} | Recall: {recalls[-1]:.4f} | F1: {f1s[-1]:.4f}")

best_k = list(k_values)[np.argmax(accuracies)]
print(f"\n    Best K = {best_k} with Accuracy = {max(accuracies):.4f}")

# ── 3. Best Model ────────────────────────────────────────────
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred_best = best_knn.predict(X_test)

print(f"\n[3] Best Model (K={best_k}) — Full Evaluation:")
print(f"    Accuracy  : {accuracy_score(y_test, y_pred_best):.4f}")
print(f"    Precision : {precision_score(y_test, y_pred_best, average='weighted'):.4f}")
print(f"    Recall    : {recall_score(y_test, y_pred_best, average='weighted'):.4f}")
print(f"    F1-Score  : {f1_score(y_test, y_pred_best, average='weighted'):.4f}")
print(f"\n    Classification Report:\n{classification_report(y_test, y_pred_best, target_names=le.classes_)}")

# ── 4. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('KNN Classifier — Iris Dataset\nIntern: Kenil Golakiya | Codveda Technology',
             fontsize=12, fontweight='bold')

# K vs Accuracy
axes[0].plot(list(k_values), accuracies, 'o-', color='steelblue', lw=2, markersize=6, label='Accuracy')
axes[0].plot(list(k_values), f1s, 's--', color='coral', lw=2, markersize=6, label='F1-Score')
axes[0].axvline(best_k, color='green', linestyle='--', lw=1.5, label=f'Best K={best_k}')
axes[0].set_xlabel('K Value'); axes[0].set_ylabel('Score')
axes[0].set_title('K Value vs Performance'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
axes[1].imshow(cm, cmap='Blues')
axes[1].set_xticks([0,1,2]); axes[1].set_yticks([0,1,2])
axes[1].set_xticklabels(le.classes_, rotation=15)
axes[1].set_yticklabels(le.classes_)
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else 'black')
axes[1].set_title(f'Confusion Matrix (K={best_k})')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

# Metrics comparison across K
axes[2].plot(list(k_values), accuracies, label='Accuracy', lw=2)
axes[2].plot(list(k_values), precisions, label='Precision', lw=2)
axes[2].plot(list(k_values), recalls,    label='Recall', lw=2)
axes[2].plot(list(k_values), f1s,        label='F1-Score', lw=2)
axes[2].set_xlabel('K Value'); axes[2].set_ylabel('Score')
axes[2].set_title('All Metrics vs K'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[4] Plot saved → knn_results.png")
print("\n" + "=" * 60)
print("  TASK 3 COMPLETE ✓")
print("=" * 60)
