# ============================================================
# Codveda Technology - Machine Learning Internship
# Week 4 | Level 3 - Task 3: Neural Network (MLP)
# Dataset : Iris Dataset
# Note    : Uses scikit-learn MLPClassifier (no TF needed)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)

print("=" * 60)
print("  WEEK 4 | TASK 9 : NEURAL NETWORK (MLP)")
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
print(f"    Input features  : {len(features)}")
print(f"    Output classes  : {len(le.classes_)} ({list(le.classes_)})")

# ── 2. Architecture ──────────────────────────────────────────
print("\n[2] Neural Network Architecture:")
print("    Input Layer   : 4 neurons (sepal/petal features)")
print("    Hidden Layer 1: 16 neurons (ReLU)")
print("    Hidden Layer 2: 8 neurons  (ReLU)")
print("    Output Layer  : 3 neurons  (Softmax → 3 classes)")

# ── 3. Train ─────────────────────────────────────────────────
mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42,
    learning_rate_init=0.01,
    early_stopping=False,
    verbose=False
)
mlp.fit(X_train, y_train)
print(f"\n[3] Training complete in {mlp.n_iter_} iterations.")
print(f"    Final training loss: {mlp.loss_:.4f}")

# ── 4. Evaluate ──────────────────────────────────────────────
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')

print(f"\n[4] Test Set Evaluation:")
print(f"    Accuracy : {acc:.4f}")
print(f"    F1-Score : {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

cv_scores = cross_val_score(mlp, X_scaled, y, cv=5, scoring='accuracy')
print(f"    5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 5. Compare Architectures ─────────────────────────────────
print("\n[5] Comparing Different Architectures:")
architectures = {
    '(8,)':        (8,),
    '(16,8)':      (16,8),
    '(32,16,8)':   (32,16,8),
    '(64,32)':     (64,32),
}
arch_results = {}
for name, arch in architectures.items():
    m = MLPClassifier(hidden_layer_sizes=arch, activation='relu', solver='adam',
                      max_iter=2000, random_state=42, learning_rate_init=0.01)
    score = cross_val_score(m, X_scaled, y, cv=5, scoring='accuracy').mean()
    arch_results[name] = score
    print(f"    Architecture {name:12s} → CV Accuracy = {score:.4f}")

# ── 6. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Neural Network (MLP) — Iris Dataset\nIntern: Kenil Golakiya | Codveda Technology',
             fontsize=12, fontweight='bold')

# Training Loss Curve
axes[0].plot(mlp.loss_curve_, color='steelblue', lw=2, label='Training Loss')

axes[0].set_xlabel('Iterations'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss Curve'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
axes[1].imshow(cm, cmap='Oranges')
axes[1].set_xticks([0,1,2]); axes[1].set_yticks([0,1,2])
axes[1].set_xticklabels(le.classes_, rotation=15); axes[1].set_yticklabels(le.classes_)
for i in range(3):
    for j in range(3):
        axes[1].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=14,fontweight='bold',
                     color='white' if cm[i,j]>cm.max()/2 else 'black')
axes[1].set_title('Confusion Matrix'); axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

# Architecture Comparison
axes[2].bar(list(arch_results.keys()), list(arch_results.values()),
            color='steelblue', edgecolor='white')
axes[2].set_xlabel('Architecture'); axes[2].set_ylabel('CV Accuracy')
axes[2].set_title('Architecture Comparison'); axes[2].grid(True, alpha=0.3, axis='y')
axes[2].set_ylim(0.8, 1.01)
for i, (name, val) in enumerate(arch_results.items()):
    axes[2].text(i, val + 0.002, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('neural_network_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[6] Plot saved → neural_network_results.png")
print("\n" + "=" * 60)
print("  TASK 9 COMPLETE ✓")
print("=" * 60)
