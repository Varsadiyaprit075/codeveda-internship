# ============================================================
# Codveda Technology - Machine Learning Internship
# Week 3 | Level 2 - Task 2: Decision Tree Classification
# Dataset : Iris Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

print("=" * 60)
print("  WEEK 3 | TASK 5 : DECISION TREE CLASSIFIER")
print("=" * 60)

# ── 1. Load & Preprocess ─────────────────────────────────────
df = pd.read_csv("iris.csv")
le = LabelEncoder()
df['label'] = le.fit_transform(df['species'])

features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features]; y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[1] Data ready → Train: {len(X_train)} | Test: {len(X_test)}")

# ── 2. Unpruned Tree ─────────────────────────────────────────
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)
y_pred_full = dt_full.predict(X_test)
print(f"\n[2] Unpruned Tree:")
print(f"    Depth    : {dt_full.get_depth()}")
print(f"    Leaves   : {dt_full.get_n_leaves()}")
print(f"    Accuracy : {accuracy_score(y_test, y_pred_full):.4f}")
print(f"    F1-Score : {f1_score(y_test, y_pred_full, average='weighted'):.4f}")

# ── 3. Prune: Find Best Depth ────────────────────────────────
print("\n[3] Finding best max_depth via 5-fold CV:")
depths = range(1, 11)
cv_scores = []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    score = cross_val_score(dt, X, y, cv=5, scoring='accuracy').mean()
    cv_scores.append(score)
    print(f"    max_depth={d:2d} → CV Accuracy = {score:.4f}")

best_depth = list(depths)[np.argmax(cv_scores)]
print(f"\n    Best max_depth = {best_depth} (CV Acc = {max(cv_scores):.4f})")

# ── 4. Pruned Tree ───────────────────────────────────────────
dt_pruned = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred = dt_pruned.predict(X_test)

print(f"\n[4] Pruned Tree (max_depth={best_depth}):")
print(f"    Depth    : {dt_pruned.get_depth()}")
print(f"    Leaves   : {dt_pruned.get_n_leaves()}")
print(f"    Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"    F1-Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("\n[5] Tree Structure:")
print(export_text(dt_pruned, feature_names=features))

# ── 5. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Decision Tree Classifier — Iris Dataset\nIntern: Kenil Golakiya | Codveda Technology',
             fontsize=12, fontweight='bold')

plot_tree(dt_pruned, feature_names=features, class_names=le.classes_,
          filled=True, rounded=True, fontsize=9, ax=axes[0])
axes[0].set_title(f'Decision Tree (max_depth={best_depth})')

axes[1].plot(list(depths), cv_scores, 'o-', color='steelblue', lw=2, markersize=7)
axes[1].axvline(best_depth, color='red', linestyle='--', label=f'Best={best_depth}')
axes[1].set_xlabel('Max Depth'); axes[1].set_ylabel('CV Accuracy')
axes[1].set_title('Pruning: Depth vs Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
axes[2].imshow(cm, cmap='Greens')
axes[2].set_xticks([0,1,2]); axes[2].set_yticks([0,1,2])
axes[2].set_xticklabels(le.classes_, rotation=15); axes[2].set_yticklabels(le.classes_)
for i in range(3):
    for j in range(3):
        axes[2].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=14,fontweight='bold',
                     color='white' if cm[i,j]>cm.max()/2 else 'black')
axes[2].set_title('Confusion Matrix'); axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('decision_tree_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[6] Plot saved → decision_tree_results.png")
print("\n" + "=" * 60)
print("  TASK 5 COMPLETE ✓")
print("=" * 60)
