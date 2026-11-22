# ============================================================
# Codveda Technology - Machine Learning Internship
# Week 1 | Level 1 - Task 1: Data Preprocessing

# Dataset : Iris Dataset
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

print("=" * 60)
print("  WEEK 1 | TASK 1 : DATA PREPROCESSING FOR ML")
print("=" * 60)

# ── 1. Load ──────────────────────────────────────────────────
df = pd.read_csv("iris.csv")
print(f"\n[1] Dataset Loaded → Shape: {df.shape}")
print(df.head())

# ── 2. EDA ───────────────────────────────────────────────────
print("\n[2] Data Types & Info:")
print(df.dtypes)
print("\n    Descriptive Statistics:")
print(df.describe())

# ── 3. Missing Values ────────────────────────────────────────
print("\n[3] Missing Values:")
print(df.isnull().sum())
for col in df.select_dtypes(include=np.number).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"    → '{col}' filled with mean")
print("    Missing value treatment complete.")

# ── 4. Encode Categorical ────────────────────────────────────
print("\n[4] Encoding 'species' with LabelEncoder:")
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])
print(f"    Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ── 5. Standardize ───────────────────────────────────────────
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[features].copy()
y = df['species_encoded']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n[5] Features after StandardScaler:")
print(f"    Mean ≈ {X_scaled.mean(axis=0).round(4)}")
print(f"    Std  ≈ {X_scaled.std(axis=0).round(4)}")

# ── 6. Train-Test Split ──────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[6] Train-Test Split (80/20):")
print(f"    Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ── 7. Save Outputs ──────────────────────────────────────────
train_out = pd.DataFrame(X_train, columns=features)
train_out['label'] = y_train.values
test_out  = pd.DataFrame(X_test,  columns=features)
test_out['label']  = y_test.values
train_out.to_csv("preprocessed_train.csv", index=False)
test_out.to_csv("preprocessed_test.csv",   index=False)

print("\n[7] Output files saved:")
print("    → preprocessed_train.csv")
print("    → preprocessed_test.csv")
print("\n" + "=" * 60)
print("  TASK 1 COMPLETE ✓")
print("=" * 60)
