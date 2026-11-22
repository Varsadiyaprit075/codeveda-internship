# ============================================================
# Codveda Technology - Machine Learning Internship
# Week 1 | Level 1 - Task 2: Linear Regression
# Dataset : House Prediction Dataset (Boston Housing)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

print("=" * 60)
print("  WEEK 1 | TASK 2 : LINEAR REGRESSION")
print("=" * 60)

# ── 1. Load ──────────────────────────────────────────────────
columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
           'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df = pd.read_csv("house_prediction.csv", header=None, sep=r'\s+', names=columns)
print(f"\n[1] Dataset Loaded → Shape: {df.shape}")
print(df.head())

# ── 2. Missing Values ────────────────────────────────────────
print(f"\n[2] Missing Values: {df.isnull().sum().sum()} total")
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# ── 3. Features & Target ────────────────────────────────────
X = df.drop('MEDV', axis=1)
y = df['MEDV']
print(f"\n[3] Features: {list(X.columns)}")
print(f"    Target  : MEDV (House Price in $1000s)")
print(f"    Price range: ${y.min():.1f}k – ${y.max():.1f}k | Mean: ${y.mean():.1f}k")

# ── 4. Split & Scale ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"\n[4] Split → Train: {len(X_train)} | Test: {len(X_test)}")

# ── 5. Train ─────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_sc, y_train)
print("\n[5] Model Trained.")

# ── 6. Coefficients ──────────────────────────────────────────
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_}
                       ).sort_values('Coefficient', ascending=False)
print("\n[6] Feature Coefficients (sorted):")
print(coef_df.to_string(index=False))
print(f"\n    Intercept: {model.intercept_:.4f}")
print("    Key insights:")
print("    → RM (avg rooms) has strongest positive effect on price")
print("    → LSTAT (% lower status population) has strongest negative effect")

# ── 7. Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"\n[7] Evaluation Metrics:")
print(f"    R²   : {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"    MSE  : {mse:.4f}")
print(f"    RMSE : {rmse:.4f} ($1000s)")

# ── 8. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Linear Regression — House Price Prediction\nIntern: Kenil Golakiya | Codveda Technology',
             fontsize=12, fontweight='bold')

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='white', s=60)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Fit')
axes[0].set_xlabel('Actual Price ($1000s)'); axes[0].set_ylabel('Predicted Price ($1000s)')
axes[0].set_title(f'Actual vs Predicted\nR² = {r2:.3f}')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, color='coral', edgecolors='white', s=60)
axes[1].axhline(0, color='black', lw=1.5, linestyle='--')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot'); axes[1].grid(True, alpha=0.3)

# Coefficients
colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
axes[2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='white')
axes[2].axvline(0, color='black', lw=1)
axes[2].set_xlabel('Coefficient'); axes[2].set_title('Feature Coefficients')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('linear_regression_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[8] Plot saved → linear_regression_results.png")
print("\n" + "=" * 60)
print("  TASK 2 COMPLETE ✓")
print("=" * 60)
