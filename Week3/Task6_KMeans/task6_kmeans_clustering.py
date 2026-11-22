# ============================================================
# Codveda Technology - Machine Learning Internship
# Week 3 | Level 2 - Task 3: K-Means Clustering
# Dataset : Stock Prices Dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=" * 60)
print("  WEEK 3 | TASK 6 : K-MEANS CLUSTERING")
print("=" * 60)

# ── 1. Load & Preprocess ─────────────────────────────────────
df = pd.read_csv("stock_prices.csv")
print(f"\n[1] Dataset Loaded → Shape: {df.shape}")
print(df.head())
print(f"\n    Columns: {list(df.columns)}")

# Group by symbol: create feature per stock
df['date'] = pd.to_datetime(df['date'])
features_df = df.groupby('symbol').agg(
    avg_close=('close', 'mean'),
    std_close=('close', 'std'),
    avg_volume=('volume', 'mean'),
    price_range=('high', lambda x: x.mean() - df.loc[x.index, 'low'].mean()),
    total_return=('close', lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)
).dropna().reset_index()

print(f"\n[2] Feature Engineering Complete:")
print(f"    Stocks after aggregation: {len(features_df)}")
print(features_df.describe())

# ── 2. Scale ─────────────────────────────────────────────────
X = features_df.drop('symbol', axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\n[3] Features scaled with StandardScaler.")

# ── 3. Elbow Method ──────────────────────────────────────────
print("\n[4] Elbow Method — finding optimal K:")
inertias = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"    K={k} → Inertia = {km.inertia_:.2f}")

# Simple elbow: pick k where drop slows (2nd derivative)
diffs = np.diff(inertias)
diffs2 = np.diff(diffs)
best_k = list(k_range)[np.argmax(diffs2) + 1]
print(f"\n    Optimal K = {best_k} (elbow point)")

# ── 4. Fit Best Model ────────────────────────────────────────
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
features_df['Cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n[5] K-Means fitted with K={best_k}")
print("\n    Cluster distribution:")
print(features_df['Cluster'].value_counts().sort_index())

print("\n    Cluster Centers (scaled):")
centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=X.columns)
print(centers_df.round(3).to_string())

print("\n    Cluster Interpretation:")
for c in range(best_k):
    grp = features_df[features_df['Cluster'] == c]
    print(f"    Cluster {c}: {len(grp)} stocks | Avg Price: ${grp['avg_close'].mean():.2f} | "
          f"Avg Return: {grp['total_return'].mean():.1f}% | Volatility: {grp['std_close'].mean():.2f}")

# ── 5. PCA for Visualization ─────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
features_df['PC1'] = X_pca[:, 0]
features_df['PC2'] = X_pca[:, 1]
print(f"\n[6] PCA: {pca.explained_variance_ratio_.sum()*100:.1f}% variance explained by 2 components")

# ── 6. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('K-Means Clustering — Stock Prices\nIntern: Kenil Golakiya | Codveda Technology',
             fontsize=12, fontweight='bold')

# Elbow
axes[0].plot(list(k_range), inertias, 'o-', color='steelblue', lw=2, markersize=7)
axes[0].axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
axes[0].set_xlabel('Number of Clusters (K)'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# PCA scatter
palette = sns.color_palette('tab10', best_k)
for c in range(best_k):
    mask = features_df['Cluster'] == c
    axes[1].scatter(features_df.loc[mask,'PC1'], features_df.loc[mask,'PC2'],
                    c=[palette[c]], s=80, label=f'Cluster {c}', alpha=0.8, edgecolors='white')
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title(f'Clusters (PCA 2D) — K={best_k}')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Avg Return by Cluster
cluster_stats = features_df.groupby('Cluster')[['avg_close','std_close','total_return']].mean()
cluster_stats['total_return'].plot(kind='bar', ax=axes[2], color=[palette[i] for i in range(best_k)],
                                    edgecolor='white')
axes[2].set_xlabel('Cluster'); axes[2].set_ylabel('Avg Total Return (%)')
axes[2].set_title('Average Return per Cluster'); axes[2].grid(True, alpha=0.3, axis='y')
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('kmeans_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[7] Plot saved → kmeans_results.png")
print("\n" + "=" * 60)
print("  TASK 6 COMPLETE ✓")
print("=" * 60)
