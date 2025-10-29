import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


# =====================1. Load Data and Vector Spaces===================================

# Load the final clustered and named DataFrame (from Step 7)
df = pd.read_csv("data/final/bbc_news_clustered_named.csv")

# Load the original vector spaces
tfidf_matrix = sp.load_npz("data/final/tfidf_matrix.npz")
embeddings = np.load("data/final/embeddings.npy")


# =========================2. Quantitative Evaluation: Silhouette Score=============================

print("\n--- Quantitative Evaluation (Silhouette Score) ---")

# --- A. TF-IDF Silhouette Score ---
# Inputs: TF-IDF matrix and TF-IDF cluster labels
try:
    tfidf_score = silhouette_score(tfidf_matrix, df['cluster_tfidf'])
    print(f"1. TF-IDF Clustering Score: {tfidf_score:.4f}")
except Exception as e:
    print(f"Error calculating TF-IDF Silhouette Score: {e}")


# --- B. Embeddings Silhouette Score ---
# Inputs: Embeddings array and Embeddings cluster labels
try:
    embedding_score = silhouette_score(embeddings, df['cluster_embed'])
    print(f"2. Embeddings Clustering Score: {embedding_score:.4f}")
except Exception as e:
    print(f"Error calculating Embeddings Silhouette Score: {e}")

print("\nComparison: The higher score indicates better-defined clusters.")



# =======================3. Visualization using PCA / TruncatedSVD ==========================
# PCA caused a memory error when converting the large sparse TF-IDF matrix to dense, so we use TruncatedSVD instead.

print("\n--- Generating PCA/SVD Visualizations ---")

# Setup the figure for two side-by-side plots (subplots)
plt.figure(figsize=(14, 6))

# --- A. TF-IDF Visualization (Using TruncatedSVD for Sparse Data) ---
# 1. Reduce dimensionality from thousands of features to 2 components
#    *** This works directly on the sparse matrix, avoiding the MemoryError ***
svd = TruncatedSVD(n_components=2, random_state=42)
tfidf_2d = svd.fit_transform(tfidf_matrix)

# 2. Create Plot for TF-IDF
plt.subplot(1, 2, 1) # 1 row, 2 columns, plot 1
plt.scatter(tfidf_2d[:, 0], tfidf_2d[:, 1], c=df['cluster_tfidf'], 
            cmap='viridis', s=8, alpha=0.7)
plt.title(f'TF-IDF Clusters (TruncatedSVD to 2D, K={df["cluster_tfidf"].nunique()})')
# Note: TruncatedSVD uses 'explained_variance_ratio_' just like PCA
plt.xlabel(f'SVD Component 1 ({svd.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'SVD Component 2 ({svd.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(label='TF-IDF Cluster ID')


# --- B. Embeddings Visualization ---
# 1. Reduce dimensionality from its original size (e.g., 512) to 2 components
pca_embed = PCA(n_components=2, random_state=42)
embeddings_2d = pca_embed.fit_transform(embeddings)

# 2. Create Plot for Embeddings
plt.subplot(1, 2, 2) # 1 row, 2 columns, plot 2
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=df['cluster_embed'], 
            cmap='plasma', s=8, alpha=0.7)
plt.title(f'Embeddings Clusters (PCA to 2D, K={df["cluster_embed"].nunique()})')
plt.xlabel(f'PCA Component 1 ({pca_embed.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PCA Component 2 ({pca_embed.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(label='Embedding Cluster ID')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

print("\nVisual comparison plots generated.")