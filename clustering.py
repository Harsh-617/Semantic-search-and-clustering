import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator

#Load Cleaned Document Data
df = pd.read_csv("data/cleaned/bbc_news_cleaned.csv")

#Load TF-IDF Matrix (sparse array)
tfidf_matrix = sp.load_npz("data/final/tfidf_matrix.npz")

#Load Dense Sentence Embeddings
embeddings = np.load("data/final/embeddings.npy")

#Load TF-IDF Feature Names (vocabulary list)
tfidf_features = np.load("data/final/tfidf_features.npy", allow_pickle=True)


# ==============================
# Find Optimal K using the Elbow Method (for both TF-IDF & Embeddings)
# ==============================

# Define K range
K_range = range(2, 16)  # test between 2 and 15 clusters
K_list = list(K_range)  # convert to list for Kneelocator

# Containers for both models
inertia_tfidf = []
inertia_embed = []

# ---------- TF-IDF ----------
print("\n Running Elbow Method on TF-IDF vectors...")
X_tfidf = tfidf_matrix

for k in K_range:
    print(f"  K = {k}")
    kmeans_tfidf = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_tfidf.fit(X_tfidf)
    inertia_tfidf.append(kmeans_tfidf.inertia_)

# ---------- Embeddings ----------
print("\n Running Elbow Method on Embedding vectors...")
X_embed = embeddings

for k in K_range:
    print(f"  K = {k}")
    kmeans_embed = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_embed.fit(X_embed)
    inertia_embed.append(kmeans_embed.inertia_)


# -----------Automatic K selection using kneelocator------------------
print("\n Automatically selecting optimal k ...")

#TF-IDF selection
kneedle_tfidf = KneeLocator(
    K_list,
    inertia_tfidf,
    curve='convex',
    direction='decreasing'
)
K_TFIDF = kneedle_tfidf.elbow
print(f'Optimal K for TF-IDF selected : {K_TFIDF}')

#Embeddings selection
kneedle_embed = KneeLocator(
    K_list,
    inertia_embed,
    curve='convex',
    direction='decreasing'
)
K_EMBED = kneedle_embed.elbow
print(f'Optimal K for embeddings selected : {K_EMBED}')



# ---------- Plot both & show automated selection points----------
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia_tfidf, marker='o', linestyle='--', label='TF-IDF Inertia')
plt.plot(K_range, inertia_embed, marker='s', linestyle='-', label='Embedding Inertia')

#add automated selection points to the plot
plt.plot(K_TFIDF, kneedle_tfidf.elbow_y, 'ro', markersize=10, label=f'TF_IDF Auto k={K_TFIDF}')
plt.plot(K_EMBED, kneedle_embed.elbow_y, 'g*', markersize=10, label=f'Embeddings Auto K={K_EMBED}')

plt.title('Elbow Method for Optimal K (TF-IDF vs Embeddings)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (WCSS)')
plt.legend()
plt.grid(True)
plt.show()

print("\n Elbow plots generated for both TF-IDF and Embeddings.")


# ==============================
# Run Final KMeans Models (TF-IDF & Embeddings)
# ==============================

# TF-IDF KMeans
kmeans_final_tfidf = KMeans(n_clusters=K_TFIDF, random_state=42, n_init=10)
kmeans_final_tfidf.fit(tfidf_matrix)
print("Final KMeans model fitted on TF-IDF matrix.")

# Embeddings KMeans
kmeans_final_embed = KMeans(n_clusters=K_EMBED, random_state=42, n_init=10)
kmeans_final_embed.fit(embeddings)
print("Final KMeans model fitted on Embeddings.")


# ==============================
# Assign Cluster Labels to DataFrame
# ==============================

print("\n Assigning cluster labels to the DataFrame...")

# Assign TF-IDF cluster labels
df["cluster_tfidf"] = kmeans_final_tfidf.labels_

# Assign Embedding cluster labels
df["cluster_embed"] = kmeans_final_embed.labels_

# Save the clustered DataFrame
output_path = "data/final/bbc_news_clustered.csv"
df.to_csv(output_path, index=False)

print(f" Cluster labels added and file saved successfully at: {output_path}")


# =====================================
# Cluster Interpretation and Naming
# =====================================

# --- TF-IDF CLUSTER INTERPRETATION ---
from sklearn.feature_extraction.text import TfidfVectorizer

print("\n Interpreting TF-IDF clusters...")

# Use preloaded TF-IDF feature names and final KMeans centroids
terms = tfidf_features
centroids = kmeans_final_tfidf.cluster_centers_


# For each cluster, print top keywords
n_terms = 10
for i in range(K_TFIDF):
    top_indices = centroids[i].argsort()[-n_terms:][::-1]
    top_terms = [terms[ind] for ind in top_indices]
    print(f"\n TF-IDF Cluster {i}:")
    print(", ".join(top_terms))

print("\n Review the printed keywords and assign descriptive names for each cluster.\n")


# --- EMBEDDING CLUSTER INTERPRETATION ---
print("\n Interpreting Embedding clusters...")

# Print a few representative document titles per embedding cluster
for i in range(K_EMBED):
    cluster_docs = df[df["cluster_embed"] == i]["title"]
    sample_docs = cluster_docs.sample(min(10, len(cluster_docs)), random_state=42)

    print(f"\n Embedding Cluster {i}:")
    for doc in sample_docs:
        print(f" - {doc}")

print("\n Review these titles and manually name each embedding cluster accordingly.\n")



# --- Final Cluster Name Mappings ---

cluster_names_tfidf = {
    0: "Royal Family & Monarchy",
    1: "Economy & Inflation",
    2: "Sports Events (General)",
    3: "Middle East Conflict / Gaza-Israel War",
    4: "Energy Prices & Cost of Living",
    5: "Football / Premier League",
    6: "Weekly News Digest",
    7: "UK Politics & Elections",
    8: "World Cup (Football)",
    9: "General UK News / Public Opinion",
    10: "Worker Strikes & Transport Disruptions",
    11: "Russia-Ukraine War"
}

cluster_names_embed = {
    0: "UK Politics & Government Affairs",
    1: "Sports - International Competitions",
    2: "Russia-Ukraine War",
    3: "Global Conflicts & Humanitarian Crises",
    4: "Entertainment, Culture & Science",
    5: "Business, Economy & Energy",
    6: "Sports & Events (Football + Entertainment)",
    7: "Crime & Public Safety"
}

# Add named clusters to DataFrame
df["tfidf_topic"] = df["cluster_tfidf"].map(cluster_names_tfidf)
df["embed_topic"] = df["cluster_embed"].map(cluster_names_embed)

# Save final version
output_path_named = "data/final/bbc_news_clustered_named.csv"
df.to_csv(output_path_named, index=False)
print(f" Named cluster file saved at: {output_path_named}")

