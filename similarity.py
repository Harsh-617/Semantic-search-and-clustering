import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Load the TF-IDF sparse matrix
tfidf_matrix = sparse.load_npz("data/final/tfidf_matrix.npz")

# Load the sentence embeddings
embeddings = np.load("data/final/embeddings.npy")

# Load TF-IDF feature names (not needed - for reference / good practice)
tfidf_features = np.load("data/final/tfidf_features.npy", allow_pickle=True)


# --- TF-IDF Cosine Similarity ---
tfidf_cos_sim = cosine_similarity(tfidf_matrix)
np.save("data/final/tfidf_cosine_similarity.npy", tfidf_cos_sim)

# --- Embedding Cosine Similarity ---
embedding_cos_sim = cosine_similarity(embeddings)
np.save("data/final/embedding_cosine_similarity.npy", embedding_cos_sim)

# --- Embedding Euclidean Distance ---
embedding_euc_dist = euclidean_distances(embeddings)
np.save("data/final/embedding_euclidean_distance.npy", embedding_euc_dist)