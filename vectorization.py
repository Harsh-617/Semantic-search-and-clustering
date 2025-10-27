from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import pandas as pd
from scipy import sparse
import numpy as np
import joblib

df = pd.read_csv("data/cleaned/bbc_news_cleaned.csv")

#-----------------------------------------------------------------------------------------------------------------
"""Term Frequency-Inverse Document Frequency"""

# Initialize TF-IDF Vectorizer
#ngram_range=(1, 2) → captures single words and two-word phrases
#This determines what kind of features (words or phrases) the model learns
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # includes unigrams and bigrams


# Fit and transform the CleanedText column
#.fit_transform() → learns vocabulary and idf weights + transforms text in one go
#This is where text → numbers happens.
tfidf_matrix = vectorizer.fit_transform(df["CleanedText"])


# Save the fitted TfidfVectorizer object
# This creates the 'brain' file for the query retrieval script to load efficiently
joblib.dump(vectorizer, "data/final/tfidf_vectorizer_fitted.joblib")

# Save the TF-IDF matrix
sparse.save_npz("data/final/tfidf_matrix.npz", tfidf_matrix)


# Save feature names (vocabulary)
#get_feature_names_out() → gives the list of all tokens/phrases in the TF-IDF vocabulary
#np.save() → saves the vocabulary to a .npy file for reuse
feature_names = vectorizer.get_feature_names_out()
np.save("data/final/tfidf_features.npy", feature_names)

"""
tfidf_matrix.npz → the values (numbers)
tfidf_features.npy → the labels (column names)

here
.npz -> tells you the “importance” values, the actual vectore
.npy -> tells you which words those importance values belong to, labels
"""

#-----------------------------------------------------------------------------------------------------------------
"""Embeddings / sementic representation"""


#select a pretrained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode text to dense embeddings
embeddings = model.encode(df['CleanedText'].tolist(), show_progress_bar=True)

# Convert to NumPy array (already is, but for safety)
embeddings = np.array(embeddings)

# Save embeddings to file
np.save("data/final/embeddings.npy", embeddings)