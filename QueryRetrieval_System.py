import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sentence_transformers import SentenceTransformer

from preprocessing import clean_document, lemmatizer, stop_words, get_wordnet_pos


# load Document Data
df = pd.read_csv("data/cleaned/bbc_news_cleaned.csv")  # full dataset for titles & descriptions

# load Vectors
# Dense embeddings
embeddings = np.load("data/final/embeddings.npy")

# Sparse TF-IDF matrix
tfidf_matrix = sp.load_npz("data/final/tfidf_matrix.npz")

# Query Transformation Models
# SentenceTransformer model for encoding queries
model = SentenceTransformer('all-MiniLM-L6-v2')

#Load the Fitted TfidfVectorizer (The Efficient Way!)
TFIDF_VECTORIZER = joblib.load("data/final/tfidf_vectorizer_fitted.joblib")


def retrieve_documents(query_string, N=3):
    #preprocess the query
    cleaned_query = clean_document(query_string)

    #Vectorize the Query
    tfidf_query_vec = TFIDF_VECTORIZER.transform([cleaned_query]) # TF-IDF vector
    embedding_query_vec = model.encode([cleaned_query]) # Embedding vector

    #Compute Similarity Scores
    tfidf_cosine_scores = cosine_similarity(tfidf_query_vec, tfidf_matrix).flatten()
    embedding_cosine_scores = cosine_similarity(embedding_query_vec, embeddings).flatten()
    embedding_euclidean_scores = euclidean_distances(embedding_query_vec, embeddings).flatten()

    #Rank and Select Top N
    top_tfidf_idx = tfidf_cosine_scores.argsort()[::-1][:N]           # descending - high score implies high similarity
    top_embed_cosine_idx = embedding_cosine_scores.argsort()[::-1][:N] # descending - similar
    top_embed_euclid_idx = embedding_euclidean_scores.argsort()[:N]  # ascending - low distance implies high similarity

    #Display the Results
    print(f"\nQuery: {query_string}\n")

    print("Top results based on TF-IDF Cosine Similarity:")
    for i in top_tfidf_idx:
        print(f"- Title: {df.loc[i, 'title']}")
        print(f"  Description: {df.loc[i, 'description']}")
        print(f"  Score: {tfidf_cosine_scores[i]:.4f}\n")

    print("Top results based on Embedding Cosine Similarity:")
    for i in top_embed_cosine_idx:
        print(f"- Title: {df.loc[i, 'title']}")
        print(f"  Description: {df.loc[i, 'description']}")
        print(f"  Score: {embedding_cosine_scores[i]:.4f}\n")

    print("Top results based on Embedding Euclidean Distance:")
    for i in top_embed_euclid_idx:
        print(f"- Title: {df.loc[i, 'title']}")
        print(f"  Description: {df.loc[i, 'description']}")
        print(f"  Distance: {embedding_euclidean_scores[i]:.4f}\n")

if __name__ == "__main__":
    # Prompt user for a query
    query = input("Enter your search query: ")

    retrieve_documents(query, N=3)