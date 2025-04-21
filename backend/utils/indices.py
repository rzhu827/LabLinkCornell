import pickle
import lzma
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))

def load_data():
    with lzma.open("utils/precomputed_data.pkl", "rb") as f:
        data = pickle.load(f)
        return data

def get_query_terms(query, count_vectorizer_analyze, tfidf_vectorizer):
    """Get all n-grams (unigrams to trigrams) present in the query and the TF-IDF vocab."""
    start = time.time()
    ngram_tokens = count_vectorizer_analyze(query)
    print(f"analyze: {(time.time() - start):.4f}")
    return [term for term in ngram_tokens if term in tfidf_vectorizer.vocabulary_]