import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocessing import preprocess_text

# GLOBAL VARIABLES
inverted_index = defaultdict(lambda: defaultdict(int))
interest_index = defaultdict(set)
prof_to_citations = {}
prof_to_publications = {}
prof_to_interests = {}
prof_index_map = {}
corpus = []
tfidf_vectorizer = None
tfidf_matrix = None

def load_data():
    with open("profs_and_publications.json", "r", encoding="utf-8") as f:
        return json.load(f)

def build_indices(data):
    global tfidf_vectorizer, tfidf_matrix

    for prof in data:
        prof_key = (prof["name"], prof["id"])
        prof_to_citations[prof_key] = prof.get("citations", 0)
        prof_to_publications[prof_key] = prof.get("publications", [])
        prof_to_interests[prof_key] = prof.get("interests", [])

        for publication in prof["publications"]:
            corpus.append(publication)
            prof_index_map[len(corpus) - 1] = prof_key
            for term in preprocess_text(publication):
                inverted_index[term][prof_key] += 1

        for interest in prof.get("interests", []):
            for term in preprocess_text(interest):
                interest_index[term].add(prof_key)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2, norm='l2')
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)