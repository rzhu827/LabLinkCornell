import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocessing import custom_tokenizer, stemmed_stop_words

# GLOBAL VARIABLES
inverted_index = defaultdict(lambda: defaultdict(int))  # {term: {prof_key: term_freq}}
interest_index = defaultdict(set)                       # {interest_term: {prof_key1, prof_key2, ...}}
prof_to_citations = {}                                  # {prof_key: citations}
prof_to_publications = {}                               # {prof_key: [publications]}
prof_to_interests = {}                                  # {prof_key: [interests]}
prof_index_map = {}                                     # {doc_index: prof_key}
corpus = []                                             # All publications for TF-IDF
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
            for term in custom_tokenizer(publication):
                inverted_index[term][prof_key] += 1

        for interest in prof.get("interests", []):
            for term in custom_tokenizer(interest):
                interest_index[term].add(prof_key)
    
    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,     # Custom tokenization and stemming
        ngram_range=(1, 3),             # Create 1-gram, 2-gram, and 3-gram tokens
        stop_words=None,                # Remove common English stop words
        max_df=0.85,                    # Ignore terms that appear in more than 85% of the documents
        min_df=2,                       # Ignore terms that appear in fewer than 2 documents
        norm='l2'                       # Normalize term frequencies to unit length (L2 norm)
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)