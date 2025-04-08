import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from collections import defaultdict, Counter
import re
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ROOT_PATH for linking with all your files. 
# Feel free to use a config.py or settings.py with a global export variable
os.environ['ROOT_PATH'] = os.path.abspath(os.path.join("..",os.curdir))

# Get the directory of the current script
current_directory = os.path.dirname(os.path.abspath(__file__))

# Specify the path to the JSON file relative to the current script
with open("profs_and_publications.json", "r", encoding="utf-8") as f:
    data = json.load(f)

app = Flask(__name__)
CORS(app)


# GLOBAL VARIABLES
inverted_index = defaultdict(lambda: defaultdict(int))  # { term : { prof_key : term_freq } }
prof_to_citations = {(prof_data.get("name"), prof_data.get("id", "")): prof_data.get('citations') for prof_data in data}  # { prof_key : citations }
interest_index = defaultdict(set) #{interest_term: {prof_key1, prof_key2, ...}}
prof_to_publications = {}  # {prof_key: [publications]}
prof_to_interests = {}  # {prof_key: [interests]}
prof_index_map = {}  # {doc_index: prof_key}
corpus = [] # All publications for TF-IDF
tfidf_vectorizer = None
tfidf_matrix = None

def preprocess_text(text):
    """Clean and tokenize text."""
    return re.sub(r'[^\w\s]', '', text).lower().split()

def build_inverted_index():
    """
    Constructs an inverted index that maps unique words across all publication
    titles to dictionaries of professors, where each professor is associated with
    the frequency of that term (tf) in their publications.
    """
    global inverted_index
    for entry in data:
        prof_key = (entry["name"], entry["id"])  # (prof_name, prof_id)
        for publication in entry["publications"]:
            terms = preprocess_text(publication)
            for term in terms:
                inverted_index[term][prof_key] += 1

# precompute the inverted_index once at startup
#build_inverted_index()

def build_indices():
    """
    Build all inverted index for use:
    interest_index, prof_to_publications, prof_to_interests, prof_index_map
    
    Also builds tfidf vectorizer and matrix:
    tfidf_vectorizer, tfidf_matrix
    """
    global inverted_index, interest_index, prof_to_citations, prof_to_publications, \
    prof_to_interests, corpus, prof_index_map, tfidf_vectorizer, tfidf_matrix
    
    for prof in data:
        prof_key = (prof["name"], prof["id"])
        prof_to_citations[prof_key] = prof.get("citations", 0)
        prof_to_publications[prof_key] = prof.get("publications", [])
        prof_to_interests[prof_key] = prof.get("interests", [])
        
        # Build publication inverted index
        for publication in prof["publications"]:
            corpus.append(publication) # Add to corpus for TF-IDF
            prof_index_map[len(corpus) - 1] = prof_key
            
            # Build term frequency index
            terms = preprocess_text(publication)
            for term in terms:
                inverted_index[term][prof_key] += 1
        
        # Build interest index
        for interest in prof.get("interests", []):
            for term in preprocess_text(interest):
                interest_index[term].add(prof_key)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=None, stop_words='english', 
                                       max_df=0.85, min_df=2, norm='l2')    #test and change these parameters 
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    print(f"Built indices for {len(prof_to_citations)} professors and {len(corpus)} publications") #Debugging
    print(f"Inverted index contains {len(inverted_index)} terms")  #Debugging

# build all inverted indices once at startup
build_indices()

def process_citation_range(citation_range):
    """Process citation range string into numeric bounds."""
    if not citation_range or citation_range == "0" or citation_range == "N/A":
        return float('-inf'), float('inf')
    elif citation_range == "100000":
        return 100000, float('inf')
    else:
        parts = re.sub(r'[^\d\-]', '', citation_range).split("-")
        return int(parts[0]), int(parts[1])
        
def get_top_publications(prof_data, query_terms):
    """
    Given a professor's data, returns the top 3 most relevant publications
    based on term frequency for the query terms.
    """
    publication_scores = []  # [ (publication, aggregated_term_freq) ]

    for publication in prof_data["publications"]:
        pub_terms = preprocess_text(publication)
        pub_term_to_tf = Counter(pub_terms)

        pub_score = 0
        for term in query_terms:
            pub_score += pub_term_to_tf.get(term, 0)
        publication_scores.append((publication, pub_score))

    return sorted(publication_scores, key=lambda x: x[1], reverse=True)[:3]


def tf_ranked_search(query, citation_range):
    """
    Given a user's input query terms and citation range, returns a ranked set of
    top 5 matched professors and their top 3 relevant publications, evaluated
    through highest aggregate term frequency for the query terms and relevance
    of citation quantity.
    """
    query_terms = set((re.sub(r'[^\w\s]', '', query)).lower().split())
    print("Citation range:", citation_range)  # Debug: print the citation range
    if citation_range == "0":  # no preference
        citation_low = float('-inf')
        citation_high = float('inf')
    elif citation_range == "100000":  # 100,000+
        citation_low = 100000
        citation_high = float('inf')
    else:
        citation_range = citation_range.split("-")
        citation_low = int(re.sub(r'[^\w\s]', '', citation_range[0]))
        citation_high = int(re.sub(r'[^\w\s]', '', citation_range[1]))

    scores = defaultdict(int)  # { prof_key : score }

    for term in query_terms:
        if term in inverted_index:
            for prof_key, term_freq in inverted_index[term].items():
                prof_citations = prof_to_citations[prof_key]                

                if citation_low <= prof_citations <= citation_high:
                    scores[prof_key] += term_freq

    ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

    results = []
    for prof_key, score in ranked_scores:
        prof_name, prof_id = prof_key
        prof_citations = prof_to_citations[prof_key]
        prof_data = next((p for p in data if p["id"] == prof_id), None)

        print(f"Score for {prof_name}: {score}")  # Debugging statement

        if prof_data:
            top_publications = get_top_publications(prof_data, query_terms)
            results.append({
                "name": prof_name,
                "id": prof_data.get("id", ""),
                "affiliation": prof_data.get("affiliation", "Cornell University"),
                "interests": prof_data.get("interests", []),
                "citations": prof_citations,
                "publications": [pub[0] for pub in top_publications]
                })
            
    return results

def cosine_sim(query_vector, doc_vector):
    """
    Compute cosine similarity between a query vector and all documents in a matrix.
    """
    dot_product = np.dot(query_vector, doc_vector)
    query_norm = np.linalg.norm(query_vector)
    doc_norm = np.linalg.norm(doc_vector)
    return dot_product / (query_norm * doc_norm + .00001)

def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity"""
    if not set1 or not set2:
        return 0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

def score_by_publications(query_vector_array, prof_scores):
    """Score professors by publication similarity by TF-IDF."""
    for idx, doc_vector in enumerate(tfidf_matrix):
        doc_vector_array = doc_vector.toarray()[0]
        similarity = cosine_sim(query_vector_array, doc_vector_array)
        
        if similarity > 0:
            prof_key = prof_index_map[idx]
            prof_scores[prof_key]['publication_score'] += similarity

def score_by_interests(query_terms, prof_scores):
    """Score professors based on interest relevance using Jaccard similarity."""
    matching_profs = set()
    for term in query_terms:
        if term in interest_index:
            matching_profs.update(interest_index[term])
    
    for prof_key in matching_profs:
        prof_interests = set(" ".join(prof_to_interests[prof_key]).lower().split())
        similarity = calculate_jaccard_similarity(query_terms, prof_interests)
        prof_scores[prof_key]['interest_score'] = similarity

def score_by_citations(citation_low, citation_high, prof_scores):
    """
    Score professors based on citation counts within range. 
    Normalized to make sure higher citation counts are not unfairly weighed higher
    If citations not in range, still considered but penalized 
    """
    max_citations = max(prof_to_citations.values()) if prof_to_citations else 1
    
    for prof_key, citations in prof_to_citations.items():
        normalized_score = citations / max_citations
        if citation_low <= citations <= citation_high:
            prof_scores[prof_key]['citation_score'] = normalized_score
        else:
            prof_scores[prof_key]['citation_score'] = normalized_score * 0.5 #If not in range, less weight

def calculate_final_scores(prof_scores):
    """Calculate final weighted scores with different weights."""
    weights = {
        'publication_score': 0.5,  # 50% weight to publications
        'interest_score': 0.3,     # 30% weight to interests
        'citation_score': 0.2      # 20% weight to citations
    }
    for prof_key, scores in prof_scores.items():
        total_score = sum(scores[factor] * weights[factor] for factor in weights)
        prof_scores[prof_key]['total_score'] = total_score

def get_relevant_publications(prof_key, query_vector):
    """Get top 3 most relevant publications for a professor using TF-IDF similarity."""
    publications = prof_to_publications[prof_key]
    pub_scores = []
    for publication in publications:
        pub_vector = tfidf_vectorizer.transform([publication])
        similarity = cosine_sim(query_vector.toarray()[0], pub_vector.toarray()[0])
        pub_scores.append((publication, similarity))
    return [pub for pub, _ in sorted(pub_scores, key=lambda x: x[1], reverse=True)[:3]]

def prepare_results(ranked_profs, query_vector):
    """Prepare final results with professor details and relevant publications."""
    results = []
    
    for prof_key, scores in ranked_profs:
        prof_name, prof_id = prof_key
        prof_data = next((p for p in data if p["id"] == prof_id), None)
        
        if prof_data:
            # Get top 3 relevant publications
            relevant_pubs = get_relevant_publications(prof_key, query_vector)
            
            results.append({
                "name": prof_name,
                "id": prof_data.get("id", ""),
                "affiliation": prof_data.get("affiliation", "Cornell University"),
                "interests": prof_data.get("interests", []),
                "citations": prof_to_citations[prof_key],
                "publications": relevant_pubs
            })
    return results

def combined_search(query, citation_range=None):
    """
    Search for professors based on query terms with weighted factors, considering 
    professor interests, publications and citations. 
    Returns list of top 5 professors with their details and top 3 relevant publications.
    """
    query_terms = set(preprocess_text(query))
    if not query_terms:
        return []
    
    # Convert query to TF-IDF vector for publication matching
    query_vector = tfidf_vectorizer.transform([query])
    query_vector_array = query_vector.toarray()[0]
    
    prof_scores = defaultdict(lambda: {
        'publication_score': 0,  # TF-IDF
        'interest_score': 0,     # Jaccard similarity
        'citation_score': 0,     # Normalized citations
        'total_score': 0         # Weighted combination
    })
    
    citation_low, citation_high = process_citation_range(citation_range)
    
    score_by_publications(query_vector_array, prof_scores)
    score_by_interests(query_terms, prof_scores)
    score_by_citations(citation_low, citation_high, prof_scores)

    calculate_final_scores(prof_scores)

    print(prof_scores) #debugginggg 
    ranked_profs = sorted(
        [(prof_key, scores) for prof_key, scores in prof_scores.items()],
        key=lambda x: x[1]['total_score'],
        reverse=True)[:5]
    return prepare_results(ranked_profs, query_vector)


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route('/search')
def search():
    query = request.args.get("query", "")
    citation_range = request.args.get("citations", "0")
    
    results = tf_ranked_search(query, citation_range) #change the algorithm method here
    return jsonify(results)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)