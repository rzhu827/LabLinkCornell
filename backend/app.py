import json
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd
from collections import defaultdict, Counter
import re

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
            terms = (re.sub(r'[^\w\s]', '', publication)).lower().split()
            for term in terms:
                inverted_index[term][prof_key] += 1

# precompute the inverted_index once at startup
build_inverted_index()


def get_top_publications(prof_data, prof_key, query_terms, inverted_index):
    """
    Given a professor's data, returns the top 3 most relevant publications
    based on term frequency for the query terms.
    """
    publication_scores = []  # [ (publication, aggregated_term_freq) ]

    for publication in prof_data["publications"]:
        pub_terms = (re.sub(r'[^\w\s]', '', publication)).lower().split()
        pub_term_to_tf = Counter(pub_terms)

        pub_score = 0
        for term in query_terms:
            pub_score += pub_term_to_tf.get(term, 0)
        publication_scores.append((publication, pub_score))

    return sorted(publication_scores, key=lambda x: x[1], reverse=True)[:3]


def tf_ranked_search(query):
    """
    Given a user's input query terms, returns a ranked set of top 5 matched
    professors and their top 3 relevant publications, evaluated through highest
    aggregate term frequency for the query terms.
    """
    query_terms = set((re.sub(r'[^\w\s]', '', query)).lower().split())
    scores = defaultdict(int)  # { prof_key : aggregated_term_freq }

    for term in query_terms:
        if term in inverted_index:
            for prof_key, term_freq in inverted_index[term].items():
                scores[prof_key] += term_freq

    ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

    results = []
    for prof_key, score in ranked_scores:
        prof_name, prof_id = prof_key
        prof_data = next((p for p in data if p["id"] == prof_id), None)

        if prof_data:
            top_publications = get_top_publications(prof_data, prof_key, query_terms, inverted_index)
            print(f"Top publications for {prof_name}: {top_publications}")  # Debugging statement
            results.append({
                "name": prof_name,
                "affiliation": prof_data.get("affiliation", "Cornell University"),
                "interests": prof_data.get("interests", []),
                "citations": prof_data.get("citations", 0),
                "publications": [pub[0] for pub in top_publications]
                })
            
    return results


@app.route("/")
def home():
    return render_template('base.html',title="sample html")

'''
@app.route("/episodes")
def episodes_search():
    text = request.args.get("title")
    return json_search(text)
'''
@app.route('/search')
def search():
    query = request.args.get("query", "")
    # citation_range = request.args.get("citations", "0")
    
    results = tf_ranked_search(query)
    return jsonify(results)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)