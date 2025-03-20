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
prof_to_citations = {(prof_data.get("name"), prof_data.get("id", "")): prof_data.get('citations') for prof_data in data}  # { prof_key : citations }


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


def get_top_publications(prof_data, query_terms):
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
                "affiliation": prof_data.get("affiliation", "Cornell University"),
                "interests": prof_data.get("interests", []),
                "citations": prof_citations,
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
    citation_range = request.args.get("citations", "0")
    
    results = tf_ranked_search(query, citation_range)
    return jsonify(results)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)