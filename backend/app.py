import json
import os
from flask import Flask, render_template, request
from flask_cors import CORS
from helpers.MySQLDatabaseHandler import MySQLDatabaseHandler
import pandas as pd

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

# Sample search using json with pandas
'''def json_search(query):
    matches = []
    merged_df = pd.merge(episodes_df, reviews_df, left_on='id', right_on='id', how='inner')
    matches = merged_df[merged_df['title'].str.lower().str.contains(query.lower())]
    matches_filtered = matches[['title', 'descr', 'imdb_rating']]
    matches_filtered_json = matches_filtered.to_json(orient='records')
    return matches_filtered_json
'''
def get_all_prof_names(data_json):
    # assumes no repeating names, use id instead?
    names = set(data_json['name'].values())
    return names

def or_search_publications(query):
    # boolean OR search to search within publications
    query_terms = set(query.lower().split())
    results = []
    for prof in data:
        publication_values = [word.lower() for word in prof['publications']]
        matched = [
            paper_title for paper_title in prof['publications'] if any(term in paper_title.lower() for term in query_terms)
        ]

        if matched:
            results.append({
                "name": prof["name"],
                "id" : prof["id"],
                "matched_publications": matched
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

def search():
    interests_query = request.args.get("interests", "")
    #citation_range = request.args.get("citations", "0")
    
    results = or_search_publications(interests_query)
    return results

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)