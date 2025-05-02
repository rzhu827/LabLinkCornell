from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, re
from collections import defaultdict
from utils import indices
from utils.indices import get_query_terms
from utils.similarity import calculate_jaccard_similarity
import csv
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("words")
nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.corpus import wordnet
from wordfreq import zipf_frequency
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

l = WordNetLemmatizer()
l.lemmatize("dogs") # warmup the lemmatizer

data = indices.load_data() # the ones commented out are unused
dataset = data["dataset"]
inverted_index = data["inverted_index"]
interest_index = data["interest_index"]
prof_to_citations = data["prof_to_citations"]
# prof_to_publications = data["prof_to_publications"]
prof_to_interests = data["prof_to_interests"]
prof_index_map = data["prof_index_map"]
profid_to_name = data["profid_to_name"]
prof_keys_list = [(name, id) for id, name in profid_to_name.items()]
# cleaned_to_original = data["cleaned_to_original"]
publications_to_idx = data["publications_to_idx"]
corpus = data["corpus"]
svd = data["svd"]
tfidf_vectorizer = data["tfidf_vectorizer"]
count_vectorizer_analyze = data["count_vectorizer_analyze"]
tfidf_matrix = data["tfidf_matrix"]
lsi_matrix = data["lsi_matrix"]
theme_axes = data["theme_axes"]
profs_to_idx = data["profs_to_idx"]
prof_sim_matrix = data["prof_sim_matrix"]
coauthor_score_map, max_coauthor_score = data["coauthor_score_tup"]
citation_score_map, max_citation_score = data["citation_score_tup"]
prof_to_themes = data["prof_to_themes"]
prof_themes_matrix = data["prof_themes_matrix"]

MAX_DOC_SCORE_CONTRIBUTION = 0.4

VALID_ENGLISH = {
    w for w in inverted_index
    if w.isalpha() and wordnet.synsets(w)
}

abbreviations = {
    "3D": "Three-Dimensional",
    "A/D": "Analog-to-Digital",
    "ACID": "Atomicity, Consistency, Isolation, Durability",
    "AI": "Artificial Intelligence",
    "AR": "Augmented Reality",
    "AS": "Autonomous System",
    "ASR": "Automatic Speech Recognition",
    "ATM": "Asynchronous Transfer Mode",
    "AUC": "Area Under the Curve",
    "AVL": "Adelson-Velsky and Landis",
    "AWGN": "Additive White Gaussian Noise",
    "BASE": "Basically Available, Soft-state, Eventually consistent",
    "BDD": "Binary Decision Diagram",
    "BERT": "Bidirectional Encoder Representations from Transformers",
    "BFT": "Byzantine Fault Tolerance",
    "BGp": "Border Gateway Protocol", # Note: Standard is BGP, using provided key
    "BRDF": "Bidirectional Reflectance Distribution Function",
    "CAM": "Computer-Aided Manufacturing",
    "CART": "Classification and Regression Tree",
    "CCD": "Charge-Coupled Device",
    "CISC": "Complex Instruction Set Computer",
    "CMOS": "Complementary Metal-Oxide-Semiconductor",
    "CNN": "Convolutional Neural Network",
    "CP": "Constraint Programming",
    "CPS": "Cyber-Physical System",
    "CPU": "Central Processing Unit",
    "CRC": "Cyclic Redundancy Check",
    "CS": "Computer Science",
    "CSCW": "Computer-Supported Cooperative Work", # Field Name
    "DAG": "Directed Acyclic Graph",
    "DC": "Direct Current",
    "DDS": "Data Distribution Service",
    "DHT": "Distributed Hash Table",
    "DNF": "Disjunctive Normal Form",
    "DoF": "Degrees of Freedom",
    "DoS": "Denial of Service",
    "DP": "Dynamic Programming",
    "DRAM": "Dynamic Random-Access Memory",
    "DRM": "Digital Rights Management",
    "DST": "Digital Signal Processing",
    "DTN": "Delay-Tolerant Networking",
    "DVD": "Digital Versatile Disc",
    "EC": "Elliptic Curve",
    "EDA": "Exploratory Data Analysis",
    "EM": "Expectation-Maximization",
    "ER": "Entity Resolution",
    "FA": "Finite Automata",
    "FFT": "Fast Fourier Transform",
    "FP": "Functional Programming",
    "FPGA": "Field-Programmable Gate Array",
    "FT": "Fault Tolerant",
    "GAN": "Generative Adversarial Network",
    "GB": "GigaByte",
    "GCD": "Greatest Common Divisor",
    "GIF": "Graphics Interchange Format",
    "GLM": "Generalized Linear Model",
    "GMM": "Gaussian Mixture Model",
    "GNSS": "Global Navigation Satellite System",
    "GP": "Gaussian Process",
    "GPU": "Graphics Processing Unit",
    "GPGPU": "General-Purpose computing on Graphics Processing Units",
    "GPT": "Generative Pre-trained Transformer",
    "GPS": "Global Positioning System",
    "GUI": "Graphical User Interface",
    "HBM": "High Bandwidth Memory",
    "HCI": "Human-Computer Interaction", # Field Name
    "HD": "High Definition",
    "HMM": "Hidden Markov Model",
    "HMD": "Head-Mounted Display",
    "HPC": "High Performance Computing",
    "HRI": "Human Robot Interaction", # Field Name
    "HSV": "Hue, Saturation, Value",
    "HT": "Hash Table",
    "Hz": "Hertz",
    "ILP": "Integer Linear Programming",
    "IM": "Instant Messaging",
    "IoT": "Internet of Things",
    "IP": "Internet Protocol",
    "IPU": "Intelligence Processing Unit",
    "IRC": "Internet Relay Chat",
    "ISA": "Instruction Set Architecture",
    "IT": "Information Technology", # Broad Field
    "IVR": "Interactive Voice Response",
    "JIT": "Just-in-Time",
    "JPEG": "Joint Photographic Experts Group",
    "JSQ": "Join-the-Shortest-Queue",
    "JVM": "Java Virtual Machine",
    "k-SAT": "k-Satisfiability",
    "KMP": "Knuth-Morris-Pratt",
    "KR": "Knowledge Representation",
    "KV": "Key-Value",
    "L1": "First Language", # As defined in source
    "L2": "Second Language", # As defined in source
    "LAPACK": "Linear Algebra PACKage",
    "LASSO": "Least Absolute Shrinkage and Selection Operator",
    "LATEX": "A document preparation system", # Common Tool
    "LCD": "Liquid Crystal Display",
    "LDA": "Latent Dirichlet Allocation", # Or Linear Discriminant Analysis - using first definition
    "LIDAR": "Light Detection and Ranging",
    "LISP": "LISt Processing",
    "LLM": "Large Language Model",
    "LP": "Linear Programming",
    "LSI": "Latent Semantic Indexing",
    "LSM": "Log-Structured Merge-tree",
    "LTL": "Linear Temporal Logic",
    "LUT": "Lookup Table",
    "MAC": "Medium Access Control", # First definition listed
    # "MAC": "Multiply-Accumulate", # Second definition also present
    "MAP": "Maximum a Posteriori",
    "MDP": "Markov Decision Process",
    "MEMS": "Microelectromechanical Systems",
    "MEV": "Miner Extractable Value",
    "MIP": "Mixed Integer Programming",
    "ML": "Machine Learning",
    "MPI": "Message Passing Interface",
    "MRF": "Markov Random Field",
    "MRI": "Magnetic Resonance Imaging", # Common in CS imaging context
    "MST": "Minimum Spanning Tree",
    "NAS": "Neural Architecture Search",
    "NeRF": "Neural Radiance Field",
    "NetKAT": "Network Kleene Algebra with Tests",
    "NFV": "Network Function Virtualization",
    "NFT": "Non-Fungible Token", # Common in CS crypto context
    "NLP": "Natural Language Processing", # Field Name
    "NoC": "Network-on-Chip",
    "NP": "Nondeterministic Polynomial time",
    "NUMA": "Non-Uniform Memory Access",
    "OLAP": "Online Analytical Processing",
    "P2P": "Peer-to-Peer",
    "PAC": "Probably Approximately Correct",
    "PaaS": "Platform as a Service",
    "Paxos": "A consensus algorithm",
    "PCA": "Principal Component Analysis",
    "PCP": "Probabilistically Checkable Proof",
    "PDL": "Propositional Dynamic Logic",
    "PH": "Polynomial Hierarchy",
    "PHY": "Physical Layer",
    "POR": "Proof of Retrievability",
    "PPL": "Probabilistic Programming Language",
    "PRF": "Pseudorandom Function",
    "PSPACE": "Polynomial Space",
    "Q&A": "Question and Answer",
    "QoS": "Quality of Service",
    "RAG": "Retrieval-Augmented Generation",
    "RAM": "Random Access Memory",
    "RDBMS": "Relational Database Management System",
    "ReLU": "Rectified Linear Unit",
    "RF": "Radio Frequency",
    "RFID": "Radio-Frequency Identification",
    "RISC": "Reduced Instruction Set Computing",
    "RISC-V": "Fifth generation of the RISC instruction set",
    "RL": "Reinforcement Learning",
    "RLHF": "Reinforcement Learning with Human Feedback",
    "RNN": "Recurrent Neural Network",
    "ROS": "Robot Operating System",
    "RSVP": "Resource reSerVation Protocol",
    "RTL": "Register Transfer Level",
    "SaaS": "Software as a Service",
    "SCIF": "Secure information flow",
    "SDN": "Software-Defined Networking",
    "SDP": "Semidefinite Programming",
    "SFI": "Software Fault Isolation",
    "SIFT": "Scale-Invariant Feature Transform",
    "SIG": "Special Interest Group",
    "SIMD": "Single Instruction, Multiple Data",
    "SIMT": "Single Instruction, Multiple Threads",
    "SNARK": "Succinct Non-interactive ARgument of Knowledge",
    "SNR": "Signal to Noise Ratio",
    "SoC": "System on a Chip",
    "SOTA": "State-of-the-Art", # Common Terminology
    "SQL": "Structured Query Language",
    "SRPT": "Shortest Remaining Processing Time",
    "SSD": "Solid State Drive",
    "STS": "Science and Technology Studies", # Relevant Field
    "SVM": "Support Vector Machine",
    "TCP": "Transmission Control Protocol",
    "TDMA": "Time Division Multiple Access",
    "TF-IDF": "Term Frequency-Inverse Document Frequency",
    "TLA": "Temporal Logic of Actions",
    "TLB": "Translation Lookaside Buffer",
    "TLS": "Transport Layer Security",
    "TSP": "Traveling Salesman Problem",
    "TTS": "Text-to-Speech",
    "UI": "User Interface",
    "UNIX": "A family of operating systems", # Foundational OS type
    "URL": "Uniform Resource Locator",
    "USB": "Universal Serial Bus",
    "VCG": "Vickrey-Clarke-Groves",
    "VLSI": "Very-Large-Scale Integration",
    "VM": "Virtual Machine",
    "VR": "Virtual Reality",
    "WAN": "Wide Area Network",
    "Wi-Fi": "Wireless Fidelity", # Common Standard
    "WoZ": "Wizard of Oz", # HCI Technique
    "WWW": "World Wide Web",
}

def replace_abbreviations(query):
    """
    Replace abbreviations in the query text with their full forms.
    Example: "AI and ML" -> "Artificial Intelligence and Machine Learning"
    """
    words = query.split()
    replaced = []
    
    for word in words:
        # Check if the word is in abbreviations (case-sensitive)
        if word in abbreviations:
            replaced.append(abbreviations[word])
        # Check if the word is in abbreviations (case-insensitive)
        elif word.upper() in abbreviations:
            replaced.append(abbreviations[word.upper()])
        else:
            replaced.append(word)
    
    return " ".join(replaced)

def process_citation_range(citation_range):
    """Process citation range string into numeric bounds."""
    if not citation_range or citation_range == "0" or citation_range == "N/A":
        return float('-inf'), float('inf')
    elif citation_range == "100000":
        return 100000, float('inf')
    else:
        parts = re.sub(r'[^\d\-]', '', citation_range).split("-")
        return int(parts[0]), int(parts[1])

def print_lsi_topics(svd_model, tfidf_vectorizer, top_n=10):
    terms = tfidf_vectorizer.get_feature_names_out()
    singular_values = svd_model.singular_values_
    
    # Order topics by their singular values
    sorted_topic_indices = sorted(range(len(singular_values)), key=lambda i: singular_values[i], reverse=True)

    for i in sorted_topic_indices[:top_n]:
        print(f"\nTopic {i} (Singular Value: {singular_values[i]:.4f}):")
        top_terms = sorted(zip(terms, svd_model.components_[i]), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        for term, weight in top_terms:
            print(f"  {term:<20} {weight:.4f}")

def get_pub_similarity(prof1_key, prof2_key):
    prof1_idx = profs_to_idx[prof1_key]
    prof2_idx = profs_to_idx[prof2_key]
    return prof_themes_matrix[prof1_idx, prof2_idx]

def get_coauthor_score(prof1_key, prof2_key):
    score = 0
    prof1_id = prof1_key[1]
    prof2_id = prof2_key[1]
    if (prof1_id, prof2_id) in coauthor_score_map:
        score = np.log(coauthor_score_map[(prof1_id, prof2_id)]) / np.log(max_coauthor_score)
    elif (prof2_id, prof1_id) in coauthor_score_map:
        score = np.log(coauthor_score_map[(prof2_id, prof1_id)]) / np.log(max_coauthor_score)
    return score

def get_citation_score(prof1_key, prof2_key):
    score = 0
    prof1_id = prof1_key[1]
    prof2_id = prof2_key[1]
    if (prof1_id, prof2_id) in citation_score_map:
        score = np.log(citation_score_map[(prof1_id, prof2_id)]) / np.log(max_citation_score)
    elif (prof2_id, prof1_id) in citation_score_map:
        score = np.log(citation_score_map[(prof2_id, prof1_id)]) / np.log(max_citation_score)
    return score

def get_similar_profs(prof_key):
    res = defaultdict(lambda : defaultdict(float))
    for prof2_key in prof_keys_list:
        if prof_key == prof2_key:
            continue
        # {prof_key : {pub_sim : score, coauthor_score : score, citation_score : score, total_score :}, ...}
        res[prof2_key]["pub_sim"] = get_pub_similarity(prof_key, prof2_key)
        res[prof2_key]["coauthor_score"] = get_coauthor_score(prof_key, prof2_key)
        res[prof2_key]["citation_score"] = get_citation_score(prof_key, prof2_key)
        res[prof2_key]["total_score"] = (1/3) * res[prof2_key]["pub_sim"] + (1/3) * res[prof2_key]["coauthor_score"] + (1/3) * res[prof2_key]["citation_score"]
    res_sorted = [(prof_key, score_dict) for prof_key, score_dict in sorted(res.items(), key=lambda x : x[1]["total_score"], reverse=True)]
    print(prof_key[0])
    print(res_sorted)
    return res_sorted

def score_by_publications_lsi_balanced(query_vector, query_terms_set, prof_scores):
    """
    Score professors based on LSI publication relevance, with balanced per-term contribution.
    """
    query_lsi = svd.transform(query_vector)
    similarities = cosine_similarity(query_lsi, lsi_matrix).flatten()

    prof_to_doc_scores = defaultdict(lambda: defaultdict(float)) # {prof_key : {doc_idx: score}, ...}

    for term in query_terms_set:
        if term not in tfidf_vectorizer.vocabulary_:
            continue

        term_col_idx = tfidf_vectorizer.vocabulary_[term]
        term_column = tfidf_matrix[:, term_col_idx]
        doc_indices = term_column.nonzero()[0]

        prof_to_summed_score = defaultdict(float) # {prof_key : score, ...}
        for doc_index in doc_indices:
            prof_key = prof_index_map[doc_index]
            doc_score = similarities[doc_index]
            prof_to_summed_score[prof_key] += doc_score
        
        max_score = max(prof_to_summed_score.values())
        if max_score == 0:
            continue

        for doc_index in doc_indices:
            prof_key = prof_index_map[doc_index]
            normalized_score = similarities[doc_index] / max_score
            prof_to_doc_scores[prof_key][doc_index] += normalized_score
            prof_scores[prof_key]['publication_score'] += normalized_score
    
    return {prof_key : dict(sorted(score_dict.items(), key= lambda x: x[1], reverse=True)) for prof_key, score_dict in prof_to_doc_scores.items()}

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

def calculate_final_scores(prof_scores):
    """Calculate final weighted scores with different weights."""
    weights = {
        'publication_score': 0.7,  # 70% weight to publications
        'interest_score': 0.3,     # 30% weight to interests
        # 'citation_score': 0.2      # 20% weight to citations
    }
    for prof_key, scores in prof_scores.items():
        total_score = sum(scores[factor] * weights[factor] for factor in weights)
        prof_scores[prof_key]['total_score'] = total_score

def get_relevant_publications(prof_key, prof_to_doc_scores):
    if prof_key not in prof_to_doc_scores or not prof_to_doc_scores[prof_key]:
        return []
    return [corpus[doc_idx] for doc_idx, _ in prof_to_doc_scores[prof_key].items()][:3]

def get_relevant_coauthors(prof_key, query_vector):
    """Get top 3 most relevant coauthors for a professor using cosine similarity with TF-IDF"""
    profs_to_pubs = {} # {prof_id : [pub1, pub2, ...], ...}
    prof_scores = defaultdict(int) # {prof_id : score}
    with open("network/coauthor_edgelist.csv", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for edge in reader:
            if edge["Source"] == prof_key[1] or edge["Target"] == prof_key[1]:
                pub_list = literal_eval(edge["shared_publications"])
                if edge["Source"] == prof_key[1]:
                    profs_to_pubs[edge["Target"]] = pub_list
                else:
                    profs_to_pubs[edge["Source"]] = pub_list
    
    for prof_id, pubs in profs_to_pubs.items():
        for pub in pubs: 
            doc_vector = tfidf_matrix[publications_to_idx[pub]]
            sim = cosine_similarity(query_vector, doc_vector)[0][0]
            if sim > 0:
                prof_scores[prof_id] += sim

    return [(profid_to_name[coauthor], coauthor) for coauthor, _ in sorted(prof_scores.items(), key=lambda x: x[1], reverse=True)[:3]]

def prepare_results(ranked_profs, query_vector, prof_to_doc_scores, prof_scores):
    """Prepare final results with professor details and relevant publications."""
    results = []
    
    for prof_key, _ in ranked_profs:
        prof_name, prof_id = prof_key
        prof_data = next((p for p in dataset if p["id"] == prof_id), None)
        
        if prof_data:
            # Get top 3 relevant publications
            relevant_pubs = get_relevant_publications(prof_key, prof_to_doc_scores)
            similar_profs = get_similar_profs(prof_key)[:3]
            themes = prof_to_themes[prof_key]
            coauthors = get_relevant_coauthors(prof_key, query_vector)  # [(coauthor_name, coauthor_id)]

            results.append({
                "name": prof_name,
                "id": prof_data.get("id", ""),
                "affiliation": prof_data.get("affiliation", "Cornell University"),
                "interests": prof_data.get("interests", []),
                "citations": prof_to_citations[prof_key],
                "publications": relevant_pubs,
                "coauthors": coauthors,
                "theme_scores": themes,
                "similar_profs" : similar_profs
            })
    return results

def combined_search(query, citation_range=None):
    """
    Search for professors based on query terms with weighted factors, considering 
    professor interests, publications and citations. 
    Returns list of top 5 professors with their details and top 3 relevant publications.
    """
    query_terms_set = get_query_terms(query, count_vectorizer_analyze, tfidf_vectorizer)
    query_vector = tfidf_vectorizer.transform([query])
    
    prof_scores = defaultdict(lambda: {
        'publication_score': 0,  # TF-IDF
        'interest_score': 0,     # Jaccard similarity
        # 'citation_score': 0,     # Normalized citations
        'total_score': 0         # Weighted combination
    })
    
    citation_low, citation_high = process_citation_range(citation_range)
    prof_to_doc_scores = score_by_publications_lsi_balanced(query_vector, query_terms_set, prof_scores)

    calculate_final_scores(prof_scores)

    filtered_prof_scores = {
        prof_key: scores for prof_key, scores in prof_scores.items()
        if citation_low <= prof_to_citations[prof_key] <= citation_high
    }

    ranked_profs = sorted(
        [(prof_key, scores) for prof_key, scores in filtered_prof_scores.items()],
        key=lambda x: x[1]['total_score'],
        reverse=True)[:5]
    
    res = prepare_results(ranked_profs, query_vector, prof_to_doc_scores, prof_scores)
    return res

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/suggest")
def suggest():
    user_input = request.args.get("input", "").lower().strip()
    matches = [
        word for word in VALID_ENGLISH
        if word.startswith(user_input)
    ]
    # sort by descending language frequency:
    matches.sort(key=lambda w: zipf_frequency(w, "en"), reverse=True)
    return jsonify(matches[:5])

@app.route('/search')
def search():
    query = request.args.get("query", "")
    citation_range = request.args.get("citations", "0")
    replaced_query = replace_abbreviations(query)
    print(replaced_query)
    results = combined_search(replaced_query, citation_range) # Change the algorithm method here


    print(results)
    return jsonify(results)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)