from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, re
from collections import defaultdict
from utils import preprocessing, indices
from utils.indices import get_query_terms
from utils.similarity import calculate_jaccard_similarity
import csv
from nltk.stem import WordNetLemmatizer
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
import time
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
# cleaned_to_original = data["cleaned_to_original"]
publications_to_idx = data["publications_to_idx"]
corpus = data["corpus"]
svd = data["svd"]
tfidf_vectorizer = data["tfidf_vectorizer"]
count_vectorizer_analyze = data["count_vectorizer_analyze"]
tfidf_matrix = data["tfidf_matrix"]
lsi_matrix = data["lsi_matrix"]
theme_axes = data["theme_axes"]
l = data["dummy"]

# load_start=  time.time()
# data = load_data()
# load_end = time.time()
# print(f"load_data(): {(load_end - load_start):.4f} s")

# build_start = time.time()
# build_indices(data)
# build_end = time.time()
# print(f"build_data(): {(build_end - build_start):.4f} s")


MAX_DOC_SCORE_CONTRIBUTION = 0.4

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

# def score_by_publications(query_vector_array, prof_scores):
#     """Score professors by publication similarity by TF-IDF."""
#     for idx, doc_vector in enumerate(indices.tfidf_matrix):
#         doc_vector_array = doc_vector.toarray()[0]
#         similarity = cosine_sim(query_vector_array, doc_vector_array)
        
#         if similarity > 0:
#             prof_key = indices.prof_index_map[idx]
#             prof_scores[prof_key]['publication_score'] += similarity

# def score_by_publications_lsi(query_vector, prof_scores):
#     prof_scores_list = defaultdict(list)
#     query_lsi = indices.svd.transform(query_vector)

#     similarities = cosine_similarity(query_lsi, indices.lsi_matrix).flatten()

#     for doc_index, score in enumerate(similarities):
#         prof_key = indices.prof_index_map[doc_index]
#         prof_scores_list[prof_key].append(score)

#     for key, scores in prof_scores_list.items():
#         prof_scores_list[key].sort()
#         prof_scores[key]['publication_score'] = sum(scores[-10:])


def score_professor_themes(prof_key):
    """
    Compute each professor's 8-theme score.
    """
    idxs = [i for i, p in prof_index_map.items() if p == prof_key]  # all docs authored by prof
    P = lsi_matrix[idxs]

    # compute mean dot-product with each theme axis
    raw = {theme_id: float(np.dot(P, axis_vec).mean()) for theme_id, axis_vec in theme_axes.items()}
    mx  = max(raw.values()) or 1.0
    return {theme_id: raw[theme_id]/mx for theme_id in raw}

def score_by_publications_lsi_balanced(query_vector, query_terms_set, prof_scores):
    """
    Score professors based on LSI publication relevance, with balanced per-term contribution.
    """
    query_lsi = svd.transform(query_vector)
    similarities = cosine_similarity(query_lsi, lsi_matrix).flatten()

    # term_weights = defaultdict(lambda: defaultdict(float))  # term -> prof_key -> score
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

    # for term, prof_scores_map in term_weights_expanded.items():
    #     if not prof_scores_map:
    #         continue
    #     summed_scores = [sum(score for score in doc_scores_map.values()) for doc_scores_map in prof_scores_map.values()]
    #     max_score = max(summed_scores)
    #     if max_score == 0:
    #         continue
    #     for prof_key, doc_scores_map in prof_scores_map.items():
    #         for doc_idx in doc_scores_map:
    #             term_weights_expanded[term][prof_key][doc_idx] /= max_score
    #             prof_scores[prof_key]['publication_score'] += term_weights_expanded[term][prof_key][doc_idx]
            
    # prof_to_scores_expanded = defaultdict(lambda : defaultdict(float)) # {prof_key : {doc_idx : score}, ...}
    # for term, prof_map in term_weights_expanded.items():
    #     for prof_key, doc_map in prof_map.items():
    #         for doc_idx, score in doc_map.items():
    #             prof_to_scores_expanded[prof_key][doc_idx] += score

    # return {prof_key : dict(sorted(score_dict.items(), key= lambda x: x[1], reverse=True)) for prof_key, score_dict in prof_to_scores_expanded.items()}


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

# def score_by_citations(citation_low, citation_high, prof_scores):
#     """
#     Score professors based on citation counts within range. 
#     Normalized to make sure higher citation counts are not unfairly weighed higher
#     If citations not in range, still considered but penalized 
#     """
#     max_citations = max(indices.prof_to_citations.values()) if indices.prof_to_citations else 1
    
#     for prof_key, citations in indices.prof_to_citations.items():
#         normalized_score = citations / max_citations
#         if citation_low <= citations <= citation_high:
#             prof_scores[prof_key]['citation_score'] = normalized_score
#         else:
#             prof_scores[prof_key]['citation_score'] = normalized_score * 0.5 #If not in range, less weight

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

# def get_relevant_publications(prof_key, query_vector):
#     """Get top 3 most relevant publications for a professor using TF-IDF similarity."""
#     publications = indices.prof_to_publications[prof_key]
#     pub_scores = []
#     for publication in publications:
#         pub_vector = indices.tfidf_vectorizer.transform([publication])
#         similarity = cosine_similarity(query_vector, pub_vector)[0][0]
#         og_title = indices.cleaned_to_original.get(publication, publication)
#         pub_scores.append((og_title, similarity))
#     return [pub for pub, _ in sorted(pub_scores, key=lambda x: x[1], reverse=True)[:3]]

def get_relevant_publications(prof_key, prof_to_doc_scores):
    return [corpus[doc_idx] for doc_idx, _ in prof_to_doc_scores[prof_key].items()][:3]

def get_relevant_coauthors(prof_key, query_vector):
    """Get top 3 most relevant coauthors for a professor using cosine similarity with TF-IDF"""
    profs_to_pubs = {} # {prof_id : [pub1, pub2, ...], ...}
    prof_scores = defaultdict(int) # {prof_id : score}
    with open("network/coauthor_network_edge.csv") as file:
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

def prepare_results(ranked_profs, query_vector, prof_to_doc_scores):
    """Prepare final results with professor details and relevant publications."""
    results = []
    
    rel_pubs_accum = 0 
    co_authors_accum = 0
    for prof_key, _ in ranked_profs:
        prof_name, prof_id = prof_key
        prof_data = next((p for p in dataset if p["id"] == prof_id), None)
        
        if prof_data:
            # Get top 3 relevant publications
            start = time.time()
            relevant_pubs = get_relevant_publications(prof_key, prof_to_doc_scores)
            rel_pubs_accum += (time.time() - start)
            start = time.time()
            coauthors = get_relevant_coauthors(prof_key, query_vector)  # [(coauthor_name, coauthor_id)]
            co_authors_accum += (time.time() - start)

            results.append({
                "name": prof_name,
                "id": prof_data.get("id", ""),
                "affiliation": prof_data.get("affiliation", "Cornell University"),
                "interests": prof_data.get("interests", []),
                "citations": prof_to_citations[prof_key],
                "publications": relevant_pubs,
                "coauthors": coauthors,
                "theme_scores": score_professor_themes(prof_key)
            })
    print(f"get_relevant_publications(): {rel_pubs_accum:.4f}")
    print(f"get_relevant_coauthors(): {co_authors_accum:.4f}")
    return results

def combined_search(query, citation_range=None):
    """
    Search for professors based on query terms with weighted factors, considering 
    professor interests, publications and citations. 
    Returns list of top 5 professors with their details and top 3 relevant publications.
    """
    # query_terms_set = set(custom_tokenizer(query))
    # if not query_terms_set:
    #     return []
    start = time.time()
    query_terms_set = get_query_terms(query, count_vectorizer_analyze, tfidf_vectorizer)
    print(f"get_query_terms(): {(time.time() - start):.4f}")
    
    # Convert query to TF-IDF vector for publication matching
    start = time.time()
    query_vector = tfidf_vectorizer.transform([query])
    print(f"query_vector(): {(time.time() - start):.4f}")
    # query_vector_array = query_vector.toarray()[0]  # raw tfidf scoring
    
    prof_scores = defaultdict(lambda: {
        'publication_score': 0,  # TF-IDF
        'interest_score': 0,     # Jaccard similarity
        # 'citation_score': 0,     # Normalized citations
        'total_score': 0         # Weighted combination
    })
    
    start = time.time()
    citation_low, citation_high = process_citation_range(citation_range)
    print(f"process_citation_range(): {(time.time() - start):.4f}")

    # score_by_publications(query_vector_array, prof_scores)
    # score_by_publications_lsi(query_vector, prof_scores)
    start = time.time()
    prof_to_doc_scores = score_by_publications_lsi_balanced(query_vector, query_terms_set, prof_scores)
    print(f"score_by_pubs(): {(time.time() - start):.4f}")
    start = time.time()
    score_by_interests(query_terms_set, prof_scores)
    print(f"score_by_interests(): {(time.time() - start):.4f}")
    # score_by_citations(citation_low, citation_high, prof_scores)

    start = time.time()
    calculate_final_scores(prof_scores)
    print(f"calc_final_scores(): {(time.time() - start):.4f}")

    # filter profs based on citation range
    start = time.time()
    filtered_prof_scores = {
        prof_key: scores for prof_key, scores in prof_scores.items()
        if citation_low <= prof_to_citations[prof_key] <= citation_high
    }
    print(f"filter prof scores: {(time.time() - start):.4f}")

    print(prof_scores) #debugginggg 
    start = time.time()
    ranked_profs = sorted(
        [(prof_key, scores) for prof_key, scores in filtered_prof_scores.items()],
        key=lambda x: x[1]['total_score'],
        reverse=True)[:5]
    print(f"rank profs: {(time.time() - start):.4f}")
    print(ranked_profs)

    start = time.time()
    res = prepare_results(ranked_profs, query_vector, prof_to_doc_scores)
    print(f"prepare_results: {(time.time() - start):.4f}")
    for prof in res: # debugging
        print(prof["name"])
        print(prof["coauthors"])
        print() 
    return res

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route("/suggest")
def suggest():
    user_input = request.args.get("input", "").lower()
    suggestions = set()

    for word in inverted_index:
        if word.startswith(user_input) and word in preprocessing.english_words:
            suggestions.add(word)

    return jsonify(sorted(suggestions)[:5])

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