from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, re
from collections import defaultdict
from utils.preprocessing import custom_tokenizer
from utils import indices
from utils.indices import build_indices, load_data
from utils.similarity import calculate_jaccard_similarity
import csv
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__)
CORS(app)

data = load_data()
build_indices(data)

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

def score_by_publications_lsi(query_vector, prof_scores):
    query_lsi = indices.svd.transform(query_vector)

    similarities = cosine_similarity(query_lsi, indices.lsi_matrix).flatten()

    for doc_index, score in enumerate(similarities):
        prof_key = indices.prof_index_map[doc_index]
        prof_scores[prof_key]['publication_score'] += score

def score_by_interests(query_terms, prof_scores):
    """Score professors based on interest relevance using Jaccard similarity."""
    matching_profs = set()
    for term in query_terms:
        if term in indices.interest_index:
            matching_profs.update(indices.interest_index[term])
    
    for prof_key in matching_profs:
        prof_interests = set(" ".join(indices.prof_to_interests[prof_key]).lower().split())
        similarity = calculate_jaccard_similarity(query_terms, prof_interests)
        prof_scores[prof_key]['interest_score'] = similarity

def score_by_citations(citation_low, citation_high, prof_scores):
    """
    Score professors based on citation counts within range. 
    Normalized to make sure higher citation counts are not unfairly weighed higher
    If citations not in range, still considered but penalized 
    """
    max_citations = max(indices.prof_to_citations.values()) if indices.prof_to_citations else 1
    
    for prof_key, citations in indices.prof_to_citations.items():
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
    publications = indices.prof_to_publications[prof_key]
    pub_scores = []
    for publication in publications:
        pub_vector = indices.tfidf_vectorizer.transform([publication])
        similarity = cosine_similarity(query_vector, pub_vector)[0][0]
        pub_scores.append((publication, similarity))
    return [pub for pub, _ in sorted(pub_scores, key=lambda x: x[1], reverse=True)[:3]]

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
            doc_vector = indices.tfidf_matrix[indices.publications_to_idx[pub]]
            sim = cosine_similarity(query_vector, doc_vector)[0][0]
            if sim > 0:
                prof_scores[prof_id] += sim

    return [(indices.profid_to_name[coauthor], coauthor) for coauthor, _ in sorted(prof_scores.items(), key=lambda x: x[1], reverse=True)[:3]]

def prepare_results(ranked_profs, query_vector):
    """Prepare final results with professor details and relevant publications."""
    results = []
    
    for prof_key, scores in ranked_profs:
        prof_name, prof_id = prof_key
        prof_data = next((p for p in data if p["id"] == prof_id), None)
        
        if prof_data:
            # Get top 3 relevant publications
            relevant_pubs = get_relevant_publications(prof_key, query_vector)
            coauthors = get_relevant_coauthors(prof_key, query_vector)  # [(coauthor_name, coauthor_id)]
            
            results.append({
                "name": prof_name,
                "id": prof_data.get("id", ""),
                "affiliation": prof_data.get("affiliation", "Cornell University"),
                "interests": prof_data.get("interests", []),
                "citations": indices.prof_to_citations[prof_key],
                "publications": relevant_pubs,
                "coauthors": coauthors
            })
    return results

def combined_search(query, citation_range=None):
    """
    Search for professors based on query terms with weighted factors, considering 
    professor interests, publications and citations. 
    Returns list of top 5 professors with their details and top 3 relevant publications.
    """
    query_terms_set = set(custom_tokenizer(query))
    if not query_terms_set:
        return []
    
    # Convert query to TF-IDF vector for publication matching
    query_vector = indices.tfidf_vectorizer.transform([query])
    # query_vector_array = query_vector.toarray()[0]  # raw tfidf scoring
    
    prof_scores = defaultdict(lambda: {
        'publication_score': 0,  # TF-IDF
        'interest_score': 0,     # Jaccard similarity
        'citation_score': 0,     # Normalized citations
        'total_score': 0         # Weighted combination
    })
    
    citation_low, citation_high = process_citation_range(citation_range)
    
    # score_by_publications(query_vector_array, prof_scores)
    score_by_publications_lsi(query_vector, prof_scores)
    score_by_interests(query_terms_set, prof_scores)  # utilizes token set
    score_by_citations(citation_low, citation_high, prof_scores)

    calculate_final_scores(prof_scores)

    # print(prof_scores) #debugginggg 
    ranked_profs = sorted(
        [(prof_key, scores) for prof_key, scores in prof_scores.items()],
        key=lambda x: x[1]['total_score'],
        reverse=True)[:5]

    res = prepare_results(ranked_profs, query_vector)
    # for prof in res: # debugging
    #     print(prof["name"])
    #     print(prof["coauthors"])
    #     print() 
    return res

###################
# def get_top_publications(prof_data, query_terms):
#     """
#     Given a professor's data, returns the top 3 most relevant publications
#     based on term frequency for the query terms.
#     """
#     publication_scores = []  # [ (publication, aggregated_term_freq) ]

#     for publication in prof_data["publications"]:
#         pub_terms = preprocess_text(publication)
#         pub_term_to_tf = Counter(pub_terms)

#         pub_score = 0
#         for term in query_terms:
#             pub_score += pub_term_to_tf.get(term, 0)
#         publication_scores.append((publication, pub_score))

#     return sorted(publication_scores, key=lambda x: x[1], reverse=True)[:3]


# def tf_ranked_search(query, citation_range):
#     """
#     Given a user's input query terms and citation range, returns a ranked set of
#     top 5 matched professors and their top 3 relevant publications, evaluated
#     through highest aggregate term frequency for the query terms and relevance
#     of citation quantity.
#     """
#     query_terms = set((re.sub(r'[^\w\s]', '', query)).lower().split())
#     print("Citation range:", citation_range)  # Debug: print the citation range
#     if citation_range == "0":  # no preference
#         citation_low = float('-inf')
#         citation_high = float('inf')
#     elif citation_range == "100000":  # 100,000+
#         citation_low = 100000
#         citation_high = float('inf')
#     else:
#         citation_range = citation_range.split("-")
#         citation_low = int(re.sub(r'[^\w\s]', '', citation_range[0]))
#         citation_high = int(re.sub(r'[^\w\s]', '', citation_range[1]))

#     scores = defaultdict(int)  # { prof_key : score }

#     for term in query_terms:
#         if term in inverted_index:
#             for prof_key, term_freq in inverted_index[term].items():
#                 prof_citations = prof_to_citations[prof_key]                

#                 if citation_low <= prof_citations <= citation_high:
#                     scores[prof_key] += term_freq

#     ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

#     results = []
#     for prof_key, score in ranked_scores:
#         prof_name, prof_id = prof_key
#         prof_citations = prof_to_citations[prof_key]
#         prof_data = next((p for p in data if p["id"] == prof_id), None)

#         print(f"Score for {prof_name}: {score}")  # Debugging statement

#         if prof_data:
#             top_publications = get_top_publications(prof_data, query_terms)
#             results.append({
#                 "name": prof_name,
#                 "id": prof_data.get("id", ""),
#                 "affiliation": prof_data.get("affiliation", "Cornell University"),
#                 "interests": prof_data.get("interests", []),
#                 "citations": prof_citations,
#                 "publications": [pub[0] for pub in top_publications]
#                 })
            
#     return results
###################

@app.route("/")
def home():
    return render_template('base.html',title="sample html")

@app.route('/search')
def search():
    query = request.args.get("query", "")
    citation_range = request.args.get("citations", "0")
    replaced_query = replace_abbreviations(query)
    print(replaced_query)
    results = combined_search(replaced_query, citation_range) # Change the algorithm method here

    return jsonify(results)

if 'DB_NAME' not in os.environ:
    app.run(debug=True,host="0.0.0.0",port=5000)