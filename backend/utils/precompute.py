import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from preprocessing import custom_tokenizer_lemmatize, default_dict_int
from sklearn.decomposition import TruncatedSVD
import pickle
import lzma

# SVD ANALYSIS
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# import matplotlib.patches as mpatches
# from collections import Counter

# GLOBAL VARIABLES
inverted_index = defaultdict(default_dict_int)  # {term: {prof_key: term_freq}}
interest_index = defaultdict(set)                       # {interest_term: {prof_key1, prof_key2, ...}}
prof_to_citations = {}                                  # {prof_key: citations}
prof_to_publications = {}                               # {prof_key: [publications]}
prof_to_interests = {}                                  # {prof_key: [interests]}
prof_index_map = {}                                     # {doc_index: prof_key}
profid_to_name = {}                                     # {prof_id: prof_name}
cleaned_to_original = {}                                # {processed publication: original publication}
publications_to_idx = None
corpus = []                                             # All publications for TF-IDF
tfidf_vectorizer = None
count_vectorizer = None
tfidf_matrix = None
svd = None
lsi_matrix = None
theme_axes = {}

# SVD ANALYSIS
terms = None

# Abbreviations dictionary
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

def replace_abbreviations(text):
    """
    Replace abbreviations in the text with their full forms.
    """
    words = text.split()
    replaced = []
    
    for word in words:
        if word in abbreviations:
            replaced.append(abbreviations[word])
        elif word.upper() in abbreviations:
            replaced.append(abbreviations[word.upper()])
        else:
            replaced.append(word)
    
    return " ".join(replaced)

def load_data():
    with open("../profs_and_publications.json", "r", encoding="utf-8") as f:
        return json.load(f)

def build_indices(data):
    global tfidf_vectorizer, tfidf_matrix, publications_to_idx, svd, lsi_matrix, count_vectorizer, terms

    for prof in data:
        prof_key = (prof["name"], prof["id"])
        profid_to_name[prof["id"]] = prof["name"]
        prof_to_citations[prof_key] = prof.get("citations", 0)
        
        # Replace abbreviations in publications
        processed_publications = []
        for publication in prof.get("publications", []):
            processed_pub = replace_abbreviations(publication)
            cleaned_to_original[processed_pub] = publication
            processed_publications.append(processed_pub)
            corpus.append(processed_pub)
            prof_index_map[len(corpus) - 1] = prof_key
            for term in custom_tokenizer_lemmatize(processed_pub):
                inverted_index[term][prof_key] += 1
        prof_to_publications[prof_key] = processed_publications

        # Replace abbreviations in interests
        processed_interests = []
        for interest in prof.get("interests", []):
            processed_interest = replace_abbreviations(interest)
            processed_interests.append(processed_interest)
            for term in custom_tokenizer_lemmatize(processed_interest):
                interest_index[term].add(prof_key)
        prof_to_interests[prof_key] = processed_interests

    tfidf_vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer_lemmatize,     
        ngram_range=(1, 3),             
        stop_words=None,                
        max_df=0.85,                    
        min_df=1,                       
        norm='l2'                       
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    # SVD ANALYSIS
    terms = tfidf_vectorizer.get_feature_names_out()

    # Create publications_to_idx mapping using both original and processed titles
    publications_to_idx = {}
    for prof in data:
        for orig_pub, proc_pub in zip(prof.get("publications", []), prof_to_publications[(prof["name"], prof["id"])]):
            idx = corpus.index(proc_pub)
            publications_to_idx[orig_pub] = idx
            publications_to_idx[proc_pub] = idx

    svd = TruncatedSVD(n_components=100)
    lsi_matrix = svd.fit_transform(tfidf_matrix)

    # start = time.time()
    count_vectorizer = CountVectorizer(ngram_range=(1, 3), tokenizer=custom_tokenizer_lemmatize)
    count_vectorizer.fit(corpus)
    # print(f"count_vectorizer: {(time.time() - start):.4f}")
    analyze = count_vectorizer.build_analyzer()

    # SVD ANALYSIS
    theme_keywords = {
        1: ["network","sensor","model","learning","ml","machine"],  # ML, models, sensors, networks
        2: ["artificial","intelligence","language", "ai"],  # AI, language
        3: ["logic","semantics","verification","proof","algebra", "math", "formal", "linear"],  # formal logic, math, algebra
        4: ["algorithm","optimization","complexity", "computation","theory", "dynamic", "programming"],  # computation, algorithms, programming
        5: ["social","human","interaction","identity","citizen","community","people","society", "ethics", "law", "moral"],  # social, human
        6: ["security","market","game","encryption","authentication"],  # security, markets
        7: ["statistics","analysis","data","modeling","regression","probability","graph"],  # statistics, data, probability
        8: ["biology","ecology","biodiversity","health","environment","animals","survey","pharmacy","disability"],  # science, health, bio
    }

    for tid, kws in theme_keywords.items():
        pseudo = " ".join(kws)
        tf    = tfidf_vectorizer.transform([pseudo])
        theme_axes[tid] = svd.transform(tf).flatten()

    with lzma.open("precomputed_data.pkl", "wb") as f:
        pickle.dump({"dataset" : data,
                    "inverted_index" : inverted_index, 
                     "interest_index" : interest_index, 
                     "prof_to_citations" : prof_to_citations, 
                     "prof_to_publications" : prof_to_publications, 
                     "prof_to_interests" : prof_to_interests, 
                     "prof_index_map" : prof_index_map,
                     "profid_to_name" : profid_to_name,
                     "cleaned_to_original" : cleaned_to_original,
                     "publications_to_idx" : publications_to_idx,
                     "corpus" : corpus,
                     "tfidf_vectorizer" : tfidf_vectorizer,
                     "count_vectorizer_analyze" : analyze,
                     "tfidf_matrix" : tfidf_matrix,
                     "svd" : svd,
                     "lsi_matrix" : lsi_matrix,
                     "theme_axes" : theme_axes,}, f)

def top_terms_for_component(comp_idx, n=5):
    component = svd.components_[comp_idx]
    top_idxs = np.argsort(component)[-n:][::-1]
    return [tfidf_vectorizer.get_feature_names_out()[i] for i in top_idxs]

# def top_titles_for_component(comp_idx, n=5):
#     comp_vals = lsi_matrix[:, comp_idx]
#     top_doc_idxs = np.argsort(np.abs(comp_vals))[-n:][::-1]
#     return [cleaned_to_original[corpus[i]] for i in top_doc_idxs]

# def clustering():
#     global km
#     km = KMeans(n_clusters=20, random_state=42)
#     km.fit(lsi_matrix[:, :20])
#     centroids = km.cluster_centers_

#     print("\nCluster Centroid Analysis\n")
#     for cid, center in enumerate(centroids):
#         top_dims = np.argsort(np.abs(center))[-3:][::-1]
#         names = [f"LSI_{d+1}" for d in top_dims]
#         terms = [top_terms_for_component(d, n=5) for d in top_dims]
#         print(f"Cluster {cid}: top dims = {names}")
#         for d, tlist in zip(top_dims, terms):
#             print("  LSI", d+1, "→", tlist)
#         print()

#     # Count frequency of LSI dimensions
#     all_dims = []
#     for center in centroids:
#         top_dims = np.argsort(np.abs(center))[-3:]
#         all_dims.extend(top_dims)

#     dim_counts = Counter(all_dims)
#     most_common_dims = dim_counts.most_common(20)

#     print("\nTop Themes by Cluster Frequency\n")
#     for dim, freq in most_common_dims:
#         print(f"LSI {dim+1} (appears in {freq} clusters):")
#         print("  Top terms:", top_terms_for_component(dim))
#         print("  Top publications:")
#         for title in top_titles_for_component(dim):
#             print("   →", title[:100], "...")
#         print()
    
def inspect_theme_axis(axis_vec, n_components=5, n_terms=5):
    top_dims = np.argsort(np.abs(axis_vec))[-n_components:][::-1]
    out = []
    for d in top_dims:
        terms = top_terms_for_component(d, n=n_terms)
        out.append((d+1, terms))
    return out

def top_docs_for_axis(axis_vec, n=5):
    scores = lsi_matrix.dot(axis_vec)
    idxs   = np.argsort(scores)[-n:][::-1]
    return [cleaned_to_original[corpus[i]] for i in idxs]
    
def dims():
    theme_keywords = {
        1: ["network","sensor","model","learning","ml","machine"],
        2: ["artificial","intelligence","language", "ai"],
        3: ["logic","semantics","verification","proof","algebra", "math", "formal", "linear"], # formal methods and algebra
        4: ["algorithm","optimization","complexity", "computation","theory", "dynamic", "programming"],
        5: ["social","human","interaction","identity","citizen","community","people","society", "ethics", "law", "moral"],
        6: ["security","market","game","encryption","authentication"],
        7: ["statistics","analysis","data","modeling","regression","probability","graph"],
        8: ["biology","ecology","biodiversity","health","environment","animals","survey","pharmacy","disability"],
    }

    # 2) Build each theme’s 100-dim LSI axis
    for tid, kws in theme_keywords.items():
        pseudo = " ".join(kws)
        tf    = tfidf_vectorizer.transform([pseudo])
        theme_axes[tid] = svd.transform(tf).flatten()

    # 3) Inspect top LSI dims per axis
    for tid, axis_vec in theme_axes.items():
        print(f"\nTheme {tid}:", theme_keywords[tid])
        for dim, terms in inspect_theme_axis(axis_vec, n_components=3, n_terms=5):
            print(f"  → LSI {dim}: {terms}")

    # 4) Show top documents per axis
    for tid, axis_vec in theme_axes.items():
        print(f"\nTheme {tid}: top docs →")
        for title in top_docs_for_axis(axis_vec, n=5):
            print("   ", title[:80], "…")


if __name__ == "__main__":
    data = load_data()
    build_indices(data)
    # clustering()
    # analyze_and_export()
    # dims()