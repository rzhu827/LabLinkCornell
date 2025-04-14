import json
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils.preprocessing import custom_tokenizer_lemmatize
from sklearn.decomposition import TruncatedSVD


# GLOBAL VARIABLES
inverted_index = defaultdict(lambda: defaultdict(int))  # {term: {prof_key: term_freq}}
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
tfidf_matrix = None
svd = None
lsi_matrix = None

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
    with open("profs_and_publications.json", "r", encoding="utf-8") as f:
        return json.load(f)

def build_indices(data):
    global tfidf_vectorizer, tfidf_matrix, publications_to_idx, svd, lsi_matrix

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
        min_df=2,                       
        norm='l2'                       
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    # Create publications_to_idx mapping using both original and processed titles
    publications_to_idx = {}
    for prof in data:
        for orig_pub, proc_pub in zip(prof.get("publications", []), prof_to_publications[(prof["name"], prof["id"])]):
            idx = corpus.index(proc_pub)
            publications_to_idx[orig_pub] = idx
            publications_to_idx[proc_pub] = idx

    svd = TruncatedSVD(n_components=100)
    lsi_matrix = svd.fit_transform(tfidf_matrix)

def get_query_terms(query):
    """Get all n-grams (unigrams to trigrams) present in the query and the TF-IDF vocab."""
    vectorizer = CountVectorizer(ngram_range=(1, 3), tokenizer=custom_tokenizer_lemmatize)
    vectorizer.fit(corpus)
    ngram_tokens = vectorizer.build_analyzer()(query)
    return [term for term in ngram_tokens if term in tfidf_vectorizer.vocabulary_]