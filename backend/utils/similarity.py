import numpy as np

def cosine_sim(query_vector, doc_vector):
    dot_product = np.dot(query_vector, doc_vector)
    norm = (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector)) + 1e-5
    return dot_product / norm

def calculate_jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0
    return len(set1 & set2) / len(set1 | set2)