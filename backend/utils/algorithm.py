from similarity import cosine_sim, calculate_jaccard_similarity
from .indices import interest_index, prof_to_citations, prof_to_publications, prof_to_interests, prof_index_map, corpus, tfidf_vectorizer, tfidf_matrix
from .preprocessing import preprocess_text
from collections import defaultdict

def process_citation_range(citation_range):
    """Process citation range string into numeric bounds."""
    if not citation_range or citation_range == "0" or citation_range == "N/A":
        return float('-inf'), float('inf')
    elif citation_range == "100000":
        return 100000, float('inf')
    else:
        parts = re.sub(r'[^\d\-]', '', citation_range).split("-")
        return int(parts[0]), int(parts[1])

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