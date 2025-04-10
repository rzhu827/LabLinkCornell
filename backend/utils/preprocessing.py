import re

def preprocess_text(text):
    """Clean and tokenize text."""
    return re.sub(r'[^\w\s]', '', text).lower().split()