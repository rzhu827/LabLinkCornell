import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import words
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("words")
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import defaultdict

stemmer = PorterStemmer()
stemmed_stop_words = [stemmer.stem(word) for word in ENGLISH_STOP_WORDS]

lemmatizer = WordNetLemmatizer()

english_words = set(w.lower() for w in words.words())

def custom_tokenizer_stem(text):
    """Custom tokenizer for stemming and tokenization."""

    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()  # remove punctuation, lower case, split into tokens
    stemmed = [stemmer.stem(token) for token in tokens]  # stem each token
    return [t for t in stemmed if t not in stemmed_stop_words]  # filter out stop words

def custom_tokenizer_lemmatize(text):
    """Custom tokenizer for lemmatization"""
    # start = time.time()
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    res = [lemmatizer.lemmatize(token) for token in tokens if token not in ENGLISH_STOP_WORDS]
    # print(f"custom_tokenizer_lemmatize(): {(time.time() - start):.4f}")
    return res

def preprocess_text(text):
    """Clean and tokenize text."""
    # generate ngrams using the custom tokenizer
    vectorizer = CountVectorizer(ngram_range=(1, 3), tokenizer=custom_tokenizer_lemmatize)
    ngrams = vectorizer.fit_transform([text])  # transform the query text
    # return n-grams and the tokenized version for reference
    return ngrams, custom_tokenizer_lemmatize(text)

def default_dict_int():
    """Helper function for pickling"""
    return defaultdict(int)