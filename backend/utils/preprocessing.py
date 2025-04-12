import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

stemmer = PorterStemmer()
stemmed_stop_words = [stemmer.stem(word) for word in ENGLISH_STOP_WORDS]

lemmatizer = WordNetLemmatizer()

def custom_tokenizer_stem(text):
    """Custom tokenizer for stemming and tokenization."""

    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()  # remove punctuation, lower case, split into tokens
    stemmed = [stemmer.stem(token) for token in tokens]  # stem each token
    return [t for t in stemmed if t not in stemmed_stop_words]  # filter out stop words

def custom_tokenizer(text):
    """Custom tokenizer for lemmatization"""
    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [lemmatizer.lemmatize(token) for token in tokens if token not in ENGLISH_STOP_WORDS]

def preprocess_text(text):
    """Clean and tokenize text."""

    # generate ngrams using the custom tokenizer
    vectorizer = CountVectorizer(ngram_range=(1, 3), tokenizer=custom_tokenizer)
    ngrams = vectorizer.fit_transform([text])  # transform the query text
    # return n-grams and the tokenized version for reference
    return ngrams, custom_tokenizer(text)