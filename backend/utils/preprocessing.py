import re
import nltk

nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("words")
nltk.download("stopwords")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

from nltk import pos_tag, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import words, wordnet, stopwords
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from collections import defaultdict
from wordfreq import zipf_frequency

stemmer = PorterStemmer()
stemmed_stop_words = [stemmer.stem(word) for word in ENGLISH_STOP_WORDS]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# english_words = set(w.lower() for w in words.words())

def custom_tokenizer_stem(text):
    """Custom tokenizer for stemming and tokenization."""

    tokens = re.sub(r'[^\w\s]', '', text.lower()).split()  # remove punctuation, lower case, split into tokens
    stemmed = [stemmer.stem(token) for token in tokens]  # stem each token
    return [t for t in stemmed if t not in stemmed_stop_words]  # filter out stop words

def get_wordnet_pos(treebank_tag):
    """Map part-of-speech tag to first character lemmatize() accepts."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun

def custom_tokenizer_lemmatize(text):
    """Tokenize + lemmatize, with a POS analysis."""
    text   = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    lemmas = []
    for token, tag in tagged:
        if token in stop_words:
            continue
        wn_pos = get_wordnet_pos(tag)
        lemma  = lemmatizer.lemmatize(token, wn_pos)
        lemmas.append(lemma)
    return lemmas

# def custom_tokenizer_lemmatize(text):
#     """Custom tokenizer for lemmatization and tokenization."""
#     # start = time.time()
#     tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
#     res = [lemmatizer.lemmatize(token) for token in tokens if token not in ENGLISH_STOP_WORDS]
#     # print(f"custom_tokenizer_lemmatize(): {(time.time() - start):.4f}")
#     return res

def preprocess_text(text):
    """Clean and tokenize text with ngrams. EDIT: Unused."""
    # generate ngrams using the custom tokenizer
    vectorizer = CountVectorizer(ngram_range=(1, 3), tokenizer=custom_tokenizer_lemmatize)
    ngrams = vectorizer.fit_transform([text])  # transform the query text
    # return n-grams and the tokenized version for reference
    return ngrams, custom_tokenizer_lemmatize(text)

def default_dict_int():
    """Helper function for pickling"""
    return defaultdict(int)