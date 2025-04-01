<<<<<<< HEAD
# app/utils/helpers.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """Performs text cleaning."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

def preprocess_for_finbert(text):
    """Preprocesses text for FinBERT."""
    text = clean_text(text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
=======
# app/utils/helpers.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    """Performs text cleaning."""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

def preprocess_for_finbert(text):
    """Preprocesses text for FinBERT."""
    text = clean_text(text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
>>>>>>> a2df2ffd820893e14d0aaea3c0fef2588c0fa6a3
    return " ".join(tokens)