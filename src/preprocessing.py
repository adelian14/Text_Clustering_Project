import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
nltk.download('stopwords')

def preprocess(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Removing digits
    text = re.sub(r'\s+', ' ', text)  # Removing extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    text = re.sub(r'\b\w{25,}\b', '', text)  # Removing long words
    text = re.sub(r'\S+@\S+', '', text)  # Removing emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Removing URLs
    text = re.sub(r'[^\x00-\x7F]+', '', text) #removing non-ASCII characters
    text = re.sub(r'<.*?>', '', text) #removing HTML tags
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

def prepare_data(x):
    temp = list(x)
    for i in range(len(temp)):
        temp[i] = preprocess(temp[i])
    return np.array(temp)