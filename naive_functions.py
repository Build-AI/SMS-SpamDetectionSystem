from nltk import data
import numpy as np
import pandas as pd 
import string
import nltk
import string
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def read_csv(spam_file):
    data_frame = pd.read_csv(spam_file, encoding = "ISO-8859-1")
    # data_frame['length'] = data_frame['text'].apply(len) # Adds Length of chars in 'text'
    return data_frame

# Removes Punctuation and English stop words in text column of csv file
def process_text(msg):
    no_punc = [char for char in msg if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

# Vectorizes text & counts total number of words
def find_total_words(data_frame):
    # Take each word found in the 'text' column of our data & returns the total qty
    bow_transformer = CountVectorizer(analyzer = process_text).fit(data_frame['text'])
    return (len(bow_transformer.vocabulary_))

# Extracts all msgs classified as spam
def extract_spam(data_frame):
    spam_msgs = data_frame[data_frame["type"] == "spam"]
    return spam_msgs

# Extracts all msgs classified as ham
def extract_ham(data_frame):
    ham_msgs = data_frame[data_frame["type"] == "ham"]
    return ham_msgs

