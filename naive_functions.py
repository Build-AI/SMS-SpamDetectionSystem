import numpy as np
import pandas as pd 
import string
import re

import nltk
from nltk import data
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class Training_data:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def set_training_data(self):
        self.x_train = ''
        self.x_test = ''
        self.y_train = ''
        self.y_test = ''

    def get_training_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test
        
def read_csv(spam_file):
    data_frame = pd.read_csv(spam_file, encoding = "ISO-8859-1")
    # data_frame['length'] = data_frame['text'].apply(len) # Adds Length of chars in 'text'
    return data_frame

# Removes Punctuation and English stop words in text column of csv file
def removes_punc(msg):
    no_punc = [char for char in msg if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words("english")]

# Vectorizes text & counts total number of words
def find_total_words(data_frame):
    # Take each word found in the 'text' column of our data & returns the total qty
    bow_transformer = CountVectorizer(analyzer = removes_punc).fit(data_frame["text"])
    return (len(bow_transformer.vocabulary_))

# Extracts all msgs classified as spam
def extract_spam(data_frame):
    spam_msgs = data_frame[data_frame["type"] == "spam"]
    return spam_msgs

# Extracts all msgs classified as ham
def extract_ham(data_frame):
    ham_msgs = data_frame[data_frame["type"] == "ham"]
    return ham_msgs

# Setting up Training & Testing Data from our data frame 
# Returns our training data obj
def create_data_model(data_frame):
    x_train, x_test, y_train, y_test = train_test_split(data_frame["text"], data_frame["type"], test_size = 0.3, random_state = 37)
    
    # Calling & Initiating Training data constructor 
    data_set = Training_data(x_train, x_test, y_train, y_test)
    print(data_set.x_train, data_set.x_test, data_set.y_train, data_set.y_test)
    return data_set

def corpus(data_frame, training_data):
    corpus_list = []

    for i in range(0, len(data_frame)):
        review = re.sub('[^a-zA-Z]', ' ', data_frame['text'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
    
    cv = CountVectorizer(max_features = 3000)
    cv.fit(training_data.x_train)

    x_train_cv = cv.transform(training_data.x_train)
    x_test_cv = cv.transform(training_data.x_test)

