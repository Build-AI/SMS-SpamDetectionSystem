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

class Training_data:
    def __init__(self, training_txt, test_txt, training_type, test_type):
        self.training_txt = training_txt
        self.test_txt = test_txt
        self.training_type = training_type
        self.test_type = test_type

    # def set_training_data(self):
    #     self.training_txt = ''
    #     self.test_txt = ''
    #     self.training_type = ''
    #     self.test_type = ''

    # def get_training_data(self):
    #     return self.training_txt, self.test_txt, self.training_type, self.test_type
        
def read_csv(spam_file):
    data_frame = pd.read_csv(spam_file, encoding = "ISO-8859-1")
    # data_frame['length'] = data_frame['text'].apply(len) # Adds Length of chars in 'text'
    return data_frame

# Removes Punctuation and English stop words in text column of csv file
def process_text(msg):
    no_punc = [char for char in msg if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words("english")]

# Vectorizes text & counts total number of words
def find_total_words(data_frame):
    # Take each word found in the 'text' column of our data & returns the total qty
    bow_transformer = CountVectorizer(analyzer = process_text).fit(data_frame["text"])
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
def create_data_model(data_frame):
    text_train, text_test, type_train, type_test = train_test_split(data_frame["text"], data_frame["type"], test_size = 0.3, random_state = 37)

    #testing if this is working
    data_set = Training_data(text_train, text_test, type_train, type_test)

    print(data_set.training_txt, data_set.test_txt, data_set.training_type, data_set.test_type)