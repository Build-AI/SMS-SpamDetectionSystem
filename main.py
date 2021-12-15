from nltk import data
from nltk import corpus
from nltk.util import pr
import numpy as np
import pandas as pd 
import string
import nltk
from nltk.corpus import stopwords
from pandas.io.parsers import read_csv 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
import naive_functions as nf
import pandas as pd

# Global variables
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer = nf.removes_punc)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

spam_file = "spam.csv"

def main():
    data_frame = nf.read_csv(spam_file)
    print(data_frame.head())

    #describing the array of the file
    print('\n')
    print(data_frame.describe())
    print('\n')
    print(data_frame.shape)
    print('\n')

    spam_data = data_frame.loc[(data_frame.type == 'spam')]
    print(spam_data.describe())

    clean_data = data_frame
    #removing punctuation/ creating a new variable to show that we're "cleaning" the text file
    clean_data['text'] = data_frame['text'].apply(lambda text: nf.removes_punc(text))
    print(clean_data.head())

    #calling find_total_words
    vectorized_words = nf.find_total_words(clean_data)
    print(vectorized_words)

    #extracting spam and ham
    spam_data = nf.extract_spam(clean_data)
    print(spam_data.head())
    print(spam_data.describe())
    print('nf')
    ham_data = nf.extract_ham(clean_data)
    print(ham_data)
    print(ham_data.describe())

    train_test = nf.create_data_model(clean_data)
    #printing lengths
    print('\n')
    print('Xtrain: ', len(train_test.x_train))
    print('Xtest: ', len(train_test.x_test))
    print('Ytrain: ', len(train_test.y_train))
    print('Ytest: ', len(train_test.y_test))

    #calling corpus
    nf.corpus(clean_data, train_test)
main()
