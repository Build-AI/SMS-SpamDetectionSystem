from nltk import data
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

# Global variables
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer = nf.process_text)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

spam_file = "spam.csv"

def main():
    data_frame = nf.read_csv(spam_file)
    ham_data = nf.extract_ham(data_frame)
    spam_data = nf.extract_spam(data_frame)
    nf.create_data_model(data_frame)
    print("Length of data frame: ", len(data_frame))

    # pipeline.fit(text_train,type_train)
    # predictions = pipeline.predict(text_test)
    # print('Naive Base Accuracy_score: ',accuracy_score(type_test,predictions))
    # print(classification_report(predictions,type_test))

main()
