from nltk import data
import numpy as np
import pandas as pd 
import string
import nltk
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def print_stats():
    bow_transformer = CountVectorizer(analyzer=process_text).fit(data_frame['text'])
    unique_text = data_frame['text'][3]

    bow4 = bow_transformer.transform([unique_text])
    text_bow = bow_transformer.transform(data_frame['text'])
    tfidf_transformer=TfidfTransformer().fit(text_bow)
    tfidf4 = tfidf_transformer.transform(bow4)
    text_tfidf=tfidf_transformer.transform(text_bow)


    print(classification_report(data_frame['type'], all_predictions))
    print(confusion_matrix(data_frame['type'], all_predictions))
    print ("accuracy_score : ", accuracy_score(type_test, predictions))

# Removes Punctuation and English stop words in text column of csv file
def process_text(msg):
    no_punc = [char for char in msg if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

# Vectorizes text & counts total number of words
def find_total_words():
    bow_transformer = CountVectorizer(analyzer=process_text).fit(data_frame['text'])
    return (len(bow_transformer.vocabulary_))

data_frame = pd.read_csv('spam.csv', encoding = "ISO-8859-1")
data_frame['length'] = data_frame['text'].apply(len) # Length of chars in text
data_frame.head()
print(data_frame.shape)

# spam_dataframe = data_frame.loc[(data_frame.type == 'spam')]
# print(spam_dataframe.describe())
# print(data_frame.describe())

data_frame['text'].head(5).apply(process_text)
data_frame.head()
print(data_frame.head())

# bow_transformer = CountVectorizer(analyzer=process_text).fit(data_frame['text'])
# unique_text = data_frame['text'][3]
# print(unique_text)

# bow4=bow_transformer.transform([unique_text])
# text_bow = bow_transformer.transform(data_frame['text'])

# print('Shape of Sparse Matrix: ', text_bow.shape)
# print('Amount of non-zero occurences:', text_bow.nnz)

# sparsity =(100.0 * text_bow.nnz/(text_bow.shape[0]*text_bow.shape[1]))
# print('sparsity:{}'.format(round(sparsity)))

from sklearn.model_selection import train_test_split
text_train,text_test,type_train,type_test = train_test_split(data_frame['text'], data_frame['type'],test_size=0.2)


