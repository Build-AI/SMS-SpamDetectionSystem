import pandas as pd
from scipy.sparse import data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

def main():
    data_file = pd.read_csv('spam.csv', encoding = "ISO-8859-1")
    data_file['label'] = data_file['type'].map({'ham': 'HAM', 'spam': 'SPAM'})

    file_text = data_file['text']
    file_label = data_file['label']

    cv = CountVectorizer()
    file_text = cv.fit_transform(file_text)
    x_train, x_test, y_train, y_test = train_test_split(file_text, file_label, test_size=0.3, random_state=42)

    #naive bayes
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)

    message = ''
    while not (message == 'quit'):
        message = input("Enter a message: ")
        message_data = [message]
        vect = cv.transform(message_data).toarray()
        predict_message = clf.predict(vect)
        print(predict_message)

main()