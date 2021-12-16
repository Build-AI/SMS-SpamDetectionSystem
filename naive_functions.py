import pandas as pd 
import string
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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

# Reads csv files & single-byte encodes data
def read_file():
    data_file = pd.read_csv('spam.csv', encoding = "ISO-8859-1")
    data_file['label'] = data_file['type'].map({'ham': 'HAM', 'spam': 'SPAM'})
    return data_file

# Removes Punctuation and English stop words in text column of csv file
def removes_punc(msg):
    no_punc = [char for char in msg if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return[word for word in no_punc.split() if word.lower() not in stopwords.words("english")]

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

# Displays Menu & Verifies User-Input
def display_menu():
    menu_input = input("\
    \t-------- Menu --------\n\
    1. View msgs classified as ham\n\
    2. View msgs classified as spam\n\
    3. Input a message\n\
    4. Quit\n\
    Please input a menu choice(1-4)\n")
    try:
        int(menu_input)
    except ValueError:
        print("\n!!! Input must be a number between 1-4 !!!\n")
        menu_input = display_menu()
    return int(menu_input)

# Executes Menu Actions from User-Input
# 1 = View Ham, #2 = View Spam, #3 = Input msg, #4 = Quit
def execute_menu(m_input, data_file, cv, clf):
    while m_input != 4:
        if m_input == 1:
            print(extract_ham(data_file))
            print("\nNote: Only the head & tail of our data(5 indexes) will be displayed\n")
            m_input = display_menu()
        elif m_input == 2:
            print(extract_spam(data_file))
            print("\nNote: Only the head & tail of our data(5 indexes) will be displayed\n")
            m_input = display_menu()
        elif m_input == 3:
            message = input("Type 'QUIT' to return back to main menu\nEnter a message: ")
            message_data = [message]

            if message == 'QUIT':
                m_input = display_menu()
            else:
                vect = cv.transform(message_data).toarray()
                predict_message = clf.predict(vect)
                print(predict_message)

        elif m_input == 4:
            exit(1)
        else:
            m_input = display_menu()


# Setting up Training & Testing Data from our data frame
# Returns our training data obj
def create_data_model(data_frame):
    x_train, x_test, y_train, y_test = train_test_split(data_frame["text"], data_frame["type"], test_size = 0.3, random_state = 37)
    
    # Calling & Initiating Training data constructor 
    data_set = Training_data(x_train, x_test, y_train, y_test)
    print(data_set.x_train, data_set.x_test, data_set.y_train, data_set.y_test)
    print('\n')
    print('Xtrain: ', len(data_set.x_train))
    print('Xtest: ', len(data_set.x_test))
    print('Ytrain: ', len(data_set.y_train))
    print('Ytest: ', len(data_set.y_test))
    return data_set

