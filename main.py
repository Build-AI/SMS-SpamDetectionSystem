from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import naive_functions as nf

def main():
    data_file = nf.read_file()

    file_text = data_file['text']
    file_label = data_file['label']

    cv = CountVectorizer()
    file_text = cv.fit_transform(file_text)
    x_train, x_test, y_train, y_test = train_test_split(file_text, file_label, test_size=0.3, random_state=42)

    # Naive Bayes
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    clf.score(x_test, y_test)

    input = nf.display_menu()
    # Executes menu option from user input at display menu
    nf.execute_menu(input, data_file, cv, clf)

main()