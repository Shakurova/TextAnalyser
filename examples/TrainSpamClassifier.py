import pandas as pd
from sklearn.model_selection import train_test_split

from TextAnalyser.SpamClassifier import SpamClassifier

if __name__ == "__main__":
    df = pd.read_csv('./data/lingspam_spam_ham_emails.csv')
    print('Dataset size: ', df.shape)

    # Split dataset to train and test
    X, y = df.text, df.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print(len(X_train), 'train docs')
    print(len(X_test), 'test docs')

    # Train and evaluate classifier
    spam_classifier = SpamClassifier()
    pipeline = spam_classifier.train(X_train, y_train)
    spam_classifier.evaluate(pipeline, X_test, y_test)
    spam_classifier.save_pipeline(pipeline, model_path='./models/spam_classifier_pipeline.lingspam.pkl')
