import pandas as pd
from sklearn.model_selection import train_test_split

from TextAnalyser.SentimentClassifier import SentimentClassifier


if __name__ == '__main__':

    df = pd.read_csv('./data/movie_sentiment.csv')
    X, y = df.text, df.label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    print(len(X_train), 'train documents')
    print(len(X_test), 'test documents')

    # Train and evaluate classifier
    sentiment_classifier = SentimentClassifier()
    pipeline = sentiment_classifier.train(X_train, y_train)
    sentiment_classifier.evaluate(pipeline, X_test, y_test)
    sentiment_classifier.save_pipeline(pipeline, model_path='./models/sentiment_classifier_pipeline.imb.pkl')
