import re
import string
import pandas as pd

from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.externals import joblib

class TextNormalizer(BaseEstimator, TransformerMixin):
    """
    Do a bunch of fancy, specialized text normalization steps,
    like tokenization, part-of-speech tagging, lemmatization
    and stopwords removal.
    """
    ...
    def normalize(self, text):

        text = re.sub(r"\\r", " ", text)
        text = re.sub(r"\\n", " ", text)

        words = word_tokenize(text)
        words = [w for w in words if len(w) > 2]
        words = [w for w in words if w not in string.punctuation]
        words = [w for w in words if w not in stopwords]
        words = [word for word in words if word.isalpha()]
        words = [w.lower() if not w.isupper() else w.upper() for w in words]

        wnl = WordNetLemmatizer()
        words = [wnl.lemmatize(word) for word in words]

        return ' '.join(words)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        for document in documents:
            yield self.normalize(document)


class SentimentClassifier():
    def __init__(self):
        pass

    def save_pipeline(self, pipeline, model_path):
        joblib.dump(pipeline, model_path)

    def train(self, X_train, y_train):

        pipeline = Pipeline([
                          ('norm', TextNormalizer()),
                          ('vect', CountVectorizer(
                                        binary=False,
                                        ngram_range=(1, 2))
                                   ),
                          ('clfr', LogisticRegression(C=0.05))
                          ])

        pipeline.fit(X_train, y_train)

        return pipeline

    def predict(self, pipeline, X_test):
        return pipeline.predict(X_test)

    def evaluate(self, pipeline, X_test, y_test):
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print('Accuracy score:', accuracy_score(y_test, y_pred))
