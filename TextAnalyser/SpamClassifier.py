import re
import string

import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.externals import joblib

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


class CustomCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def preprocess_text(self, text):

        text = re.sub(r"\\r", " ", text)
        text = re.sub(r"\\n", " ", text)
        text = re.sub(r'http\S+', 'xxURLxx', text)
        text = re.sub(r'www\S+', 'xxURLxx', text)
        text = re.sub('\S+@\S+', 'xxEMAILxx', text)
        text = re.sub('[0-9]+ ', 'xxNUMxx ', text)

        words = word_tokenize(text)
        words = [w for w in words if len(w) > 2]
        words = [w for w in words if w not in string.punctuation]
        words = [w for w in words if w not in stopwords]
        words = [word for word in words if word.isalnum()]
        words = [w.lower() if not w.isupper() else w.upper() for w in words]

        wnl = WordNetLemmatizer()
        words = [wnl.lemmatize(word) for word in words]

        return words

    def extract_features(self, text):
        features = []
        tokens = self.preprocess_text(text)
        features.append(sum([1 for word in tokens if word.isupper()]))
        features.append(sum([1 for word in tokens if word.isupper()]) / len(tokens))
        features.append(sum([1 for word in tokens if word.isalnum()]) / len(tokens))
        features.append(len(tokens))
        features.append(len(text))
        return features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X_custom = []
        for text in X:
            features = self.extract_features(text)
            X_custom.append(features)

        return X_custom


class SpamClassifier():
    def __init__(self):
        pass

    def save_pipeline(self, pipeline, model_path):
        joblib.dump(pipeline, model_path)

    def identity(self, words):
        return words

    def train(self, X_train, y_train, grid_search=False):
        mnb      = MultinomialNB()
        lin_svm  = LinearSVC()
        svm      = SVC()
        dtc      = DecisionTreeClassifier()
        lr       = LogisticRegression()
        rfc      = RandomForestClassifier(n_estimators=50, random_state=1)

        eclf = VotingClassifier(
            estimators=[
                ('mnb', mnb), ('dtc', dtc), ('lr', lr), ('rfc', rfc)],
                voting='soft'
        )

        pipeline = Pipeline([
                            ('features', FeatureUnion([
                                ('text_pipeline', Pipeline([
                                                ('CountVectorize', CountVectorizer(tokenizer=self.identity, lowercase=False)),
                                                ('tfidf', TfidfTransformer())])),
                                ('pos_count', CustomCounter())
                                                      ])),
                         ('clf', eclf)
                         ])
        if grid_search:
            param_grid = [{'clf__lr__C': [0.01, 0.1, 1, 10, 100],
                           'clf__mnb__alpha': [0.1, 1.0, 10.0, 100.0]
                           }]

            voting_model = GridSearchCV(pipeline,
                                        param_grid=param_grid,
                                        scoring='accuracy',
                                        cv=5)
            print(voting_model.best_params_)
            print(voting_model.best_score_)
            pipeline = voting_model

        pipeline.fit(X_train, y_train)

        return pipeline

    def predict(self, pipeline, X_test):
        return pipeline.predict(X_test)

    def evaluate(self, pipeline, X_test, y_test):
        y_pred = pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
