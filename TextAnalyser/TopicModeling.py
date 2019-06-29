import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords

from wordcloud import WordCloud, ImageColorGenerator

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.model_selection import train_test_split

import logging

lemm = WordNetLemmatizer()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class TopicModeling():

    def __init__(self):
        pass

    def load_model(self, model_name, vectorizer_name):
        self.vectorizer = pickle.load(open(vectorizer_name, 'rb'))
        self.feature_names = self.vectorizer.get_feature_names()
        self.model = pickle.load(open(model_name, 'rb'))
        self.components = self.model.components_

    def save_model(self, model_name, vectorizer_name):
        pickle.dump(self.model, open(model_name, 'wb'))
        pickle.dump(self.vectorizer, open(vectorizer_name, 'wb'))

    def clean_text(self, document):
        tokens = RegexpTokenizer(r'\w+').tokenize(document.lower())
        tokens_clean = [token for token in tokens if token not in spacy_stopwords]
        tokens_stemmed = [lemm.lemmatize(token) for token in tokens_clean]
        return ' '.join(tokens_stemmed)

    def train_model(self, train_docs, model_type, n_topics=100, n_features=1000):
        """ Train one of LSI, LDA or NMF models. """

        train_docs = [self.clean_text(doc) for doc in train_docs]

        if model_type == 'nmf':
            # Get features
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95,
                                               min_df=2,
                                               max_features=n_features,
                                               stop_words='english')
            tfidf = tfidf_vectorizer.fit_transform(train_docs)
            tfidf_feature_names = tfidf_vectorizer.get_feature_names()

            # Train the model
            nmf_model = NMF(n_components=n_topics,
                            random_state=1,
                            alpha=.1,
                            l1_ratio=.5,
                            init='nndsvd').fit(tfidf)
            nmf_topic_values = nmf_model.fit_transform(tfidf)
            nmf_components = nmf_model.components_
            print('NMF topic values: ', nmf_topic_values.shape)

            self.feature_names = tfidf_feature_names
            self.vectorizer = tfidf_vectorizer
            self.model = nmf_model
            self.components = nmf_components


        elif model_type == 'lda' or model_type == 'lsi':
            # Get features
            tf_vectorizer = CountVectorizer(max_df=0.95,
                                            min_df=2,
                                            max_features=n_features,
                                            stop_words='english')
            tf = tf_vectorizer.fit_transform(train_docs)
            tf_feature_names = tf_vectorizer.get_feature_names()

            # Train the model
            if model_type == 'lda':
                lda_model = LatentDirichletAllocation(n_components=n_topics,
                                                      max_iter=5,
                                                      learning_method='online',
                                                      learning_offset=50.,
                                                      random_state=0).fit(tf)
                lda_topic_values = lda_model.fit_transform(tf)
                lda_components = lda_model.components_
                print('LDA topic values: ', lda_topic_values.shape)

                self.feature_names = tf_feature_names
                self.vectorizer = tf_vectorizer
                self.model = lda_model
                self.components = lda_components


            elif model_type == 'lsi':
                lsi_model = TruncatedSVD(n_components=n_topics,
                                         algorithm='randomized',
                                         n_iter=100)
                lsi_topic_values = lsi_model.fit_transform(tf)
                lsi_components = lsi_model.components_
                print('LSI topic values: ', lsi_topic_values.shape)

                self.feature_names = tf_feature_names
                self.vectorizer = tf_vectorizer
                self.model = lsi_model
                self.components = lsi_components



    def get_top_keywords(self, topic_number, n_top_words=10):
        """ For a given topic number, returns top keywords. """
        top_keywords = [self.feature_names[i] for i in self.model.components_[topic_number].argsort()[:-n_top_words-1:-1]]
        return top_keywords

    def predict_topic(self, document, n_top_words=10):
        """ For a given document, returns the best topic number and top keywords. """

        document = [self.clean_text(document)]
        transformed_doc = self.vectorizer.transform(document)
        topic_values = self.model.transform(transformed_doc)

        best_topic_id = topic_values.argmax()
        top_keywords = self.get_top_keywords(best_topic_id, n_top_words)

        return best_topic_id, top_keywords

    def create_topics_df(self, n_top_words=20):
        " For all topics in the model prints top keywords and creates a dataframe. "

        topic_keywords = []
        for topic_id, topic in enumerate(self.components):
            top_keywords = self.get_top_keywords(topic_id, n_top_words=n_top_words)
            print('Topic #', topic_id, ': ', top_keywords)
            topic_keywords.append(top_keywords)

        df_topic_keywords = pd.DataFrame(topic_keywords)
        df_topic_keywords.columns = ['Word ' + str(i) for i in range(df_topic_keywords.shape[1])]
        df_topic_keywords.index = ['Topic ' + str(i) for i in range(df_topic_keywords.shape[0])]

        return df_topic_keywords

    def get_wordcloud(self, document, n_top_words=20):
        """ For a given document, generates word cloud """

        best_topic_id, top_keywords = self.predict_topic(document, n_top_words=n_top_words)
        print(best_topic_id, top_keywords)

        wordcloud = WordCloud(
                                  stopwords=spacy_stopwords,
                                  background_color='black',
                                  width=2500,
                                  height=1800
                                 ).generate(" ".join(top_keywords))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()

    def get_topic_modeling_vector(self, document):
        """ For a given document, returns topic probability distributions. """
        document = [self.clean_text(document)]
        transformed_doc = self.vectorizer.transform(document)
        topic_values = self.model.transform(transformed_doc)

        return topic_values
