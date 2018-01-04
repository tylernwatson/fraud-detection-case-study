"""
Module containing model fitting code for a web application that implements a
text classification model.
When run as a module, this will load a csv dataset, train a classification
model, and then pickle the resulting model object to disk.
"""
import cPickle as pickle
import pandas as pd
import numpy as np
from HTMLParser import HTMLParser
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from zipfile import ZipFile

import EDA

class Classifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit a naive bayes model to the resulting features.
    """

    def __init__(self):
        self._classifier = RandomForestClassifier(n_estimators=40, oob_score=True)
        self._vectorizer = TfidfVectorizer(stop_words='english',
                                            preprocessor=strip_tags,
                                            analyzer='word', max_df=.5)
        self._naive_bayes = MultinomialNB(alpha=.01)
        self.labels = ['approx_payout_date',
                         'body_length',
                         'channels',
                         'delivery_method',
                         'event_created',
                         'event_end',
                         'event_published',
                         'event_start',
                         'fb_published',
                         'gts',
                         'has_analytics',
                         'has_header',
                         'has_logo',
                         'name_length',
                         'num_order',
                         'num_payouts',
                         'object_id',
                         'org_facebook',
                         'org_twitter',
                         'sale_duration',
                         'sale_duration2',
                         'show_map',
                         'user_age',
                         'user_created',
                         'user_type',
                         'venue_latitude',
                         'venue_longitude',
                         'NLP_proba']
        self.nlp_label = 'NLP_proba'
    def get_columns(self):
        return self.labels

    def fit(self, X, y):
        """Fit a classifier model.
        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.
        Returns
        -------
        self: The fit model object.
        """

        self._classifier.fit(X, y)
        return self

    def fit_NB(self, X_description, y):
        self._naive_bayes.fit(self._vectorizer.fit_transform(X_description), y)
        return self._naive_bayes

    def predict_NB_proba(self, X_description):
        return self._naive_bayes.predict_proba(self._vectorizer
                                        .transform(X_description))[:,1]

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        return self._classifier.predict_proba(X)

    def predict(self, X):
        """Make predictions on new data."""
        return self._classifier.predict(X)

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        return self._classifier.score(X, y)

class MLStripper(HTMLParser):
    '''
    Used to strip HTML tags from event descriptions
    '''
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    '''
    Used to strip HTML tags from event descriptions
    '''
    s = MLStripper()
    s.feed(html)
    return re.sub('[^A-Za-z]+', ' ', s.get_data())

def prep_data(df):
    '''
    returns X_numeric, X_description, y with X_numeric as all numeric
    features from df, X_description as the descriptions of
    events and y as fraud label
    '''
    new_df = df.copy()
    fraud_labels = ['fraudster', 'fraudster_att', 'fraudster_event']
    new_df = EDA.create_response_label(new_df, fraud_labels)
    df_num = new_df.select_dtypes(include=[np.number])
    df_num.fillna(value=0, inplace=True)
    X_numeric = df_num.drop(EDA.get_fraud_label(), axis=1)
    X_description = new_df['description']
    y = df_num[EDA.get_fraud_label()]
    return (X_numeric, X_description, y)

def get_dataframe_from_zip(filename):
    '''
    returns dataframe of zipped JSON file
    '''
    zip = ZipFile(filename)
    zip.extractall('files/')
    # file should be data.json
    return pd.read_json('files/data.{}'.format('json'))

if __name__ == '__main__':
    from model2 import Classifier
    df = get_dataframe_from_zip("files/data.zip")
    X_numeric, X_description, y = prep_data(df)
    # X_numeric['probas'] = get_NB_probas(X_description, y)
    modeler = Classifier()
    modeler.fit_NB(X_description, y)
    X_numeric['probas'] = modeler.predict_NB_proba(X_description)
    modeler.fit(X_numeric, y)
    score = modeler.score(X_numeric, y)
    print('Self score: ', score)
    with open('files/model.pkl', 'w') as f:
        pickle.dump(modeler, f)
