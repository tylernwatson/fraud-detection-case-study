"""
Module containing model fitting code for a web application that implements a
text classification model.

When run as a module, this will load a csv dataset, train a classification
model, and then pickle the resulting model object to disk.
"""
import cPickle as pickle
import pandas as pd
import numpy as np

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
        self._classifier = RandomForestClassifier()

    def fit(self, X, y):
        """Fit a text classifier model.

        Parameters
        ----------
        X: A numpy array or list of text fragments, to be used as predictors.
        y: A numpy array or python list of labels, to be used as responses.

        Returns
        -------
        self: The fit model object.
        """
        self = self._classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        return self._classifier.predict_proba(X)

    def predict(self, X):
        """Make predictions on new data."""
        return self._classifier.predict(X)

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        return self._classifier.score(X, y)

def prep_data(df):
    new_df = df.copy()
    fraud_labels = ['fraudster', 'fraudster_att', 'fraudster_event']
    new_df = EDA.create_response_label(new_df, fraud_labels)
    df_num = new_df.select_dtypes(include=[np.number])
    df_num.fillna(value=0, inplace=True)
    X = df_num.drop(EDA.get_fraud_label(), axis=1)
    y = df_num[EDA.get_fraud_label()]
    return (X, y)

def get_dataframe_from_zip(filename):
    #returns dataframe of zipped JSON file
    zip = ZipFile(filename)
    zip.extractall('files/')
    # file should be data.json
    return pd.read_json('files/data.{}'.format('json'))

if __name__ == '__main__':
    from model import Classifier
    df = get_dataframe_from_zip("files/data.zip")
    X, y = prep_data(df)
    modeler = Classifier()
    modeler.fit(X, y)
    score = modeler.score(X, y)
    print('Self score: ', score)
    with open('files/model.pkl', 'w') as f:
        pickle.dump(modeler, f)
