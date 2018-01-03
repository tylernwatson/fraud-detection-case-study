"""
Module containing model fitting code for a web application that implements a
text classification model.

When run as a module, this will load a csv dataset, train a classification
model, and then pickle the resulting model object to disk.
"""
import cPickle as pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from zipfile import ZipFile

class TextClassifier(object):
    """A text classifier model:
        - Vectorize the raw text into features.
        - Fit a naive bayes model to the resulting features.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
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
        self = self._classifier.fit(self._vectorizer.fit_transform(X), y)

        return self

    def predict_proba(self, X):
        """Make probability predictions on new data."""
        return self._classifier.predict_proba(self._vectorizer.fit_transform(X))
        pass

    def predict(self, X):
        """Make predictions on new data."""
        return self._classifier.predict(self._vectorizer.fit_transform(X))
        pass

    def score(self, X, y):
        """Return a classification accuracy score on new data."""
        return self._classifier.score(self._vectorizer.fit_transform(X), y)
        pass

def create_response_label(json_file, fraudulent_types):
    '''
    Take a json file and acct_types defined as fraud, return DataFrame with "fraud" column created.
    Events associated with fraudulent acct_types are labeled 1, others labeled 0.
    :param json_file: json file of data
    :param fraudulent_types: list of strings defining which acct_types are fraudulent
    :return: DataFrame with "fraud" column added
    '''
    df = pd.read_json(json_file)
    df['fraud'] = df['acct_type'].isin(fraudulent_types).astype(int)

    return df

def prep_data(filename):
    """Load raw data from a file and return training data and responses.

    Parameters
    ----------
    filename: The path to a zip file containing data.

    Returns
    -------
    X: A numpy array containing the text fragments used for training.
    y: A numpy array containing labels, used for model response.
    """
    df = load_zip_to_pd('data.zip')
    df = create_response_label(json_file, fraudulent_types)
    y = df['section_name']
    return X, y
    pass


def load_zip_to_pd(filename):
    #returns dataframe of zipped JSON file
    zip = ZipFile(filename)
    zip.extractall()

    return pd.read_json('data.{}'.format('json'))

if __name__ == '__main__':
    X, y = get_data("data/articles.csv")
    tc = TextClassifier()
    tc.fit(X, y)
    with open('data/model.pkl', 'w') as f:
        pickle.dump(tc, f)
