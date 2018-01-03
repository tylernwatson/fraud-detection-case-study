import pandas as pd
import numpy as np

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