import pandas as pd
import numpy as np

def get_fraud_label():
    return 'fraud'

def create_response_label_from_json(json_file, fraudulent_types):
    '''
    Take a json file and acct_types defined as fraud, return DataFrame with "fraud" column created.
    Events associated with fraudulent acct_types are labeled 1, others labeled 0.

    :param json_file: json file of data
    :param fraudulent_types: list of strings defining which acct_types are fraudulent
    :return: DataFrame with "fraud" column added
    '''
    df = pd.read_json(json_file)
    df = create_response_label(df, fraudulent_types)
    return df

def create_response_label(df, fraudulent_types):
    '''
    Take a json file and acct_types defined as fraud, return DataFrame with "fraud" column created.
    Events associated with fraudulent acct_types are labeled 1, others labeled 0.

    :param json_file: json file of data
    :param fraudulent_types: list of strings defining which acct_types are fraudulent
    :return: DataFrame with "fraud" column added
    '''
    new_df = df.copy()
    new_df[get_fraud_label()] = new_df['acct_type'].isin(fraudulent_types).astype(int)
    return new_df
