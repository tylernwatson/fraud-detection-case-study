import numpy as np
import pandas as pd
import pickle

def make_prediction(filename, model_pickle):
    df = pd.read_json(filename, orient='index')
    first_row = df.iloc[0:1]
    first_row_num = first_row.select_dtypes(include=[np.number])
    first_row_num.fillna(value=0, inplace=True)
    with open(model_pickle, 'rb') as f:
        model = pickle.load(f)
    first_row_num['predict_proba'] = model.predict_proba(first_row_num)[:, 1]
    return first_row_num
