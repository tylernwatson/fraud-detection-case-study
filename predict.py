import numpy as np
import pandas as pd
import pickle

'''
Add a column with predicted probability (using our pickled model) to the example data passed in.
'''

def make_prediction(filename, model_pickle):
    df = pd.read_json(filename, orient='index')
    first_row = df.iloc[0:1]
    first_row_num = first_row.select_dtypes(include=[np.number])
    first_row_num.fillna(value=0, inplace=True)
    with open(model_pickle, 'rb') as f:
        model = pickle.load(f)
    first_row_num['predict_proba'] = model.predict_proba(first_row_num)[:,1]
    temp_dict = first_row_num.to_dict()
    output_dict = {}
    for k,v in temp_dict.items():
        for v1, v2 in v.items():
            output_dict[k] = v2
    return output_dict
