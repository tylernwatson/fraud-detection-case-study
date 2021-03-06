import numpy as np
import pandas as pd
import pickle

output_column = 'fraud_probability'

'''
Add a column with predicted probability (using our pickled model) to the example data passed in.
'''
def make_prediction(filename, model):
    df = pd.read_json(filename, orient='index')
    first_row = df.iloc[0:1]
    first_row_num = first_row.select_dtypes(include=[np.number])
    first_row_num.fillna(value=0, inplace=True)
    first_row_num[output_column] = model.predict_proba(first_row_num)[:,1]
    temp_dict = first_row_num.to_dict()
    output_dict = {}
    for k,v in temp_dict.items():
        for v1, v2 in v.items():
            output_dict[k] = v2
    return output_dict

def make_prediction_df(df, model):
    first_row = df.copy()
    # first_row_num = first_row.select_dtypes(include=[np.number])
    first_row.fillna(value=0, inplace=True)
    NB_prob = model.predict_NB_proba(first_row['description'])
    first_row['NLP_proba'] = pd.Series(NB_prob, index=first_row.index)
    prepped_data = first_row[model.get_columns()]
    print('frn2, ', list(prepped_data.columns))
    first_row[output_column] = model.predict_proba(prepped_data)[:,1]
    temp_dict = first_row.to_dict()
    output_dict = {}
    for k,v in temp_dict.items():
        for v1, v2 in v.items():
            output_dict[k] = v2
    print(output_dict)
    return output_dict
