# import predict
import requests
import pandas as pd
import json
import cPickle as pickle
import model
from db_wrapper import MongoWrapper

import predict

from flask import Flask, url_for
app = Flask(__name__)

with open('files/model.pkl', 'rb') as f:
    print('loading up that pickle')
    model = pickle.load(f)
db = MongoWrapper()

# @app.before_first_request
# def activate_job():
#     def run_job():
#         while True:
#             print("Run recurring task")
#             time.sleep(3)
#
#     thread = threading.Thread(target=run_job)
#     thread.start()

@app.route('/', methods=['GET'])
def api_root():
    return 'Put a route in. \n'

@app.route('/hello', methods=['GET'])
def hey_yall():
    return 'Hello World! \n'

@app.route('/score', methods=['GET'])
def score():
    prediction = predict.make_prediction('../files/example.json', model)
    return prediction
    return 'Probably Success'

# Needs to be POST
@app.route('/heroku_score', methods=['POST'])
def heroku_score():
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    response = requests.get(url).content
    son = json.loads(response)
    # df = pd.DataFrame([son])
    df = pd.io.json.json_normalize(son)
    NB_proba = model.predict_NB_proba(df['descriptions'])
    prediction = predict.make_prediction_df(df, model)
    db.insert_one_data(prediction)
    return 'OK'

if __name__ == '__main__':
    app.run()
