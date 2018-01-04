# import predict
import requests
import pandas as pd
import json
import cPickle as pickle
import model2 as model
from db_wrapper import MongoWrapper

import predict

from flask import Flask, url_for, render_template
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

# @app.route('/hello', methods=['GET'])
# def hey_yall():
#     return 'Hello World! \n'

# @app.route('/score', methods=['GET'])
# def score():
#     prediction = predict.make_prediction('../files/example.json', model)
#     return prediction
#     return 'Probably Success'

@app.route('/predictions', methods=['GET'])
def predictions():
    # Connect to mongo db and return entries broken up by high, medium, low risk
    entries = db.collection
    high = entries.find({'fraud_probability': {'$gte': .67}})
    med = entries.find({'fraud_probability': {'$gt':  .33, '$lt': .67}})
    low = entries.find({'fraud_probability': {'$lte': .33}})
    return render_template('predictions.html', title='Fraud Predictions', high_data=high, med_data=med, low_data=low)

# Needs to be POST
@app.route('/heroku_score', methods=['POST'])
def heroku_score():
    url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
    response = requests.get(url).content
    son = json.loads(response)
    # df = pd.DataFrame([son])
    df = pd.io.json.json_normalize(son)
    prediction = predict.make_prediction_df(df, model)
    db.insert_one_data(prediction)
    return 'OK'

if __name__ == '__main__':
    app.run()
