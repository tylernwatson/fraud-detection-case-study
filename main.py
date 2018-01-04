import requests
import pandas as pd
import json
import threading
import time
import cPickle as pickle
import model2 as model
from db_wrapper import MongoWrapper

import predict

from flask import Flask, render_template
app = Flask(__name__)

'''
Flask app which scores new datapoints based on our model
as well as serving a list of predictions for review
'''

# Start by loading model one time for all predictions
with open('files/model.pkl', 'rb') as f:
    print('loading up that pickle')
    model = pickle.load(f)
db = MongoWrapper()

# Base Route
@app.route('/', methods=['GET'])
def api_root():
    return 'Put a route in. \n'

# Predictions in database
@app.route('/predictions', methods=['GET'])
def predictions():
    # Connect to mongo db and return entries broken up by high, medium, low risk
    entries = db.collection
    high = list(entries.find({'fraud_probability': {'$gte': .67}}))
    med = list(entries.find({'fraud_probability': {'$gt':  .33, '$lt': .67}}))
    low = list(entries.find({'fraud_probability': {'$lte': .33}}))
    return render_template('predictions.html', title='Fraud Predictions', high_data=high, med_data=med, low_data=low)

# Grabs an event from heroku and labels the probability that it is fraud
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

# Hit the heroku score endpoint every 60 seconds to populate db
def run_job():
    while True:
        print("Hitting http://galvanize-case-study-on-fraud.herokuapp.com/data_point")
        heroku_score()
        time.sleep(60)
thread = threading.Thread(target=run_job)
thread.start()

if __name__ == '__main__':
    app.run()
