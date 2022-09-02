import sys
import logging
import joblib
import json
import pandas as pd
from flask import Flask, request, Response, send_file


application = Flask(__name__)
logger = logging.getLogger("beer_project_logger")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


@application.route('/make_single_pred', methods=['POST'])
def predict_transaction():
    response = None
    try:
        logger.info("Receiving Information")
        file = request.files['single_data']
        prediction = make_prediction(file)
        print(prediction)
        response = Response(content_type='application/json', status = 200, response = json.dumps(prediction))
    except BaseException as error:
        error_content = {"Error": str(error)}
        logger.error("Error", json.dumps(error_content))
        response = Response(content_type='application/json', status = 500, response = json.dumps(error_content))
    return response

@application.route('/make_batch_pred', methods=['POST'])
def predict_transactions_batch():
    file = request.files['batch_data']
    predictions = make_predictions_batch(file)
    predictions.to_csv("prediction_batch.csv")
    return send_file('prediction_batch.csv')
    
def make_prediction(data):
    
    dataset = pd.read_csv(data)
    dataset_to_pred = dataset.drop(columns=['ID_code'], axis=1)
    data_treated = treat_data(dataset_to_pred)

    model = joblib.load('model.pickle')
    prediction = model.predict(data_treated)
    prediction_json = {
        "ID_code": dataset['ID_code'][0],
        "will_transaction_be_done":str(prediction[0])
    }

    return prediction_json

def make_predictions_batch(file):
    dataset = pd.read_csv(file)
    dataset_to_pred = dataset.drop(columns=['ID_code'], axis=1)

    dataset_treated = treat_data(dataset_to_pred)
    model = joblib.load('model.pickle')
    predictions = model.predict(dataset_treated)
    dataset_to_pred.reset_index(drop=True, inplace=True)
    dataset['will_transaction_be_done'] = predictions
    return dataset[['ID_code', 'will_transaction_be_done']]

def treat_data(data):
    scaler = joblib.load('scaler_obj')
    pca = joblib.load('pca_obj')
    data = scaler.transform(data)
    data = pca.transform(data)
    return data

if __name__ == '__main__':
     application.run(host='0.0.0.0', port=8080)