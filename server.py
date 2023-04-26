import joblib
import os, sys, requests, json
from multiprocessing import Process
from __future__ import print_function
from flask import Flask, request, abort, logging

app = Flask(__name__)


path_classifier = 'svm_model.joblib'
path_dim_red = "ica_model.joblib"
CLASSIFIER = joblib.load(path_classifier)
DIM_RED_METHOD = joblib.load(path_dim_red)
class_index = ['im_Superficial-Intermediate',
 'im_Dyskeratotic',
 'im_Parabasal',
 'im_Metaplastic',
 'im_Koilocytotic']

def pipeline(row):
  row = row.reshape(1, -1)
  ica_representation = DIM_RED_METHOD.transform(row)
  index_predicted = CLASSIFIER.predict(ica_representation)[0]
  return {
      "class_predicted": class_index[index_predicted],
      "class_index": index_predicted
  }


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data as array of arrays from request
        data = request.json['data']
        row_data = np.array(data)

        # Convert image data into a vector
        vector = np.array(row_data).flatten()
        output = pipeline(vector)
        return json.dumps(str(output))

    except Exception as e:
        # Return error message if any exception occurs
        response = {'error': str(e)}
        return json.dumps(response)

def stop_server():
  global server
  if server is not None:
    server.terminate()
    server.join()

def start_server(run_thread):
  global server
  if run_thread:
    server = Process(target=app.run, kwargs={'host':local_ip,'port':80})
    server.start()
  else:
    app.run(host=local_ip, port=80)
