from flask import Flask, render_template
from flask import request, redirect, make_response
# from flask.ext.responses import json_response
from flask_cors import CORS
import os
import os.path
import json
import joblib

import numpy as np
from sklearn.metrics import PredictionErrorDisplay
from preProcessNew import prediction

app = Flask(__name__, static_url_path = "", static_folder = "tmp")
CORS(app)

#initialize the data and model
obj = prediction()

@app.route("/")
def first_page():
    return render_template("index.html")
    

def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 11) 
    loaded_model = joblib.load(open("model.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0] 
     #return json_response(r, status_code=201)
@app.route('/result', methods = ['POST']) 
def result(): 
    print("In RESULTS:")
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        print(to_predict_list)
        to_predict_list = list(to_predict_list.values()) 
        # to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='slight'
        elif int(result)== 2: 
            prediction ='serious'
        elif int(result)== 3: 
            prediction ='fatal'            
        return render_template("result.html", prediction = prediction) 
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, threaded=True)