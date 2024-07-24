import pandas as pd 
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
import pickle

appliction = Flask(__name__)
app = appliction

ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temprature = float(request.form.get('Temprature')) 
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temprature,RH,Ws,Rain,FFMC,DMC,DMC,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', results = result[0])

    else:
        return render_template("home.html")
if __name__=="__main__":
    app.run(host="0.0.0.0")
