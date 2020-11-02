# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:33:03 2020

@author: KUSH CHADHA
"""
from flask import Flask , request
import numpy as np
import pickle
import pandas as pd


app=Flask(__name__)


pickle_in = open("classifier.pkl","rb") # opened in read byte mode 
classifier=pickle.load(pickle_in) #loading my file 

@app.route('/')
def welcome():
    return "Welcome All"
    #app.run(host ='0.0.0.0')
    #app.run(host ='127.0.0.1')
@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance') # we use request.get to get the variables 
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted value is " + str(prediction)

#@app.route('/predict', methods=['POST'])
#def predict_note_file():
    #df_test = pd.read_csv(request.files.get("file"))
    #prediction=classifier.predict(df_test)
    #return "The predicted value for the csv is " + str(list(prediction))

    
    
if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)