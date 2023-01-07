#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os


from flask import Flask, render_template,redirect,request,url_for
from auto_ml.preprocess import Preprocess
from auto_ml.classification import Classification
from auto_ml.regression import Regression

DEVELOPMENT_ENV  = True

app = Flask(__name__)

app_data = {
    "name":         "AUTO ML",
    "description":  "Automated ML Software",
    "author":       "Hari Priya D",
    "html_title":   "AUTO ML",
    "project_name": "AUTO ML",
    "keywords":     "flask, webapp, template, basic"
}


@app.route('/',methods=['POST','GET'])
def index():
    return render_template('index.html', app_data=app_data)

# Dashboard Route
@app.route('/dashboard',methods=['POST','GET'])
def dashboard():
    return render_template('dashboard.html', app_data=app_data)


@app.route('/submit_form',methods=['POST','GET'])
def submit_form():
    if request.method =='POST':
        print(request.form.get("dataset_type"))

        # File Storage
        f = request.files['file']
        print(f)
        file_dir = os.path.join("temp_files",f.filename)
        f.save(file_dir)

        # Train_Test_Split
        preprocess = Preprocess(file_dir,test_size=0.2)
        X_train,y_train,X_val,y_val = preprocess.preprocess()

        # Choose Option
        if(request.form.get("dataset_type") == 1):
            classification = Classification(X_train,y_train,X_val,y_val,X_train,y_train,X_val,y_val)
            classification.model()

    return redirect(url_for('dashboard'))


@app.route('/service')
def service():
    return render_template('service.html', app_data=app_data)



if __name__ == '__main__':
    app.run(debug=DEVELOPMENT_ENV)