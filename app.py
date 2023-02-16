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

# Global variable
enable_modal = 0
test_results = {'DECISION TREE':None,'LOGISTIC_REGRESSION':None,'NAIVE BAYES':None,'NEURAL NETWORK':None}
test_results_reg = MSE_TEST1  = {'LINEAR_REGRESSION':0,'LINEAR_REGRESSION_SKLEARN':0,'DECISION_TREE':0,'RANDOM_FOREST':0}
file_stats = {}

@app.route('/',methods=['POST','GET'])
def index():
    global enable_modal
    global test_results
    enable_modal = 0
    return render_template('index.html', app_data=app_data)


# Dashboard Route
@app.route('/dashboard',methods=['POST','GET'])
def dashboard():
    global enable_modal
    global test_results
    global test_results_reg
    global file_stats

    return render_template('dashboard.html', app_data=app_data,enable_modal=enable_modal,test_results= test_results,file_stats=file_stats,test_results_reg=test_results_reg)


@app.route('/submit_form',methods=['POST','GET'])
def submit_form():
    global enable_modal
    global test_results
    global file_stats
    global test_results_reg

    if request.method =='POST':

        # Global Flag
        enable_modal = 1

        # File Storage
        f = request.files['file']
        file_dir = os.path.join("auto_ml/temp_files",f.filename)
        f.save(file_dir)
        file_stats['file_name'] = f.filename

        # Train_Test_Split
        preprocess = Preprocess(file_dir)
        X_train,y_train,X_val,y_val = preprocess.preprocess(test_size=(1-(int(request.form.get("training_range"))/100)))

        # Size Params
        file_stats['type'] = {True:"Classification",False:"Regression"}[int(request.form.get("dataset_type")) == 1]
        file_stats['test_size'] = 100-int(request.form.get("training_range"))
        file_stats['training_size'] = int(request.form.get("training_range"))


        # Choose Option - Classification
        if(int(request.form.get("dataset_type")) == 1):
            print("classification")
            classification = Classification(X_train,y_train,X_val,y_val,X_val,y_val,X_val,y_val)
            classification.model()
            test_results = classification.tabulate()
        
        # Choose Option -  Regression
        elif(int(request.form.get("dataset_type")) == 2):
            regression = Regression(X_train,y_train,X_val,y_val,X_val,y_val,X_val,y_val)
            regression.model()
            test_results_reg = regression.tabulate()
            print("TEST RESULTS")
            print(test_results)

    return redirect(url_for('dashboard'))


@app.route('/service')
def service():
    return render_template('service.html', app_data=app_data)



if __name__ == '__main__':
    app.run(debug=DEVELOPMENT_ENV)