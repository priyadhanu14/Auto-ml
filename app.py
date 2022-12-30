#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template,redirect,request,url_for

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

@app.route('/dashboard',methods=['POST','GET'])
def dashboard():
    return render_template('dashboard.html', app_data=app_data)


@app.route('/service')
def service():
    return render_template('service.html', app_data=app_data)


@app.route('/contact')
def contact():
    return render_template('contact.html', app_data=app_data)


if __name__ == '__main__':
    app.run(debug=DEVELOPMENT_ENV)