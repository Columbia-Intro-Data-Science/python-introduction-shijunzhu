from pandas import DataFrame

from app import app
from flask import render_template, request , jsonify

import detectresult


@app.route('/',methods=['GET'])
@app.route('/heartdisease',methods=['GET'])
def heartdisease():
    return render_template('heartdisease.html')

@app.route('/detect',methods=['POST'])
def detect():
    fields = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
    values = []
    for field in fields:
        value = (request.form[field])
        value = float(value)
        values.append(value)
    r = detectresult.predict(DataFrame([values]))
    if r :
        return render_template('disease.html')
    else:
        return render_template('healthy.html')

