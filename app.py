from flask import Flask, request, render_template
import pickle

flask_app = Flask(__name__)
model = pickle.load(open("model_pre.sav", "rb"))
model1=pickle.load(open("model_pre1.sav","rb"))

@flask_app.route("/")
def Home():
    result=''
    result1=''
    return render_template("index.html",**locals())

@flask_app.route("/aboutus.html")
def about():
    return render_template('aboutus.html')

@flask_app.route("/predict", methods = ["POST","GET"])
def predict():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker = float(request.form['smoker'])
    region = float(request.form['region'])
    result = model.predict([[age,sex, bmi, children, smoker,region]])[0]
    result1 = model1.predict([[age,sex, bmi, children, smoker,region]])[0]
    return render_template('index.html', **locals())

if __name__=='__main__':
    flask_app.run(debug=True)

