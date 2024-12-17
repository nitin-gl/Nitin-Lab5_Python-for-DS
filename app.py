from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['Car_Name']
    data2 = request.form['Fuel_Type']
    arr = np.array([[data1, data2]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)