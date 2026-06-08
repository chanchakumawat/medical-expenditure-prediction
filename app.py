from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

data = pd.read_csv("insurance.csv")

X = data[['age','bmi','children']]
y = data['charges']

model = LinearRegression()
model.fit(X,y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])

    prediction = model.predict([[age,bmi,children]])

    return render_template(
        'index.html',
        prediction_text=f"Estimated Medical Cost: ₹{prediction[0]:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
