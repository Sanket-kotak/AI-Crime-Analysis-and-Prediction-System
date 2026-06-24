from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("../models/crime_predictor.pkl")

crime_mapping = {
    0: 'ARSON',
    1: 'ASSAULT',
    2: 'BATTERY',
    3: 'BURGLARY',
    5: 'CRIMINAL DAMAGE',
    8: 'DECEPTIVE PRACTICE',
    14: 'MOTOR VEHICLE THEFT',
    26: 'ROBBERY',
    29: 'THEFT',
    30: 'WEAPONS VIOLATION'
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        district = int(request.form["district"])
        month = int(request.form["month"])
        hour = int(request.form["hour"])

        features = np.array([[district, month, hour]])

        result = int(model.predict(features)[0])

        prediction = crime_mapping.get(
            result,
            f"Crime Code: {result}"
        )

    return render_template(
        "index.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)