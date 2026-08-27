from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)


# ============================================================
# LOAD BEST MODEL
# ============================================================

with open(
    "model/best_model.pkl",
    "rb"
) as file:

    model = pickle.load(file)


# ============================================================
# LOAD SCALER
# ============================================================

with open(
    "model/scaler.pkl",
    "rb"
) as file:

    scaler = pickle.load(file)


# ============================================================
# LOAD ACCURACY
# ============================================================

with open(
    "model/accuracy.pkl",
    "rb"
) as file:

    accuracy = pickle.load(file)


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

with open(
    "model/model_results.pkl",
    "rb"
) as file:

    model_results = pickle.load(file)


# ============================================================
# FIND BEST MODEL NAME
# ============================================================

best_model_name = model_results.iloc[0]["Model"]


# ============================================================
# PREDICTION HISTORY
# ============================================================

prediction_history = []


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)

def home():

    prediction = None

    probability = None

    no_purchase_probability = None

    age = None

    salary = None


    # ========================================================
    # PREDICTION
    # ========================================================

    if request.method == "POST":

        age = float(
            request.form["age"]
        )

        salary = float(
            request.form["salary"]
        )


        data = np.array([
            [age, salary]
        ])


        data_scaled = scaler.transform(
            data
        )


        result = model.predict(
            data_scaled
        )[0]


        probabilities = model.predict_proba(
            data_scaled
        )[0]


        no_purchase_probability = (
            probabilities[0] * 100
        )


        probability = (
            probabilities[1] * 100
        )


        if result == 1:

            prediction = (
                "Customer is likely to purchase."
            )

        else:

            prediction = (
                "Customer is unlikely to purchase."
            )


        prediction_history.insert(

            0,

            {

                "age":
                    int(age),

                "salary":
                    f"{salary:,.0f}",

                "probability":
                    f"{probability:.2f}%",

                "prediction":
                    prediction

            }

        )


        if len(
            prediction_history
        ) > 10:

            prediction_history.pop()


    return render_template(

    "index.html",

    prediction=prediction,

    probability=probability,

    no_purchase_probability=
        no_purchase_probability,

    accuracy=accuracy,

    best_model=best_model_name,

    history=prediction_history,

    model_results=
        model_results.to_dict(
            orient="records"
        )

)

# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )