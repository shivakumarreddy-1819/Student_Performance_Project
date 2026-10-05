from flask import Flask, render_template, request, send_from_directory
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


app = Flask(__name__)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data = pd.read_csv("data/StudentsPerformance.csv")


# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

features = [
    "Gender",
    "race/ethnicity",
    "Parental level of education",
    "Lunch",
    "Test preparation course",
    "Reading score",
    "Writing score"
]

X = data[features]
y = data["Math score"]


# ==========================================
# 3. DEFINE FEATURE TYPES
# ==========================================

categorical_features = [
    "Gender",
    "race/ethnicity",
    "Parental level of education",
    "Lunch",
    "Test preparation course"
]

numerical_features = [
    "Reading score",
    "Writing score"
]


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ==========================================
# 5. CREATE MACHINE LEARNING PIPELINE
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ]
)


# ==========================================
# 6. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 8. MODEL EVALUATION
# ==========================================

test_predictions = model.predict(X_test)

r2 = r2_score(y_test, test_predictions)
mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))


print("\nFINAL MACHINE LEARNING MODEL")
print("=" * 40)
print("R² Score:", round(r2, 3))
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))


# ==========================================
# 9. WEB APPLICATION
# ==========================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    score_band = None

    if request.method == "POST":

        student = pd.DataFrame({
            "Gender": [request.form["gender"]],
            "race/ethnicity": [request.form["race_ethnicity"]],
            "Parental level of education": [
                request.form["parental_education"]
            ],
            "Lunch": [request.form["lunch"]],
            "Test preparation course": [
                request.form["test_preparation"]
            ],
            "Reading score": [
                float(request.form["reading_score"])
            ],
            "Writing score": [
                float(request.form["writing_score"])
            ]
        })

        # Predict Math Score
        prediction = model.predict(student)[0]

        # Keep prediction within 0-100
        prediction = max(0, min(100, prediction))

        prediction = round(prediction, 2)

        # Project-defined score band
        if prediction < 40:
            score_band = "Needs Improvement"

        elif prediction < 60:
            score_band = "Developing"

        elif prediction < 75:
            score_band = "Good"

        elif prediction < 90:
            score_band = "Very Good"

        else:
            score_band = "Excellent"

    return render_template(
        "index.html",
        prediction=prediction,
        score_band=score_band,
        r2=round(r2, 3),
        mae=round(mae, 3),
        rmse=round(rmse, 3)
    )


# ==========================================
# 10. RUN APPLICATION
# ==========================================
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/graphs/<path:filename>")
def graphs(filename):
    return send_from_directory("graphs", filename)

if __name__ == "__main__":
    app.run(debug=True)