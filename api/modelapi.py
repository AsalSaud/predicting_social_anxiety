import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from predicting_social_anxiety.logic.build_model import map_anxiety_level


model = joblib.load('cb2_model.joblib')
preprocessor = joblib.load('cb2_preprocessor.joblib')

app = FastAPI()


@app.get("/predict")
def predict(
    Age: int,
    Gender: str,
    Occupation: str,
    Sleep_Hours: float,
    Physical_Activity: float,
    Caffeine_Intake: float,
    Alcohol_Consumption: float,
    Smoking: str,
    Family_History: str,
    Stress_level: int,
    Hear_Rate: float,
    Breathing_Rate: float,
    Sweating_Level: int,
    Dizziness: str,
    Medication: str,
    Therapy_sessions: int,
    Recent_life_events: str,
    Diet_Quality: int
    ):

    input_data = {
        "Age": Age,
        "Gender": Gender,
        "Occupation": Occupation,
        "Sleep Hours": Sleep_Hours,
        "Physical Activity (hrs/week)": Physical_Activity,
        "Caffeine Intake (mg/day)": Caffeine_Intake,
        "Alcohol Consumption (drinks/week)": Alcohol_Consumption,
        "Smoking": Smoking,
        "Family History of Anxiety": Family_History,
        "Stress Level (1-10)": Stress_level,
        "Heart Rate (bpm)": Hear_Rate,
        "Breathing Rate (breaths/min)": Breathing_Rate,
        "Sweating Level (1-5)": Sweating_Level,
        "Dizziness": Dizziness,
        "Medication": Medication,
        "Therapy Sessions (per month)": Therapy_sessions,
        "Recent Major Life Event": Recent_life_events,
        "Diet Quality (1-10)": Diet_Quality
    }

    X = pd.DataFrame([input_data])

    X_ohe = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    X_ohe = pd.DataFrame(X_ohe, columns=feature_names)

    num_features = ['Age', 'Sleep Hours', 'Physical Activity (hrs/week)', 'Caffeine Intake (mg/day)', 'Alcohol Consumption (drinks/week)',
                    'Stress Level (1-10)', 'Heart Rate (bpm)', 'Breathing Rate (breaths/min)', 'Sweating Level (1-5)', 'Therapy Sessions (per month)', 'Diet Quality (1-10)']
    X_merged = pd.concat([X_ohe, X[num_features].reset_index(drop=True)], axis=1)


    prediction = np.round(model.predict(X_merged)[0])
    anxiety_level = map_anxiety_level(prediction)

    message = f"You got {prediction} which indicates a {anxiety_level}"
    return {"anxiety_prediction": message}
